"""
QLoRA Fine-Tuning Module using Unsloth
Handles training on uploaded documents and model conversion
"""
import os
import json
import threading
import subprocess
from datetime import datetime
from rag_processor import RAGProcessor

# Import psutil BEFORE unsloth to ensure it's available for Unsloth's compiled cache
try:
    import psutil
    # Make psutil available globally for Unsloth's compiled cache
    import sys
    if 'psutil' not in sys.modules:
        sys.modules['psutil'] = psutil
except ImportError:
    print("[Warning] psutil not installed. Please install with: pip install psutil>=5.9.0")
    psutil = None

# Import unsloth at module level (before transformers/peft) to ensure optimizations are applied
# This also triggers cache creation before Flask's reloader starts watching
try:
    import unsloth  # This triggers the patching and cache creation early
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

class QLoRATrainer:
    def __init__(self):
        self.training_status = {
            'status': 'idle',  # idle, training, completed, error
            'progress': {'step': 0, 'total': 0, 'loss': 0},
            'model_name': None,
            'error': None
        }
        self.training_thread = None
        self.rag_processor = RAGProcessor()
    
    def prepare_training_data_from_documents(self):
        """Extract text from uploaded documents and format for training"""
        try:
            # Get all documents from uploads folder
            uploads_folder = 'uploads'
            if not os.path.exists(uploads_folder):
                return []
            
            documents = []
            for filename in os.listdir(uploads_folder):
                filepath = os.path.join(uploads_folder, filename)
                if os.path.isfile(filepath):
                    try:
                        # Load document using RAG processor
                        docs = self.rag_processor._load_document(filepath)
                        for doc in docs:
                            documents.append({
                                'text': doc.page_content,
                                'source': filename
                            })
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        continue
            
            # Format as instruction-following dataset
            # Simple format: use document chunks as both instruction and response
            # In production, you'd want more sophisticated formatting
            training_data = []
            for doc in documents:
                # Create simple Q&A pairs from document chunks
                text = doc['text'].strip()
                if len(text) > 50:  # Only use substantial chunks
                    training_data.append({
                        "conversations": [
                            {"role": "user", "content": f"Tell me about: {text[:200]}..."},
                            {"role": "assistant", "content": text}
                        ]
                    })
            
            return training_data
    
        except Exception as e:
            raise Exception(f"Error preparing training data: {str(e)}")
    
    def _detect_model_type(self, base_model, model):
        """Detect model type from model name or config"""
        model_name_lower = base_model.lower()
        
        # Check model name patterns
        if 'gemma' in model_name_lower or 'functiongemma' in model_name_lower:
            return 'gemma'
        elif 'mistral' in model_name_lower:
            return 'mistral'
        elif 'qwen' in model_name_lower:
            return 'qwen'
        elif 'llama' in model_name_lower:
            return 'llama'
        elif 'phi' in model_name_lower:
            return 'phi'
        
        # Try to detect from model config
        try:
            config = model.config if hasattr(model, 'config') else None
            if config:
                model_type = getattr(config, 'model_type', '').lower()
                if model_type:
                    return model_type
        except:
            pass
        
        # Default to llama if can't detect
        return 'llama'
    
    def _get_chat_template_for_model(self, model_type):
        """Get appropriate chat template name for model type"""
        template_map = {
            'gemma': 'gemma',
            'mistral': 'mistral',
            'qwen': 'qwen',
            'llama': 'llama-3.1',
            'phi': 'phi',
        }
        return template_map.get(model_type, 'llama-3.1')
    
    def _detect_target_modules(self, model):
        """Auto-detect target modules from model architecture"""
        # Common target modules for different architectures
        common_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlp_modules = ["gate_proj", "up_proj", "down_proj"]
        
        # Get model's module names
        model_modules = set()
        try:
            for name, module in model.named_modules():
                model_modules.add(name.split('.')[-1])  # Get the last part of the module name
        except:
            pass
        
        # Build target modules list based on what's available
        target_modules = []
        
        # Add attention modules if available
        for module in common_modules:
            if module in model_modules:
                target_modules.append(module)
        
        # Add MLP modules if available
        for module in mlp_modules:
            if module in model_modules:
                target_modules.append(module)
        
        # If we found some modules, use them
        if target_modules:
            return target_modules
        
        # Fallback: try common alternatives
        alternative_modules = [
            "query", "key", "value", "dense",  # BERT-style
            "qkv", "out_proj",  # Some architectures
            "attention", "mlp",  # Generic
        ]
        
        for module in alternative_modules:
            if module in model_modules:
                target_modules.append(module)
        
        # Final fallback: use default if nothing found
        if not target_modules:
            print(f"[Training] Warning: Could not auto-detect target modules, using defaults")
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        return target_modules
    
    def _get_train_on_responses_parts(self, model_type):
        """Get instruction_part and response_part for train_on_responses_only based on model type"""
        # Llama models (Llama 3.1+)
        if model_type == 'llama':
            return (
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        # Gemma models
        elif model_type == 'gemma':
            return (
                "<start_of_turn>user\n",
                "<start_of_turn>model\n"
            )
        # Mistral models
        elif model_type == 'mistral':
            return (
                "[INST] ",
                " [/INST]"
            )
        # Qwen models
        elif model_type == 'qwen':
            return (
                "<|im_start|>user\n",
                "<|im_start|>assistant\n"
            )
        # Phi models
        elif model_type == 'phi':
            return (
                "<|user|>\n",
                "<|assistant|>\n"
            )
        # Unknown model type - return None to skip train_on_responses_only
        else:
            print(f"[Training] Unknown model type {model_type}, skipping train_on_responses_only")
            return None, None
    
    def train_model(self, base_model, model_name, lora_rank, max_steps, learning_rate):
        """Train a QLoRA model using Unsloth"""
        # Check if training is already in progress
        if self.training_status['status'] == 'training':
            raise Exception("Training is already in progress. Please wait for it to complete.")
        
        # Store parameters for the training worker to access
        training_base_model = base_model
        training_model_name = model_name
        training_lora_rank = lora_rank
        training_max_steps = max_steps
        training_learning_rate = learning_rate
        
        def training_worker():
            try:
                # Use the stored parameters
                base_model = training_base_model
                model_name = training_model_name
                lora_rank = training_lora_rank
                max_steps = training_max_steps
                learning_rate = training_learning_rate
                
                print(f"[Training Thread] Starting training for model: {model_name}")
                self.training_status['status'] = 'training'
                self.training_status['model_name'] = model_name
                self.training_status['progress'] = {'step': 0, 'total': max_steps, 'loss': 0}
                self.training_status['error'] = None
                print(f"[Training Thread] Status set to training")
                
                # Aggressive memory clearing before training
                import torch
                import gc
                import os
                
                # Set PyTorch memory allocation config to reduce fragmentation
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                if torch.cuda.is_available():
                    # Force garbage collection first
                    gc.collect()
                    
                    # Clear all caches multiple times
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Get memory info
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    allocated = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                    free = total_memory - reserved
                    
                    print(f"[Training] GPU Memory Status:")
                    print(f"  Total: {total_memory / 1024**3:.2f} GB")
                    print(f"  Allocated: {allocated / 1024**3:.2f} GB")
                    print(f"  Reserved: {reserved / 1024**3:.2f} GB")
                    print(f"  Free: {free / 1024**3:.2f} GB")
                    
                    # If memory is too full, wait and try again
                    if free < 2 * 1024**3:  # Less than 2GB free
                        print(f"[Training] Warning: Low GPU memory ({free / 1024**3:.2f} GB free)")
                        print(f"[Training] Attempting aggressive memory cleanup...")
                        
                        # Try to free reserved memory
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # Wait a bit for memory to be released
                        import time
                        time.sleep(2)
                        
                        # Check again
                        reserved = torch.cuda.memory_reserved(0)
                        free = total_memory - reserved
                        print(f"[Training] After cleanup - Free: {free / 1024**3:.2f} GB")
                        
                        if free < 1 * 1024**3:  # Still less than 1GB
                            raise Exception(
                                f"Insufficient GPU memory to start training. "
                                f"Only {free / 1024**3:.2f} GB free out of {total_memory / 1024**3:.2f} GB total.\n\n"
                                f"Please:\n"
                                f"1. Close other applications using the GPU\n"
                                f"2. Restart the server to clear all cached models\n"
                                f"3. Use a smaller model\n"
                                f"4. Reduce batch size or other memory-intensive settings"
                            )
                
                # Check if unsloth is available
                if not UNSLOTH_AVAILABLE:
                    raise Exception("Unsloth not installed. Please install: pip install 'unsloth[colab-new]'")
                
                # Import required libraries
                try:
                    import accelerate
                except ImportError:
                    raise Exception("accelerate is required. Install with: pip install accelerate")
                
                # Ensure psutil is available (required by Unsloth's compiled cache)
                try:
                    import psutil
                    # Make sure it's accessible
                    _ = psutil.cpu_count()
                    
                    # CRITICAL: Inject psutil into unsloth's compiled cache namespace
                    # The unsloth compiled cache file uses psutil but doesn't import it
                    import sys
                    import builtins
                    
                    # Inject into builtins so it's available globally (most reliable method)
                    builtins.psutil = psutil
                    print(f"[Training] Injected psutil into builtins namespace")
                    
                    # Also inject into sys.modules to ensure it's available
                    sys.modules['psutil'] = psutil
                    
                    # Try to inject into all unsloth-related modules that might be loaded
                    try:
                        for module_name in list(sys.modules.keys()):
                            if 'unsloth' in module_name.lower():
                                module = sys.modules[module_name]
                                if not hasattr(module, 'psutil'):
                                    module.psutil = psutil
                                    # Also inject into module's __dict__ for compiled modules
                                    if hasattr(module, '__dict__'):
                                        module.__dict__['psutil'] = psutil
                        print(f"[Training] Injected psutil into unsloth modules")
                    except Exception as e:
                        print(f"[Training] Warning: Could not inject psutil into unsloth modules: {e}")
                        
                except (ImportError, NameError):
                    raise Exception(
                        "psutil is required but not available. "
                        "Please install it with: pip install psutil>=5.9.0\n"
                        "If already installed, try: pip install --upgrade --force-reinstall psutil"
                    )
                
                # Import unsloth components (already imported at module level, but import specific classes here)
                from unsloth import FastLanguageModel
                from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
                from transformers import TrainingArguments
                
                # Import SFTTrainer - this will trigger loading of unsloth's compiled cache
                # We need to ensure psutil is available before this import
                from trl import SFTTrainer
                
                # After importing SFTTrainer, patch any newly loaded unsloth compiled cache modules
                try:
                    import psutil as psutil_module
                    import sys
                    # Re-inject into any modules that were just loaded
                    for module_name in list(sys.modules.keys()):
                        if 'unsloth' in module_name.lower():
                            module = sys.modules[module_name]
                            if not hasattr(module, 'psutil') or module.psutil is not psutil_module:
                                module.psutil = psutil_module
                                if hasattr(module, '__dict__'):
                                    module.__dict__['psutil'] = psutil_module
                    print(f"[Training] Re-patched psutil after SFTTrainer import")
                except Exception as e:
                    print(f"[Training] Warning: Could not re-patch unsloth after import: {e}")
                
                from transformers import DataCollatorForSeq2Seq
                from datasets import Dataset
                
                # Check if model is GGUF (inference-only, cannot be used for training)
                if 'gguf' in base_model.lower():
                    raise Exception(
                        f"GGUF models cannot be used for training! '{base_model}' is an inference-only model.\n\n"
                        "GGUF models are quantized formats for inference (used with llama.cpp), not for PyTorch training.\n\n"
                        "Please use a training-compatible model instead, such as:\n"
                        "• unsloth/Llama-3.2-3B-Instruct-bnb-4bit\n"
                        "• unsloth/Llama-3.1-8B-Instruct-bnb-4bit\n"
                        "• unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit\n"
                        "• Or any model WITHOUT '-GGUF' in the name"
                    )
                
                # Prepare training data
                training_data = self.prepare_training_data_from_documents()
                if not training_data:
                    raise Exception("No documents found. Please upload documents first.")
                
                # Load model with proper device handling
                import torch
                import gc
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[Training] Loading model on device: {device}")
                
                # Aggressive memory clearing before model load
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Get accurate memory info
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    reserved = torch.cuda.memory_reserved(0)
                    free_memory = total_memory - reserved
                    print(f"[Training] Available GPU memory before loading: {free_memory / 1024**3:.2f} GB")
                    
                    # If still too little memory, try one more aggressive cleanup
                    if free_memory < 3 * 1024**3:  # Less than 3GB
                        print(f"[Training] Performing final aggressive memory cleanup...")
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import time
                        time.sleep(1)
                        
                        reserved = torch.cuda.memory_reserved(0)
                        free_memory = total_memory - reserved
                        print(f"[Training] After final cleanup - Free: {free_memory / 1024**3:.2f} GB")
                
                # Try loading with 4-bit first, fallback to 8-bit or full precision if needed
                load_success = False
                model = None
                tokenizer = None
                
                # First attempt: 4-bit quantization
                try:
                    print(f"[Training] Attempting to load model with 4-bit quantization...")
                    
                    # Final memory check before loading
                    if torch.cuda.is_available():
                        reserved = torch.cuda.memory_reserved(0)
                        total = torch.cuda.get_device_properties(0).total_memory
                        free = total - reserved
                        print(f"[Training] Final memory check - Free: {free / 1024**3:.2f} GB, Reserved: {reserved / 1024**3:.2f} GB")
                        
                        if free < 1 * 1024**3:  # Less than 1GB free
                            raise Exception(
                                f"Cannot load model: Only {free / 1024**3:.2f} GB free GPU memory. "
                                f"PyTorch has reserved {reserved / 1024**3:.2f} GB indicating severe fragmentation.\n\n"
                                f"SOLUTION: Please restart the Flask server to completely clear GPU memory."
                            )
                    
                    # Explicitly avoid device_map to prevent meta tensor issues
                    # Use low_cpu_mem_usage to reduce memory footprint during loading
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=base_model,
                        max_seq_length=2048,
                        dtype=None,
                        load_in_4bit=True,
                        low_cpu_mem_usage=True,  # Critical for reducing memory usage
                        # Don't use device_map="auto" as it can cause meta tensor issues
                        # device_map=None,  # Explicitly set to None
                    )
                    
                    # Don't move quantized models - they're already on the correct device
                    # Quantized models (4-bit/8-bit) cannot be moved with .cuda()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Verify model is not meta tensor by checking lm_head
                    if hasattr(model, 'lm_head'):
                        # Try to access lm_head to ensure it's not meta
                        try:
                            # Force materialization of lm_head if it's meta
                            if hasattr(model.lm_head, 'weight'):
                                weight = model.lm_head.weight
                                # Check if it's a meta tensor
                                if hasattr(weight, 'device') and str(weight.device) == 'meta':
                                    raise NotImplementedError("lm_head is still a meta tensor")
                                # Try to access data
                                _ = weight.data
                            print(f"[Training] Model loaded successfully with 4-bit quantization")
                            load_success = True
                        except (NotImplementedError, RuntimeError) as e:
                            error_str = str(e).lower()
                            if "meta tensor" in error_str or "no data" in error_str or "cuda()" in error_str:
                                print(f"[Training] 4-bit loading issue: {e}, trying 8-bit...")
                                del model
                                del tokenizer
                                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            else:
                                raise
                    else:
                        load_success = True
                        
                except Exception as e:
                    error_str = str(e).lower()
                    if "meta tensor" in error_str or "no data" in error_str or "cuda()" in error_str or "8-bit" in error_str:
                        print(f"[Training] 4-bit loading failed: {e}, trying 8-bit...")
                    else:
                        print(f"[Training] 4-bit loading failed: {e}, trying 8-bit...")
                
                # Second attempt: 8-bit quantization if 4-bit failed
                if not load_success:
                    try:
                        print(f"[Training] Attempting to load model with 8-bit quantization...")
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            model_name=base_model,
                            max_seq_length=2048,
                            dtype=None,
                            load_in_4bit=False,  # Must set to False for 8-bit
                            load_in_8bit=True,
                        )
                        # Don't move quantized models - they're already on the correct device
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"[Training] Model loaded successfully with 8-bit quantization")
                        load_success = True
                    except Exception as e:
                        print(f"[Training] 8-bit loading failed: {e}, trying full precision...")
                
                # Third attempt: Full precision (no quantization) if both failed
                if not load_success:
                    print(f"[Training] Attempting to load model with full precision (no quantization)...")
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=base_model,
                        max_seq_length=2048,
                        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        load_in_4bit=False,
                        load_in_8bit=False,
                    )
                    # Move non-quantized model to device explicitly
                    if torch.cuda.is_available():
                        model = model.cuda()
                        torch.cuda.empty_cache()
                    else:
                        model = model.cpu()
                    print(f"[Training] Model loaded successfully with full precision")
                    load_success = True  # Mark as successful
                
                # Final verification: Ensure we have a model
                if not load_success or model is None or tokenizer is None:
                    raise Exception("Failed to load model with any quantization method")
                
                # Verify model has actual weights (not meta tensors)
                # Check lm_head specifically as that's where the error occurs
                try:
                    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                        lm_head_weight = model.lm_head.weight
                        # Check if it's a meta tensor
                        if hasattr(lm_head_weight, 'device'):
                            device_str = str(lm_head_weight.device)
                            if 'meta' in device_str:
                                # Try to materialize the tensor
                                print(f"[Training] Warning: lm_head is on meta device, attempting to materialize...")
                                # For quantized models, we might need to access through the base model
                                if hasattr(model, 'base_model'):
                                    base_model = model.base_model
                                    if hasattr(base_model, 'model') and hasattr(base_model.model, 'lm_head'):
                                        # Try to access the base model's lm_head
                                        _ = base_model.model.lm_head.weight.data
                                        print(f"[Training] Successfully materialized lm_head through base_model")
                                else:
                                    raise NotImplementedError("Cannot materialize meta tensor for this model architecture")
                        
                        # Try to access the data to ensure it's not meta
                        _ = lm_head_weight.data
                        print(f"[Training] lm_head verified - not meta tensor")
                    
                    # Verify other parameters
                    param = next(iter(model.parameters()))
                    if hasattr(param, 'device'):
                        device_str = str(param.device)
                        if 'meta' in device_str:
                            raise RuntimeError("Model still has meta tensors after loading")
                        print(f"[Training] Model weights verified - not meta tensors, device: {param.device}")
                except NotImplementedError as e:
                    if "meta tensor" in str(e).lower() or "no data" in str(e).lower():
                        print(f"[Training] Error: Model has meta tensors that cannot be materialized")
                        raise Exception(
                            f"Model loading failed: Model has meta tensors that cannot be materialized.\n\n"
                            f"This often happens with certain model architectures or when using device_map='auto'.\n\n"
                            f"Solutions:\n"
                            f"1. Try a different model (e.g., unsloth/Llama-3.2-3B-Instruct-bnb-4bit)\n"
                            f"2. Ensure you have sufficient GPU memory\n"
                            f"3. Try loading without quantization (will use more memory)\n"
                            f"4. Check if the model is compatible with Unsloth"
                        )
                    else:
                        raise
                except Exception as e:
                    print(f"[Training] Error verifying model weights: {e}")
                    # Don't fail here, let it try to train and see if it works
                    print(f"[Training] Warning: Could not fully verify model weights, proceeding anyway...")
                
                # Try to use model's native chat template, fallback to auto-detection
                try:
                    # Check if tokenizer has a chat template
                    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                        print(f"[Training] Using model's native chat template")
                    else:
                        # Try to auto-detect and set chat template
                        model_type = self._detect_model_type(base_model, model)
                        chat_template_name = self._get_chat_template_for_model(model_type)
                        print(f"[Training] No native template found, detected model type: {model_type}, using chat template: {chat_template_name}")
                        tokenizer = get_chat_template(
                            tokenizer,
                            chat_template=chat_template_name,
                        )
                except Exception as e:
                    print(f"[Training] Warning: Could not set chat template: {e}. Using model's default.")
                
                # Clear cache before applying LoRA to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Auto-detect target modules from model architecture
                target_modules = self._detect_target_modules(model)
                print(f"[Training] Using target modules: {target_modules}")
                
                # Configure LoRA with auto-detected target modules
                # This should properly initialize the model if it wasn't before
                try:
                    model = FastLanguageModel.get_peft_model(
                        model,
                        r=lora_rank,
                        target_modules=target_modules,
                        lora_alpha=lora_rank,
                        lora_dropout=0,
                        bias="none",
                        use_gradient_checkpointing="unsloth",
                        random_state=3407,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    # Clear cache and try again
                    print(f"[Training] Out of memory when applying LoRA, clearing cache and retrying...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    # Try with lower rank
                    print(f"[Training] Retrying with lower LoRA rank: {lora_rank // 2}")
                    model = FastLanguageModel.get_peft_model(
                        model,
                        r=max(8, lora_rank // 2),  # Reduce rank, minimum 8
                        target_modules=target_modules,
                        lora_alpha=max(8, lora_rank // 2),
                        lora_dropout=0,
                        bias="none",
                        use_gradient_checkpointing="unsloth",
                        random_state=3407,
                    )
                    print(f"[Training] Successfully applied LoRA with reduced rank")
                
                # Clear cache after LoRA application
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Ensure model is ready for training (not in meta state)
                model.train()
                print(f"[Training] Model set to training mode")
                
                # Prepare dataset
                dataset = Dataset.from_list(training_data)
                dataset = standardize_sharegpt(dataset)
                
                # Format dataset with error handling
                # When batched=True, examples is a dict where each value is a list
                def formatting_prompts_func(examples):
                    # Get conversations list (each element is a conversation which is a list of messages)
                    if "conversations" not in examples:
                        raise ValueError("Dataset must have 'conversations' field after standardize_sharegpt")
                    
                    convos_list = examples["conversations"]  # List of conversations
                    texts = []
                    
                    for convo in convos_list:
                        try:
                            # convo should be a list of message dicts: [{"role": "user", "content": "..."}, ...]
                            if not isinstance(convo, list):
                                print(f"[Training] Warning: Conversation is not a list: {type(convo)}")
                                convo = []
                            
                            # Try to use chat template
                            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                                text = tokenizer.apply_chat_template(
                                    convo,
                                    tokenize=False,
                                    add_generation_prompt=False
                                )
                            else:
                                # Fallback: simple formatting
                                text = ""
                                for msg in convo:
                                    if isinstance(msg, dict):
                                        role = msg.get("role", "")
                                        content = msg.get("content", "")
                                        if role == "user":
                                            text += f"User: {content}\n\n"
                                        elif role == "assistant":
                                            text += f"Assistant: {content}\n\n"
                            texts.append(text)
                        except Exception as e:
                            # If chat template fails, use simple fallback
                            print(f"[Training] Warning: Chat template failed for one example: {e}. Using fallback.")
                            text = ""
                            if isinstance(convo, list):
                                for msg in convo:
                                    if isinstance(msg, dict):
                                        role = msg.get("role", "")
                                        content = msg.get("content", "")
                                        if role == "user":
                                            text += f"User: {content}\n\n"
                                        elif role == "assistant":
                                            text += f"Assistant: {content}\n\n"
                            texts.append(text)
                    
                    return {"text": texts}
                
                # Format the dataset
                print(f"[Training] Dataset columns before formatting: {dataset.column_names}")
                dataset = dataset.map(formatting_prompts_func, batched=True)
                
                # After mapping, ensure dataset only has 'text' column
                # Get current column names after mapping
                current_columns = dataset.column_names
                print(f"[Training] Dataset columns after mapping: {current_columns}")
                
                # Always recreate dataset with only 'text' column to ensure clean format
                # This is the safest approach to avoid any residual fields
                try:
                    texts = [item["text"] for item in dataset]
                    dataset = Dataset.from_dict({"text": texts})
                    print(f"[Training] Recreated dataset with only 'text' column")
                except Exception as e:
                    print(f"[Training] Error recreating dataset: {e}")
                    # Fallback: try to remove columns
                    columns_to_remove = [col for col in current_columns if col != "text"]
                    if columns_to_remove:
                        print(f"[Training] Attempting to remove columns: {columns_to_remove}")
                        dataset = dataset.remove_columns(columns_to_remove)
                
                # Final verification
                final_columns = dataset.column_names
                print(f"[Training] Final dataset columns: {final_columns}")
                if "text" not in final_columns:
                    raise Exception("Dataset must have 'text' column after formatting")
                
                if len(final_columns) > 1:
                    print(f"[Training] WARNING: Dataset has extra columns: {final_columns}")
                    # Force cleanup
                    texts = [item["text"] for item in dataset]
                    dataset = Dataset.from_dict({"text": texts})
                    print(f"[Training] Force cleaned dataset - now only has: {dataset.column_names}")
                
                # Verify sample format
                if len(dataset) > 0:
                    sample = dataset[0]
                    sample_keys = list(sample.keys())
                    print(f"[Training] Sample entry keys: {sample_keys}")
                    if "text" in sample:
                        text_preview = sample["text"][:100] if len(sample["text"]) > 100 else sample["text"]
                        print(f"[Training] Sample text preview: {text_preview}...")
                
                # Training arguments with memory optimization
                from unsloth import is_bfloat16_supported
                
                # Adjust batch size based on available memory
                per_device_batch_size = 2
                if torch.cuda.is_available():
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    if free_memory < 5 * 1024**3:  # Less than 5GB free
                        per_device_batch_size = 1
                        print(f"[Training] Low GPU memory detected, using batch size 1")
                
                training_args = TrainingArguments(
                    output_dir=f"./qlora_models/{model_name}",
                    per_device_train_batch_size=per_device_batch_size,
                    gradient_accumulation_steps=4,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    warmup_steps=10,
                    lr_scheduler_type="linear",
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    logging_steps=10,
                    seed=3407,
                    # Memory optimization settings
                    dataloader_pin_memory=False,  # Reduce memory usage
                    remove_unused_columns=True,  # Remove columns not used by the model
                )
                
                # Create trainer with error handling for meta tensor issues
                # CRITICAL: Ensure psutil is available in all namespaces before creating SFTTrainer
                # The unsloth compiled cache uses psutil but doesn't import it
                try:
                    import psutil
                    import sys
                    import builtins
                    import importlib
                    
                    # Ensure psutil is in builtins (most reliable)
                    builtins.psutil = psutil
                    
                    # Inject into all currently loaded unsloth modules
                    for module_name, module in list(sys.modules.items()):
                        if 'unsloth' in module_name.lower():
                            # Inject into module globals
                            if hasattr(module, '__dict__'):
                                module.__dict__['psutil'] = psutil
                            # Also set as attribute
                            if not hasattr(module, 'psutil'):
                                setattr(module, 'psutil', psutil)
                            # Patch __builtins__ if it exists
                            if hasattr(module, '__builtins__'):
                                if isinstance(module.__builtins__, dict):
                                    module.__builtins__['psutil'] = psutil
                                elif hasattr(module.__builtins__, '__dict__'):
                                    module.__builtins__.__dict__['psutil'] = psutil
                    
                    print(f"[Training] Final psutil injection before SFTTrainer creation")
                except Exception as e:
                    print(f"[Training] Warning: Could not inject psutil before SFTTrainer: {e}")
                
                # Final dataset verification before creating trainer
                print(f"[Training] Final dataset check before trainer creation:")
                print(f"  - Columns: {dataset.column_names}")
                print(f"  - Dataset size: {len(dataset)}")
                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f"  - Sample keys: {list(sample.keys())}")
                    if "conversations" in sample:
                        print(f"[Training] CRITICAL: Dataset still has 'conversations' field! Forcing cleanup...")
                        texts = [item["text"] for item in dataset]
                        dataset = Dataset.from_dict({"text": texts})
                        print(f"[Training] Dataset cleaned. New columns: {dataset.column_names}")
                
                try:
                    trainer = SFTTrainer(
                        model=model,
                        train_dataset=dataset,
                        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                        tokenizer=tokenizer,
                        dataset_text_field="text",
                        max_seq_length=2048,
                        dataset_num_proc=2,
                        packing=False,
                        args=training_args,
                    )
                except NameError as e:
                    # Catch NameError for psutil and retry with aggressive patching
                    if 'psutil' in str(e).lower():
                        print(f"[Training] NameError for psutil detected, applying aggressive patch and retrying...")
                        import psutil
                        import sys
                        import builtins
                        
                        # Aggressive injection into all possible namespaces
                        builtins.psutil = psutil
                        sys.modules['psutil'] = psutil
                        
                        # Find and patch the unsloth compiled cache module
                        for module_name, module in list(sys.modules.items()):
                            if 'unsloth' in module_name.lower() or 'sft' in module_name.lower():
                                try:
                                    if hasattr(module, '__dict__'):
                                        module.__dict__['psutil'] = psutil
                                    setattr(module, 'psutil', psutil)
                                    # Try to patch globals if it's a function/class
                                    if hasattr(module, '__globals__'):
                                        module.__globals__['psutil'] = psutil
                                except:
                                    pass
                        
                        # Retry creating the trainer
                        trainer = SFTTrainer(
                            model=model,
                            train_dataset=dataset,
                            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                            tokenizer=tokenizer,
                            dataset_text_field="text",
                            max_seq_length=2048,
                            dataset_num_proc=2,
                            packing=False,
                            args=training_args,
                        )
                        print(f"[Training] Successfully created trainer after psutil patch")
                    else:
                        raise
                except (NotImplementedError, RuntimeError) as e:
                    if "meta tensor" in str(e).lower() or "no data" in str(e).lower():
                        print(f"[Training] Error creating trainer due to meta tensors: {e}")
                        print(f"[Training] Attempting to reload model without quantization as fallback...")
                        
                        # Try reloading without quantization
                        del model
                        del tokenizer
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # Reload without quantization
                        print(f"[Training] Reloading model without quantization...")
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            model_name=base_model,
                            max_seq_length=2048,
                            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            load_in_4bit=False,
                        )
                        
                        # Move to device
                        if torch.cuda.is_available():
                            model = model.cuda()
                        else:
                            model = model.cpu()
                        
                        # Re-apply chat template and LoRA
                        try:
                            model_type = self._detect_model_type(base_model, model)
                            chat_template_name = self._get_chat_template_for_model(model_type)
                            tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
                        except:
                            pass
                        
                        target_modules = self._detect_target_modules(model)
                        model = FastLanguageModel.get_peft_model(
                            model,
                            r=lora_rank,
                            target_modules=target_modules,
                            lora_alpha=lora_rank,
                            lora_dropout=0,
                            bias="none",
                            use_gradient_checkpointing="unsloth",
                            random_state=3407,
                        )
                        model.train()
                        
                        # Recreate trainer
                        trainer = SFTTrainer(
                            model=model,
                            train_dataset=dataset,
                            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                            tokenizer=tokenizer,
                            dataset_text_field="text",
                            max_seq_length=2048,
                            dataset_num_proc=2,
                            packing=False,
                            args=training_args,
                        )
                        print(f"[Training] Successfully created trainer without quantization")
                    else:
                        raise
                
                # Try to use train_on_responses_only if we can detect the format
                # This is optional and will fall back to standard training if it fails
                try:
                    model_type = self._detect_model_type(base_model, model)
                    instruction_part, response_part = self._get_train_on_responses_parts(model_type)
                    
                    if instruction_part and response_part:
                        print(f"[Training] Using train_on_responses_only with detected format for {model_type}")
                        trainer = train_on_responses_only(
                            trainer,
                            instruction_part=instruction_part,
                            response_part=response_part,
                        )
                    else:
                        print(f"[Training] Skipping train_on_responses_only (using standard training)")
                except Exception as e:
                    print(f"[Training] Warning: Could not apply train_on_responses_only: {e}. Using standard training.")
                
                # Custom callback to update progress
                from transformers import TrainerCallback
                
                class ProgressCallback(TrainerCallback):
                    def __init__(self, trainer_instance, total_steps):
                        self.trainer = trainer_instance
                        self.total_steps = total_steps
                    
                    def on_train_begin(self, args, state, control, **kwargs):
                        """Called at the beginning of training"""
                        try:
                            # Set total steps from training args or state
                            if hasattr(state, 'max_steps') and state.max_steps:
                                self.trainer.training_status['progress']['total'] = int(state.max_steps)
                            else:
                                self.trainer.training_status['progress']['total'] = self.total_steps
                            print(f"[ProgressCallback] Training started. Total steps: {self.trainer.training_status['progress']['total']}")
                        except Exception as e:
                            print(f"Error in on_train_begin: {e}")
                    
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        """Called when logging occurs"""
                        try:
                            if logs and 'loss' in logs:
                                self.trainer.training_status['progress']['loss'] = float(logs['loss'])
                            if hasattr(state, 'global_step'):
                                self.trainer.training_status['progress']['step'] = int(state.global_step)
                        except Exception as e:
                            print(f"Error updating progress: {e}")
                    
                    def on_train_end(self, args, state, control, **kwargs):
                        """Called at the end of training"""
                        try:
                            if hasattr(state, 'global_step'):
                                self.trainer.training_status['progress']['step'] = int(state.global_step)
                            print(f"[ProgressCallback] Training completed. Final step: {self.trainer.training_status['progress']['step']}")
                        except Exception as e:
                            print(f"Error in on_train_end: {e}")
                
                trainer.add_callback(ProgressCallback(self, max_steps))
                
                # Train with better error handling
                try:
                    trainer.train()
                except Exception as train_error:
                    # Check if it's a train_on_responses_only error
                    error_str = str(train_error).lower()
                    if "all labels" in error_str and "-100" in error_str:
                        print(f"[Training] Error with train_on_responses_only, retrying without it...")
                        # Recreate trainer without train_on_responses_only
                        trainer = SFTTrainer(
                            model=model,
                            train_dataset=dataset,
                            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                            tokenizer=tokenizer,
                            dataset_text_field="text",
                            max_seq_length=2048,
                            dataset_num_proc=2,
                            packing=False,
                            args=training_args,
                        )
                        trainer.train()
                    else:
                        raise train_error
                
                # Save model (LoRA adapters only)
                model.save_pretrained(f"./qlora_models/{model_name}")
                tokenizer.save_pretrained(f"./qlora_models/{model_name}")
                
                # Save training config for later reference
                config = {
                    'base_model': base_model,
                    'lora_rank': lora_rank,
                    'max_steps': max_steps,
                    'learning_rate': learning_rate,
                    'trained_at': datetime.now().isoformat()
                }
                with open(f"./qlora_models/{model_name}/training_config.json", 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.training_status['status'] = 'completed'
                self.training_status['progress']['step'] = max_steps
                
            except Exception as e:
                error_msg = str(e)
                self.training_status['status'] = 'error'
                self.training_status['error'] = error_msg
                # Keep existing progress if available
                if 'progress' not in self.training_status:
                    self.training_status['progress'] = {'step': 0, 'total': 0, 'loss': 0}
                print(f"Training error: {error_msg}")
                import traceback
                traceback.print_exc()
        
        # Start training in background thread
        # Use daemon=False to ensure thread completes even if main thread exits
        self.training_thread = threading.Thread(target=training_worker, daemon=False)
        self.training_thread.start()
        print(f"Training thread started for model: {model_name}")
        print(f"Thread ID: {self.training_thread.ident}, Status: {self.training_status['status']}")
    
    def convert_to_ollama(self, model_path):
        """Convert fine-tuned model to Ollama format"""
        try:
            # Check if model exists
            if not os.path.exists(model_path):
                raise Exception(f"Model not found at {model_path}")
            
            # Import Unsloth for merging
            try:
                from unsloth import FastLanguageModel
            except ImportError:
                raise Exception("Unsloth not installed. Cannot merge LoRA adapters.")
            
            model_name = os.path.basename(model_path)
            ollama_model_name = model_name.replace('_', '-').lower()
            
            # Step 1: Load and merge LoRA adapters with base model
            # First, try to get base model from training config
            config_path = os.path.join(model_path, "training_config.json")
            base_model = 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit'  # Default
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    training_config = json.load(f)
                    base_model = training_config.get('base_model', base_model)
            else:
                # Fallback to adapter config
                adapter_config_path = os.path.join(model_path, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        base_model = adapter_config.get('base_model_name_or_path', base_model)
            
            # Load the model with adapters
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Merge adapters into base model
            merged_model_path = os.path.join(model_path, "merged_model")
            os.makedirs(merged_model_path, exist_ok=True)
            
            # Save merged model (16-bit for better compatibility)
            model.save_pretrained_merged(
                merged_model_path,
                tokenizer,
                save_method="merged_16bit",
            )
            
            # Step 2: Create Ollama Modelfile
            modelfile_path = os.path.join(merged_model_path, "Modelfile")
            
            with open(modelfile_path, 'w') as f:
                f.write(f"FROM {merged_model_path}\n")
                f.write("TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>\"\"\"\n")
                f.write("PARAMETER stop \"<|start_header_id|>\"\n")
                f.write("PARAMETER stop \"<|end_header_id|>\"\n")
                f.write("PARAMETER stop \"<|eot_id|>\"\n")
                f.write("PARAMETER temperature 0.7\n")
            
            # Step 3: Use Ollama to import the model
            # Note: Ollama needs the model in GGUF format
            # For now, we'll create the modelfile and let user know they may need to convert manually
            # In production, you'd use llama.cpp to convert to GGUF first
            
            # Try to create model with Ollama (may fail if not in GGUF format)
            result = subprocess.run(
                ['ollama', 'create', ollama_model_name, '-f', modelfile_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                # If direct import fails, provide instructions
                raise Exception(
                    f"Direct Ollama import failed. The merged model is saved at {merged_model_path}. "
                    f"You may need to convert it to GGUF format using llama.cpp first. "
                    f"Error: {result.stderr}"
                )
            
            return ollama_model_name
            
        except Exception as e:
            raise Exception(f"Error converting to Ollama: {str(e)}")
    
    def list_trained_models(self):
        """List all trained models"""
        models = []
        qlora_dir = "./qlora_models"
        
        if os.path.exists(qlora_dir):
            for model_name in os.listdir(qlora_dir):
                model_path = os.path.join(qlora_dir, model_name)
                if os.path.isdir(model_path):
                    # Check if converted to Ollama
                    ollama_ready = False
                    try:
                        result = subprocess.run(
                            ['ollama', 'list'],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if model_name.lower().replace('_', '-') in result.stdout:
                            ollama_ready = True
                    except:
                        pass
                    
                    models.append({
                        'name': model_name,
                        'path': model_path,
                        'ollama_ready': ollama_ready
                    })
        
        return models
    
    def get_status(self):
        """Get current training status"""
        return self.training_status

