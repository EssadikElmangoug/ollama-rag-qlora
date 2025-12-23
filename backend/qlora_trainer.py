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
    
    def train_model(self, base_model, model_name, lora_rank, max_steps, learning_rate):
        """Train a QLoRA model using Unsloth"""
        # Check if training is already in progress
        if self.training_status['status'] == 'training':
            raise Exception("Training is already in progress. Please wait for it to complete.")
        
        def training_worker():
            try:
                print(f"[Training Thread] Starting training for model: {model_name}")
                self.training_status['status'] = 'training'
                self.training_status['model_name'] = model_name
                self.training_status['progress'] = {'step': 0, 'total': max_steps, 'loss': 0}
                self.training_status['error'] = None
                print(f"[Training Thread] Status set to training")
                
                # Check if unsloth is available
                if not UNSLOTH_AVAILABLE:
                    raise Exception("Unsloth not installed. Please install: pip install 'unsloth[colab-new]'")
                
                # Import required libraries
                try:
                    import accelerate
                except ImportError:
                    raise Exception("accelerate is required. Install with: pip install accelerate")
                
                # Import unsloth components (already imported at module level, but import specific classes here)
                from unsloth import FastLanguageModel
                from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
                from transformers import TrainingArguments
                from trl import SFTTrainer
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
                
                # Load model
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                )
                
                # Configure LoRA
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=lora_rank,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    lora_alpha=lora_rank,
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                )
                
                # Setup chat template
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template="llama-3.1",
                )
                
                # Prepare dataset
                dataset = Dataset.from_list(training_data)
                dataset = standardize_sharegpt(dataset)
                
                # Format dataset
                def formatting_prompts_func(examples):
                    convos = examples["conversations"]
                    texts = [
                        tokenizer.apply_chat_template(
                            convo,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        for convo in convos
                    ]
                    return {"text": texts}
                
                dataset = dataset.map(formatting_prompts_func, batched=True)
                
                # Training arguments
                from unsloth import is_bfloat16_supported
                training_args = TrainingArguments(
                    output_dir=f"./qlora_models/{model_name}",
                    per_device_train_batch_size=2,
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
                )
                
                # Create trainer
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
                
                # Train on responses only
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
                )
                
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
                
                # Train
                trainer.train()
                
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

