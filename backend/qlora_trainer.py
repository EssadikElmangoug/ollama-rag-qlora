"""
QLoRA Fine-Tuning Module using Unsloth
Handles training on uploaded documents and model conversion
"""
import os
import json
import threading
import subprocess
import requests
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
    
    def convert_to_ollama(self, model_path, ollama_base_url=None):
        """Convert fine-tuned model to Ollama format"""
        try:
            # Get Ollama base URL from environment or parameter
            if ollama_base_url is None:
                ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            
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
            
            print(f"[Convert] Loading model from {model_path}...")
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
            
            print(f"[Convert] Merging LoRA adapters...")
            # Save merged model (16-bit for better compatibility)
            model.save_pretrained_merged(
                merged_model_path,
                tokenizer,
                save_method="merged_16bit",
            )
            
            # Step 2: Export to GGUF format (required by Ollama)
            print(f"[Convert] Exporting to GGUF format...")
            gguf_path = os.path.join(model_path, "gguf")
            os.makedirs(gguf_path, exist_ok=True)
            
            # Try to export to GGUF using Unsloth
            # Note: Unsloth's push_to_hub_gguf is for HuggingFace Hub
            # We need to use a different approach for local export
            try:
                # Check if we can use Unsloth's local export
                # First, try to save in a format that can be converted
                print(f"[Convert] Attempting GGUF export...")
                
                # Unsloth doesn't have direct local GGUF export in current version
                # We'll need to use llama.cpp or provide instructions
                # For now, save the merged model and provide conversion instructions
                print(f"[Convert] GGUF export requires llama.cpp. Providing conversion instructions...")
                
                # Create a note file with conversion instructions
                instructions_path = os.path.join(gguf_path, "CONVERSION_INSTRUCTIONS.txt")
                with open(instructions_path, 'w') as f:
                    f.write(f"""To convert this model to GGUF format for Ollama:

1. Install llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make

2. Convert the model:
   python convert.py {os.path.abspath(merged_model_path)} --outtype f16

3. Quantize (optional, for smaller size):
   ./quantize ggml-model-f16.gguf ggml-model-q4_k_m.gguf q4_k_m

4. Then use the GGUF file in the Ollama Modelfile.

Alternatively, you can use the merged model directly with the fine-tuned model loader (no Ollama conversion needed).
""")
                
                # For now, we'll use the merged model path and let Ollama handle it
                # if it supports PyTorch models, or provide clear error
                gguf_file = None
                raise Exception("GGUF export not yet implemented. Use the model directly without Ollama conversion, or convert manually using llama.cpp.")
                
            except Exception as e:
                # If GGUF export fails, we can't proceed with Ollama conversion
                # But we can still use the model directly
                error_msg = str(e)
                if "GGUF export not yet implemented" in error_msg:
                    # Provide a clear, user-friendly error message
                    raise Exception(
                        f"✅ GOOD NEWS: Your fine-tuned model is ready to use!\n\n"
                        f"RECOMMENDED: Use the model directly (no Ollama conversion needed):\n"
                        f"  • Go to the chat page\n"
                        f"  • Select '{model_name}' from the model dropdown\n"
                        f"  • Start chatting! The model works perfectly without Ollama.\n\n"
                        f"📦 Merged model saved at: {merged_model_path}\n\n"
                        f"⚠️  Ollama conversion requires GGUF format (manual conversion needed):\n"
                        f"  • Ollama cannot directly use PyTorch models\n"
                        f"  • Requires conversion using llama.cpp\n"
                        f"  • See instructions at: {instructions_path}\n\n"
                        f"💡 Tip: The direct model usage is faster and doesn't require conversion!"
                    )
                else:
                    raise Exception(f"GGUF export failed: {error_msg}")
            
            # Step 3: Create Ollama Modelfile
            modelfile_path = os.path.join(gguf_path, "Modelfile")
            
            # Find the GGUF file
            gguf_files = [f for f in os.listdir(gguf_path) if f.endswith('.gguf')]
            if not gguf_files:
                raise Exception(f"No GGUF files found in {gguf_path}")
            
            gguf_file = os.path.join(gguf_path, gguf_files[0])
            
            with open(modelfile_path, 'w') as f:
                # Use absolute path for the GGUF file
                abs_gguf_path = os.path.abspath(gguf_file)
                f.write(f"FROM {abs_gguf_path}\n")
                f.write("TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>\"\"\"\n")
                f.write("PARAMETER stop \"<|start_header_id|>\"\n")
                f.write("PARAMETER stop \"<|end_header_id|>\"\n")
                f.write("PARAMETER stop \"<|eot_id|>\"\n")
                f.write("PARAMETER temperature 0.7\n")
            
            # Step 4: Use Ollama API to create the model
            print(f"[Convert] Creating model in Ollama via API...")
            try:
                # Read the Modelfile content
                with open(modelfile_path, 'r') as f:
                    modelfile_content = f.read()
                
                # Use Ollama API to create the model
                response = requests.post(
                    f"{ollama_base_url}/api/create",
                    json={
                        "name": ollama_model_name,
                        "modelfile": modelfile_content
                    },
                    timeout=600  # 10 minutes timeout for large models
                )
                
                if response.status_code != 200:
                    error_msg = response.text or f"HTTP {response.status_code}"
                    raise Exception(f"Ollama API error: {error_msg}")
                
                print(f"[Convert] Model '{ollama_model_name}' created successfully in Ollama")
                
                # Update status
                self._update_trained_model_status(model_path, ollama_ready=True, ollama_name=ollama_model_name)
                
                return ollama_model_name
                
            except requests.exceptions.RequestException as e:
                # Fallback: provide instructions for manual import
                raise Exception(
                    f"Failed to create model via Ollama API at {ollama_base_url}. "
                    f"The Modelfile is saved at {modelfile_path}. "
                    f"You can manually import it using:\n"
                    f"  ollama create {ollama_model_name} -f {modelfile_path}\n"
                    f"API Error: {str(e)}"
                )
            
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

