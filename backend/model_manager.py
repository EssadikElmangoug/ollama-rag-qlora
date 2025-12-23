"""
Model Manager for Hugging Face Models
Handles model installation, listing, and loading
"""
import os
import json
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

class ModelManager:
    def __init__(self, models_dir="./hf_models"):
        """
        Initialize Model Manager
        
        Args:
            models_dir: Directory to store downloaded models
        """
        self.models_dir = models_dir
        self.models_index_file = os.path.join(models_dir, "models_index.json")
        os.makedirs(models_dir, exist_ok=True)
        
        # Load models index
        self.models_index = self._load_models_index()
    
    def _load_models_index(self):
        """Load the models index file"""
        if os.path.exists(self.models_index_file):
            try:
                with open(self.models_index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_models_index(self):
        """Save the models index file"""
        with open(self.models_index_file, 'w') as f:
            json.dump(self.models_index, f, indent=2)
    
    def install_model(self, model_name, load_in_4bit=True):
        """
        Install/download a model from Hugging Face
        
        Args:
            model_name: Hugging Face model identifier (e.g., "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
            load_in_4bit: Whether to use 4-bit quantization (for Unsloth models)
            
        Returns:
            dict with installation status
        """
        try:
            # Check if model is already installed
            if model_name in self.models_index:
                return {
                    "success": True,
                    "message": f"Model {model_name} is already installed",
                    "model_name": model_name
                }
            
            print(f"[Model Manager] Installing model: {model_name}")
            
            # Download model using snapshot_download
            # This will cache the model in the default Hugging Face cache
            # We'll track it in our index
            try:
                # For Unsloth models, we can use FastLanguageModel which handles download
                if load_in_4bit:
                    # Just verify the model exists by trying to load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    print(f"[Model Manager] Model {model_name} verified successfully")
                else:
                    # Download full model
                    if snapshot_download:
                        snapshot_download(
                            repo_id=model_name,
                            local_dir=None,  # Use default cache
                            local_dir_use_symlinks=True
                        )
                    else:
                        # Fallback: just verify tokenizer exists
                        AutoTokenizer.from_pretrained(model_name)
                
                # Add to index
                self.models_index[model_name] = {
                    "name": model_name,
                    "installed_at": str(Path.home() / ".cache" / "huggingface" / "hub"),  # Default HF cache
                    "load_in_4bit": load_in_4bit,
                    "type": "huggingface"
                }
                self._save_models_index()
                
                return {
                    "success": True,
                    "message": f"Model {model_name} installed successfully",
                    "model_name": model_name
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error installing model: {str(e)}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error installing model: {str(e)}"
            }
    
    def list_installed_models(self):
        """
        List all installed Hugging Face models
        
        Returns:
            list of model dictionaries
        """
        models = []
        
        # Add models from index
        for model_name, model_info in self.models_index.items():
            models.append({
                "name": model_name,
                "type": "huggingface",
                "installed": True,
                "load_in_4bit": model_info.get("load_in_4bit", True)
            })
        
        # Also check for fine-tuned models
        qlora_models_dir = "./qlora_models"
        if os.path.exists(qlora_models_dir):
            for model_dir in os.listdir(qlora_models_dir):
                model_path = os.path.join(qlora_models_dir, model_dir)
                if os.path.isdir(model_path):
                    # Check if it's a valid model directory
                    if any(os.path.exists(os.path.join(model_path, f)) for f in ["config.json", "adapter_config.json"]):
                        models.append({
                            "name": model_dir,
                            "type": "fine-tuned",
                            "installed": True,
                            "path": model_path,
                            "load_in_4bit": True
                        })
        
        return models
    
    def load_model(self, model_name, max_seq_length=2048, load_in_4bit=True):
        """
        Load a model for inference
        
        Args:
            model_name: Model identifier
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to use 4-bit quantization
            
        Returns:
            tuple of (model, tokenizer) or None if error
        """
        try:
            # Check if it's a fine-tuned model
            fine_tuned_path = f"./qlora_models/{model_name}"
            if os.path.exists(fine_tuned_path):
                # Load fine-tuned model
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=fine_tuned_path,
                    max_seq_length=max_seq_length,
                    dtype=None,
                    load_in_4bit=load_in_4bit,
                )
                FastLanguageModel.for_inference(model)
                return (model, tokenizer)
            
            # Load regular Hugging Face model
            # Check if model is in index (installed)
            if model_name not in self.models_index:
                # Try to install it first
                install_result = self.install_model(model_name, load_in_4bit)
                if not install_result.get("success"):
                    print(f"[Model Manager] Warning: Could not install model {model_name}")
            
            # Load using Unsloth for optimized inference
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
            
            # Add to index if not already there
            if model_name not in self.models_index:
                self.models_index[model_name] = {
                    "name": model_name,
                    "installed_at": "huggingface_cache",
                    "load_in_4bit": load_in_4bit,
                    "type": "huggingface"
                }
                self._save_models_index()
            
            return (model, tokenizer)
            
        except Exception as e:
            print(f"[Model Manager] Error loading model {model_name}: {e}")
            return None
    
    def remove_model(self, model_name):
        """
        Remove a model from the index (doesn't delete from cache)
        
        Args:
            model_name: Model identifier
            
        Returns:
            dict with removal status
        """
        if model_name in self.models_index:
            del self.models_index[model_name]
            self._save_models_index()
            return {
                "success": True,
                "message": f"Model {model_name} removed from index"
            }
        return {
            "success": False,
            "error": f"Model {model_name} not found in index"
        }
    
    def delete_model(self, model_name):
        """
        Delete a model completely (from index and filesystem if fine-tuned)
        
        Args:
            model_name: Model identifier
            
        Returns:
            dict with deletion status
        """
        try:
            # Check if it's a fine-tuned model
            fine_tuned_path = f"./qlora_models/{model_name}"
            if os.path.exists(fine_tuned_path):
                # Delete the fine-tuned model directory
                import shutil
                shutil.rmtree(fine_tuned_path)
                # Remove from index if present
                if model_name in self.models_index:
                    del self.models_index[model_name]
                    self._save_models_index()
                return {
                    "success": True,
                    "message": f"Fine-tuned model {model_name} deleted successfully"
                }
            
            # For Hugging Face models, just remove from index
            # (actual model files are in HF cache, we don't delete those)
            if model_name in self.models_index:
                del self.models_index[model_name]
                self._save_models_index()
                return {
                    "success": True,
                    "message": f"Model {model_name} removed from index (Hugging Face cache not deleted)"
                }
            
            return {
                "success": False,
                "error": f"Model {model_name} not found"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error deleting model: {str(e)}"
            }
    
    def export_model(self, model_name, export_path):
        """
        Export a model to a specified path
        
        Args:
            model_name: Model identifier
            export_path: Destination path to export the model
            
        Returns:
            dict with export status
        """
        try:
            import shutil
            
            # Check if it's a fine-tuned model
            fine_tuned_path = f"./qlora_models/{model_name}"
            if os.path.exists(fine_tuned_path):
                # Export fine-tuned model
                if not os.path.exists(export_path):
                    os.makedirs(export_path, exist_ok=True)
                
                # Copy entire model directory
                destination = os.path.join(export_path, model_name)
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                
                shutil.copytree(fine_tuned_path, destination)
                
                return {
                    "success": True,
                    "message": f"Fine-tuned model {model_name} exported to {destination}",
                    "export_path": destination
                }
            
            # For Hugging Face models, we need to download/copy from cache
            # This is more complex, so we'll just provide instructions
            return {
                "success": False,
                "error": f"Hugging Face models are stored in the Hugging Face cache. To export, use the Hugging Face CLI or download directly from Hugging Face Hub.",
                "suggestion": f"Fine-tuned models can be exported. Hugging Face models should be downloaded from the Hub or use 'huggingface-cli download {model_name}'"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error exporting model: {str(e)}"
            }

