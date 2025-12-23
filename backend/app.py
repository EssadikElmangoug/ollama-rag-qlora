from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import subprocess
from werkzeug.utils import secure_filename
from datetime import datetime
from rag_processor import RAGProcessor
from qlora_trainer import QLoRATrainer
from model_manager import ModelManager

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all API routes from any origin

# Initialize RAG Processor
rag_processor = RAGProcessor()

# Initialize QLoRA Trainer
qlora_trainer = QLoRATrainer()

# Initialize Model Manager
model_manager = ModelManager()

# Cache for loaded models (to avoid reloading on every request)
loaded_models_cache = {}

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'md', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'ppt', 'pptx'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({
        'message': 'Flask server is running!',
        'status': 'success'
    })

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Flask Backend'
    })

@app.route('/api/models', methods=['GET'])
def list_models():
    """Get list of available Hugging Face models (installed and fine-tuned)"""
    try:
        models = model_manager.list_installed_models()
        
        return jsonify({
            'models': models,
            'count': len(models)
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Error listing models: {str(e)}',
            'models': [],
            'count': 0
        }), 500

@app.route('/api/models/install', methods=['POST'])
def install_model():
    """Install a model from Hugging Face"""
    try:
        data = request.get_json()
        model_name = data.get('model_name', '').strip()
        load_in_4bit = data.get('load_in_4bit', True)
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Check if it's a GGUF model
        if 'gguf' in model_name.lower():
            return jsonify({
                'error': 'GGUF models cannot be used for training or inference in this system. Please use models compatible with transformers (e.g., models with "-bnb-4bit" suffix).'
            }), 400
        
        result = model_manager.install_model(model_name, load_in_4bit)
        
        if result.get('success'):
            return jsonify(result), 200
        else:
            return jsonify(result), 500
    
    except Exception as e:
        return jsonify({'error': f'Error installing model: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'File type not allowed',
            'allowed_types': list(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        filename = secure_filename(file.filename)
        # Add timestamp to avoid filename conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        file_size = os.path.getsize(filepath)
        
        # Process document for RAG
        original_filename = file.filename
        processing_result = rag_processor.process_document(filepath, original_filename)
        
        if not processing_result.get('success'):
            # File was saved but processing failed
            return jsonify({
                'message': 'File uploaded but processing failed',
                'filename': filename,
                'original_filename': original_filename,
                'size': file_size,
                'uploaded_at': datetime.now().isoformat(),
                'processing_error': processing_result.get('error')
            }), 200
        
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'filename': filename,
            'original_filename': original_filename,
            'size': file_size,
            'uploaded_at': datetime.now().isoformat(),
            'processing': {
                'chunks_count': processing_result.get('chunks_count'),
                'total_pages': processing_result.get('total_pages')
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    try:
        files = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    files.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'uploaded_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return jsonify({'files': files}), 200
    except Exception as e:
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500

@app.route('/api/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Remove from RAG (note: full deletion requires rebuilding index)
        rag_processor.remove_document(filename)
        
        os.remove(filepath)
        return jsonify({'message': 'File deleted successfully'}), 200
    
    except Exception as e:
        return jsonify({'error': f'Error deleting file: {str(e)}'}), 500

def load_model_for_inference(model_name):
    """Load a model (Hugging Face or fine-tuned) for inference"""
    # Check if model is already loaded
    if model_name in loaded_models_cache:
        return loaded_models_cache[model_name]
    
    try:
        # Use model manager to load the model
        result = model_manager.load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        
        if result:
            model, tokenizer = result
            # Cache the loaded model
            loaded_models_cache[model_name] = (model, tokenizer)
            print(f"Loaded model: {model_name}")
            return (model, tokenizer)
        else:
            return None
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def generate_with_model(model, tokenizer, messages, max_new_tokens=512):
    """Generate response using a Hugging Face model"""
    try:
        # Format messages for chat template
        formatted_messages = []
        for msg in messages:
            if msg['role'] == 'user':
                formatted_messages.append({"role": "user", "content": msg['content']})
            elif msg['role'] == 'assistant':
                formatted_messages.append({"role": "assistant", "content": msg['content']})
        
        # Check if tokenizer has chat template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            # Apply chat template
            inputs = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            # Fallback: simple formatting
            text = ""
            for msg in formatted_messages:
                if msg['role'] == 'user':
                    text += f"User: {msg['content']}\n\n"
                elif msg['role'] == 'assistant':
                    text += f"Assistant: {msg['content']}\n\n"
            text += "Assistant: "
            inputs = tokenizer(text, return_tensors="pt")
            inputs = inputs['input_ids']
        
        # Move to device (GPU if available, else CPU)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device)
        # Model should already be on the correct device from loading, but ensure it is
        if next(model.parameters()).device.type != device:
            model = model.to(device)
        
        # Generate response
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
        
        # Decode response (skip the prompt)
        if isinstance(inputs, torch.Tensor):
            prompt_length = inputs.shape[1]
        else:
            prompt_length = len(inputs[0]) if isinstance(inputs, list) else inputs.shape[1]
        
        response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        raise Exception(f"Error generating with model: {str(e)}")

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with RAG retrieval using Hugging Face models"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        model_name = data.get('model', 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit')  # Default Hugging Face model
        conversation_history = data.get('conversation_history', [])  # Full conversation history
        
        if not query:
            return jsonify({'error': 'No message provided'}), 400
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Load model (Hugging Face or fine-tuned)
        model_result = load_model_for_inference(model_name)
        
        if not model_result:
            # Try to install the model if it's not found
            install_result = model_manager.install_model(model_name, load_in_4bit=True)
            if not install_result.get('success'):
                return jsonify({
                    'error': f'Model {model_name} not found and could not be installed. Please install it first or use an installed model.',
                    'suggestion': 'Use /api/models/install endpoint to install models'
                }), 404
            
            # Try loading again after installation
            model_result = load_model_for_inference(model_name)
            if not model_result:
                return jsonify({'error': f'Failed to load model {model_name} after installation'}), 500
        
        model, tokenizer = model_result
        
        # Prepare messages with current query
        messages = conversation_history + [{'role': 'user', 'content': query}]
        
        # Check if we have documents for RAG
        has_docs = rag_processor.has_documents()
        
        sources = []
        if has_docs:
            # Retrieve relevant documents
            results_with_scores = rag_processor.search_similar_with_scores(query, k=4)
            MAX_DISTANCE_THRESHOLD = 1.2
            
            relevant_docs = []
            min_distance = float('inf')
            
            for doc, distance in results_with_scores:
                if distance < min_distance:
                    min_distance = distance
                if distance < MAX_DISTANCE_THRESHOLD:
                    relevant_docs.append(doc)
            
            # Add context if relevant documents found
            if relevant_docs and min_distance < MAX_DISTANCE_THRESHOLD:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                system_message = f"""You are a helpful AI assistant. Answer questions using the provided context from uploaded documents when relevant.

If the question is related to the provided context, use that context to give a detailed and accurate answer. Reference the documents when relevant.

If the question is NOT related to the provided context (e.g., general conversation, greetings, unrelated topics), ignore the context and answer naturally using your general knowledge.

Context from uploaded documents:
{context}"""
                
                # Prepend system message
                messages = [{'role': 'user', 'content': system_message}] + messages
                sources = list(set([doc.metadata.get('source_file', 'Unknown') for doc in relevant_docs]))
        
        # Generate response
        response_text = generate_with_model(model, tokenizer, messages)
        
        # Determine model type
        model_type = 'fine-tuned' if os.path.exists(f"./qlora_models/{model_name}") else 'huggingface'
        
        return jsonify({
            'response': response_text,
            'sources': sources,
            'model_type': model_type
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error processing chat: {str(e)}'}), 500

@app.route('/api/qlora/train', methods=['POST'])
def qlora_train():
    """Start QLoRA fine-tuning"""
    try:
        data = request.get_json()
        base_model = data.get('base_model', 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit')
        model_name = data.get('model_name', '').strip()
        lora_rank = data.get('lora_rank', 16)
        max_steps = data.get('max_steps', 100)
        learning_rate = data.get('learning_rate', 0.0002)
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Check if training is already in progress
        if qlora_trainer.training_status['status'] == 'training':
            return jsonify({'error': 'Training already in progress'}), 400
        
        # Start training
        qlora_trainer.train_model(
            base_model=base_model,
            model_name=model_name,
            lora_rank=lora_rank,
            max_steps=max_steps,
            learning_rate=learning_rate
        )
        
        return jsonify({
            'message': 'Training started',
            'model_name': model_name
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error starting training: {str(e)}'}), 500

@app.route('/api/qlora/status', methods=['GET'])
def qlora_status():
    """Get QLoRA training status"""
    try:
        return jsonify(qlora_trainer.training_status), 200
    except Exception as e:
        return jsonify({'error': f'Error getting status: {str(e)}'}), 500

@app.route('/api/qlora/models', methods=['GET'])
def qlora_models():
    """List all fine-tuned models"""
    try:
        models = qlora_trainer.list_trained_models()
        return jsonify({'models': models}), 200
    except Exception as e:
        return jsonify({'error': f'Error listing models: {str(e)}'}), 500

if __name__ == '__main__':
    # Disable reloader to prevent Flask from restarting when Unsloth creates cache files
    # This ensures training threads aren't killed by server restarts
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

