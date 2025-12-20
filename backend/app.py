from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import subprocess
from werkzeug.utils import secure_filename
from datetime import datetime
from rag_processor import RAGProcessor
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qlora_trainer import QLoRATrainer

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all API routes from any origin

# Initialize RAG Processor
rag_processor = RAGProcessor()

# Initialize QLoRA Trainer
qlora_trainer = QLoRATrainer()

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
    """Get list of available Ollama models (including fine-tuned)"""
    try:
        # Run ollama list command
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        models = []
        
        if result.returncode == 0:
            # Parse the output
            lines = result.stdout.strip().split('\n')
            
            # Skip the header line (if present)
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 1:
                        model_name = parts[0]
                        # Extract size if available
                        size = parts[1] if len(parts) > 1 else None
                        models.append({
                            'name': model_name,
                            'size': size
                        })
        
        # Also include fine-tuned models that are ready
        try:
            fine_tuned = qlora_trainer.list_trained_models()
            for model in fine_tuned:
                if model['ollama_ready']:
                    # Check if already in list
                    if not any(m['name'] == model['name'] for m in models):
                        models.append({
                            'name': model['name'],
                            'size': None
                        })
        except:
            pass
        
        return jsonify({
            'models': models,
            'count': len(models)
        }), 200
    
    except FileNotFoundError:
        return jsonify({
            'error': 'Ollama not found. Please make sure Ollama is installed and in your PATH.',
            'models': []
        }), 503
    except subprocess.TimeoutExpired:
        return jsonify({
            'error': 'Timeout while listing models',
            'models': []
        }), 500
    except Exception as e:
        return jsonify({
            'error': f'Error listing models: {str(e)}',
            'models': []
        }), 500

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

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with RAG retrieval"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        model_name = data.get('model', 'qwen3:14b')  # Default model or from request
        conversation_history = data.get('conversation_history', [])  # Full conversation history
        
        if not query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if Ollama is available
        try:
            llm = Ollama(model=model_name)
        except Exception as e:
            # Fallback to simple retrieval if Ollama is not available
            results = rag_processor.search_similar(query, k=4)
            context = "\n\n".join([doc.page_content for doc in results])
            
            return jsonify({
                'response': f"Based on your documents:\n\n{context[:500]}...",
                'sources': [doc.metadata.get('source_file', 'Unknown') for doc in results],
                'note': 'Ollama not available, showing retrieved context only'
            }), 200
        
        # Check if we have documents in the vector store
        has_docs = rag_processor.has_documents()
        
        if not has_docs:
            # No documents uploaded, use LLM normally
            response = llm.invoke(query)
            return jsonify({
                'response': response,
                'sources': []
            }), 200
        
        # Retrieve relevant documents with similarity scores
        results_with_scores = rag_processor.search_similar_with_scores(query, k=4)
        
        # FAISS uses L2 distance (lower = more similar)
        # Set a distance threshold - if all results are too far, query is not related
        # Typical good matches have distance < 1.0, adjust based on your embedding model
        MAX_DISTANCE_THRESHOLD = 1.2
        
        # Check if we have any reasonably relevant documents
        relevant_docs = []
        min_distance = float('inf')
        
        for doc, distance in results_with_scores:
            if distance < min_distance:
                min_distance = distance
            if distance < MAX_DISTANCE_THRESHOLD:
                relevant_docs.append(doc)
        
        # If no relevant documents found or all are too distant, use LLM normally with conversation history
        if not relevant_docs or min_distance > MAX_DISTANCE_THRESHOLD:
            if conversation_history:
                # Use conversation history with ChatOllama for proper context
                from langchain_community.chat_models import ChatOllama
                from langchain_core.messages import HumanMessage, AIMessage
                
                chat_llm = ChatOllama(model=model_name)
                
                # Convert conversation history to LangChain messages
                langchain_messages = []
                for msg in conversation_history:
                    if msg['role'] == 'user':
                        langchain_messages.append(HumanMessage(content=msg['content']))
                    elif msg['role'] == 'assistant':
                        langchain_messages.append(AIMessage(content=msg['content']))
                
                # Get response with full conversation context
                response = chat_llm.invoke(langchain_messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback to simple invoke if no history
                response_text = llm.invoke(query)
            
            return jsonify({
                'response': response_text,
                'sources': []
            }), 200
        
        # We have relevant documents, use RAG
        # Extract sources
        sources = list(set([
            doc.metadata.get('source_file', 'Unknown')
            for doc in relevant_docs
        ]))
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Use ChatOllama for conversation history support
        from langchain_community.chat_models import ChatOllama
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        chat_llm = ChatOllama(model=model_name)
        
        # Create system message with context
        system_message = SystemMessage(content=f"""You are a helpful AI assistant. Answer questions using the provided context from uploaded documents when relevant.

If the question is related to the provided context, use that context to give a detailed and accurate answer. Reference the documents when relevant.

If the question is NOT related to the provided context (e.g., general conversation, greetings, unrelated topics), ignore the context and answer naturally using your general knowledge.

Context from uploaded documents:
{context}""")
        
        # Convert conversation history to LangChain messages
        langchain_messages = [system_message]
        for msg in conversation_history:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=msg['content']))
        
        # Get response with full conversation context and RAG
        response = chat_llm.invoke(langchain_messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return jsonify({
            'response': response_text,
            'sources': sources
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

@app.route('/api/qlora/convert', methods=['POST'])
def qlora_convert():
    """Convert fine-tuned model to Ollama format"""
    try:
        data = request.get_json()
        model_path = data.get('model_path', '').strip()
        
        if not model_path:
            return jsonify({'error': 'Model path is required'}), 400
        
        ollama_model_name = qlora_trainer.convert_to_ollama(model_path)
        
        return jsonify({
            'message': 'Model converted successfully',
            'ollama_model_name': ollama_model_name
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error converting model: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

