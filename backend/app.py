from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from rag_processor import RAGProcessor
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize RAG Processor
rag_processor = RAGProcessor()

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
        
        if not query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if Ollama is available
        try:
            llm = Ollama(model="llama3.1")
        except Exception as e:
            # Fallback to simple retrieval if Ollama is not available
            results = rag_processor.search_similar(query, k=4)
            context = "\n\n".join([doc.page_content for doc in results])
            
            return jsonify({
                'response': f"Based on your documents:\n\n{context[:500]}...",
                'sources': [doc.metadata.get('source_file', 'Unknown') for doc in results],
                'note': 'Ollama not available, showing retrieved context only'
            }), 200
        
        # Create RetrievalQA chain
        retriever = rag_processor.get_retriever(k=4)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get response
        result = qa_chain.invoke({"query": query})
        
        # Extract sources
        sources = []
        if 'source_documents' in result:
            sources = list(set([
                doc.metadata.get('source_file', 'Unknown')
                for doc in result['source_documents']
            ]))
        
        return jsonify({
            'response': result.get('result', 'No response generated'),
            'sources': sources
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error processing chat: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

