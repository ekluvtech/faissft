from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from main import init_llm, load_index, load_index,  query_pdf, SYSTEM_PROMPT, create_embeddings
import logging
import numpy as np
import json
from datetime import datetime
import traceback

class NumpyFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        "supports_credentials": True
    }
})
app.json_encoder = NumpyFloatEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure upload settings
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md', 'doc', 'docx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload and process multiple documents
    ---
    tags:
      - Documents
    parameters:
      - in: formData
        name: files
        type: array
        items:
          type: file
        required: true
        description: The files to upload and process
      - in: formData
        name: create_embeddings
        type: boolean
        required: false
        default: true
        description: Whether to create embeddings for the uploaded files
    responses:
      200:
        description: Files uploaded and processed successfully
      400:
        description: Bad request (invalid files)
      500:
        description: Internal server error
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('file')
    create_embeddings_flag = request.form.get('create_embeddings', 'true').lower() == 'true'
    
    if not files:
        return jsonify({'error': 'No selected file'}), 400

    uploaded_files = []
    skipped_files = []
    
    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(file_path)
                    uploaded_files.append({
                        'filename': filename,
                        'path': file_path,
                        'size': os.path.getsize(file_path)
                    })
                except Exception as e:
                    error_msg = f"Error saving file: {str(e)}"
                    logging.error(f"{error_msg}\n{traceback.format_exc()}")
                    skipped_files.append({'filename': filename, 'error': error_msg})
            else:
                if file.filename:
                    skipped_files.append({
                        'filename': file.filename,
                        'error': f'Invalid file type. Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}'
                    })
        
        if not uploaded_files:
            return jsonify({
                'error': 'No valid files uploaded',
                'skipped_files': skipped_files
            }), 400

        response_data = {
            'message': 'Files uploaded successfully',
            'files': uploaded_files,
            'skipped_files': skipped_files
        }

        if create_embeddings_flag:
            try:
                index, docs = load_index()
                response_data.update({
                    'message': 'Files uploaded and indexed successfully',
                    'document_count': len(docs),
                    'index_status': 'created'
                })
            except Exception as e:
                error_msg = f"Error during indexing: {str(e)}"
                logging.error(f"{error_msg}\n{traceback.format_exc()}")
                response_data.update({
                    'index_status': 'failed',
                    'index_error': error_msg
                })

        return jsonify(response_data), 200

    except Exception as e:
        error_msg = f"Unexpected error during file upload: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({
            'error': error_msg,
            'files': uploaded_files,
            'skipped_files': skipped_files
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """
    List all uploaded documents
    ---
    tags:
      - Documents
    responses:
      200:
        description: List of all documents
      500:
        description: Internal server error
    """
    try:
        documents = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                documents.append({
                    'filename': filename,
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
        return jsonify({
            'document_count': len(documents),
            'documents': documents
        }), 200
    except Exception as e:
        error_msg = f"Error listing documents: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/documents/<filename>', methods=['DELETE'])
def delete_document(filename):
    """
    Delete a specific document
    ---
    tags:
      - Documents
    parameters:
      - in: path
        name: filename
        type: string
        required: true
        description: The name of the file to delete
    responses:
      200:
        description: Document deleted successfully
      404:
        description: Document not found
      500:
        description: Internal server error
    """
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(file_path):
            return jsonify({'error': 'Document not found'}), 404
        
        os.remove(file_path)
        return jsonify({
            'message': 'Document deleted successfully',
            'filename': filename
        }), 200
    except Exception as e:
        error_msg = f"Error deleting document: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/embeddings', methods=['POST'])
def create_document_embeddings():
    """
    Create embeddings for specific documents or all documents
    ---
    tags:
      - Embeddings
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            filenames:
              type: array
              items:
                type: string
              description: List of filenames to process (optional, if not provided all documents will be processed)
    responses:
      200:
        description: Embeddings created successfully
      500:
        description: Internal server error
    """
    try:
        data = request.get_json() or {}
        filenames = data.get('filenames', [])
        
        # If no specific files are provided, process all documents
        if not filenames:
            filenames = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
        
        processed_files = []
        skipped_files = []
        
        for filename in filenames:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
            if not os.path.exists(file_path):
                skipped_files.append({
                    'filename': filename,
                    'error': 'File not found'
                })
                continue
            
            try:
                # Create embeddings for the document
                index, docs = load_index()
                processed_files.append({
                    'filename': filename,
                    'status': 'processed'
                })
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                logging.error(f"{error_msg}\n{traceback.format_exc()}")
                skipped_files.append({
                    'filename': filename,
                    'error': error_msg
                })
        
        return jsonify({
            'message': 'Embeddings creation completed',
            'processed_files': processed_files,
            'skipped_files': skipped_files,
            'total_processed': len(processed_files),
            'total_skipped': len(skipped_files)
        }), 200
    
    except Exception as e:
        error_msg = f"Error creating embeddings: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/query', methods=['POST'])
def query_documents():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query = data['query']
    try:
        # Initialize LLM if not already initialized
       # init_llm()
        # Search documents
        # results = search_documents(query)
        #results = search_documents(query)
        #formatted_output = format_search_results(results)
        results = query_pdf(query)
        if not results:
            return jsonify({
                'message': 'No relevant documents found',
                'results': []
            }), 200
        
        
        
        return jsonify(results), 200
    
    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        return jsonify({'error': f'Error during search: {str(e)}'}), 500

def determine_document_type(results):
    # Simple document type detection based on content and metadata
    for result in results:
        content = result['content'].lower()
        if 'owasp' in content or 'vulnerability' in content or 'security' in content:
            return 'security'
        elif 'proposal' in content or 'budget' in content or 'timeline' in content:
            return 'business'
    return 'technical'

def extract_context(results):
    # Extract context from the most relevant result
    if results:
        return results[0]['content'][:200] + "..."  # First 200 chars of most relevant result
    return ""

def extract_main_points(results):
    # Extract key points from top results
    main_points = []
    for result in results[:2]:  # Use top 2 results for main points
        # Split content into sentences and take first one as main point
        sentences = result['content'].split('.')
        if sentences:
            main_points.append(sentences[0].strip())
    return main_points

def extract_supporting_details(results):
    # Extract supporting details from results
    supporting_details = []
    for result in results:
        # Split content into sentences and take remaining ones as supporting details
        sentences = result['content'].split('.')
        if len(sentences) > 1:
            supporting_details.extend([s.strip() for s in sentences[1:] if s.strip()])
    return supporting_details[:5]  # Limit to top 5 supporting details

def extract_references(results):
    # Extract references from metadata
    references = []
    for result in results:
        if 'metadata' in result and 'source' in result['metadata']:
            ref = f"{result['metadata']['source']}"
            if 'page' in result['metadata']:
                ref += f" (Page {result['metadata']['page']})"
            references.append(ref)
    return list(set(references))  # Remove duplicates

def calculate_average_confidence(results):
    # Calculate average confidence score
    if not results:
        return 0.0
    return sum(r['confidence_score'] for r in results) / len(results)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    init_llm() 
    load_index()
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001) 
