"""
Flask web application for cleaning Yiddish transcripts.
"""
import os
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from document_processor import DocumentProcessor
from drive_downloader import DriveDownloader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'docx', 'doc'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

processor = DocumentProcessor()


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a .docx or .doc file'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the document
        result = processor.process_document(filepath, filename)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/process-drive', methods=['POST'])
def process_drive():
    """Handle Google Drive folder processing."""
    try:
        data = request.get_json()
        drive_url = data.get('drive_url', '').strip()
        
        if not drive_url:
            return jsonify({'error': 'No Google Drive URL provided'}), 400
        
        # Check if Google credentials exist
        if not os.path.exists('credentials.json'):
            return jsonify({
                'error': 'Google Drive credentials not configured. Please see README for setup instructions.',
                'success': False
            }), 400
        
        # Download documents from Drive
        downloader = DriveDownloader()
        downloaded_files = downloader.download_folder(drive_url, app.config['TEMP_FOLDER'])
        
        if not downloaded_files:
            return jsonify({'error': 'No documents found in the folder', 'success': False}), 404
        
        # Process each downloaded file
        results = []
        for file_info in downloaded_files:
            try:
                result = processor.process_document(file_info['path'], file_info['name'])
                results.append(result)
                # Clean up temp file
                os.remove(file_info['path'])
            except Exception as e:
                results.append({
                    'filename': file_info['name'],
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_files': len(results)
        })
    
    except Exception as e:
        # Clean up temp files on error
        if os.path.exists(app.config['TEMP_FOLDER']):
            shutil.rmtree(app.config['TEMP_FOLDER'])
            os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
        
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/download-cleaned', methods=['POST'])
def download_cleaned():
    """Generate and download cleaned document."""
    try:
        data = request.get_json()
        cleaned_text = data.get('cleaned_text', '')
        filename = data.get('filename', 'cleaned_document.docx')
        
        # Ensure filename ends with .docx
        filename = filename.rsplit('.', 1)[0] + '_cleaned.docx'
        
        # Save cleaned document
        output_path = os.path.join(app.config['TEMP_FOLDER'], filename)
        processor.save_cleaned_document(cleaned_text, output_path)
        
        return send_file(output_path, as_attachment=True, download_name=filename)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
