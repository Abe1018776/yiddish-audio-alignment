"""
Flask web application for cleaning Yiddish transcripts.
"""
import os
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from document_processor import DocumentProcessor
from drive_downloader import DriveDownloader
from cleaner import TranscriptCleaner, DEFAULT_PROFILE

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'docx', 'doc'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

processor = DocumentProcessor()

# Cache for cleaning profiles to avoid repeated instantiation
_cached_profiles = None


def get_cleaner_profiles():
    """Get available cleaning profiles (cached)."""
    global _cached_profiles
    if _cached_profiles is None:
        cleaner = TranscriptCleaner()
        _cached_profiles = cleaner.get_available_profiles()
    return _cached_profiles


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/profiles', methods=['GET'])
def get_profiles():
    """Get available cleaning profiles."""
    profiles = get_cleaner_profiles()
    return jsonify(profiles)


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
        
        # Get the selected profile from the form data (use default if not specified)
        profile = request.form.get('profile', DEFAULT_PROFILE)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the document with selected profile
        result = processor.process_document(filepath, filename, profile)
        
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
        profile = data.get('profile', DEFAULT_PROFILE)
        
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
                result = processor.process_document(file_info['path'], file_info['name'], profile)
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
    # Only enable debug mode if explicitly set via environment variable
    # In production, use a WSGI server like gunicorn instead
    import os
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
