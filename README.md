# Clean Yiddish Transcripts

A web application for cleaning Yiddish transcripts by removing titles, headings, narrator notes, redactor notes, and other non-transcript content. The app provides a clean, modern web interface for processing individual Word documents or entire Google Drive folders.

## Features

- 🔤 **Clean Transcripts**: Automatically removes titles, headings, brackets, timestamps, and other non-transcript content
- 📄 **Single File Upload**: Upload and process individual Word documents (.doc, .docx)
- 📁 **Google Drive Integration**: Process entire folders of documents from Google Drive
- 📊 **Detailed Statistics**: See what was removed and what remains
- ⬇️ **Download Results**: Download cleaned transcripts as Word documents
- 🎨 **Modern UI**: Clean, responsive web interface with drag-and-drop support
- 🐳 **Docker Ready**: Easy deployment with Docker

## What Gets Removed

The cleaner removes:
- Bracketed content: `[narrator notes]`, `(redactor notes)`
- Headings with colons: `CHAPTER 1: Introduction`
- Chapter/Section headings: `Chapter 1`, `Section 2`
- Timestamps: `12:34:56`, `[12:34]`
- Speaker labels: `Speaker 1:`, `Interviewer:`, `Narrator:`
- Page numbers and separator lines
- Special characters and excessive whitespace

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shloimy15e/clean-yiddish-transcripts.git
   cd clean-yiddish-transcripts
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t yiddish-transcript-cleaner .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 yiddish-transcript-cleaner
   ```

3. **Access the application**
   Navigate to `http://localhost:5000`

## Google Drive Integration (Optional)

To use the Google Drive folder processing feature:

1. **Create a Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable the Google Drive API

2. **Create OAuth 2.0 Credentials**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop app" as application type
   - Download the credentials JSON file

3. **Setup Credentials**
   - Rename the downloaded file to `credentials.json`
   - Place it in the root directory of the application

4. **First-time Authentication**
   - The first time you use the Drive feature, a browser window will open
   - Sign in with your Google account and grant permissions
   - The authentication token will be saved for future use

## Usage

### Processing a Single File

1. Click on the "📄 Upload File" tab
2. Drag and drop a Word document or click to browse
3. Click "Process Document"
4. View the results:
   - Statistics showing what was removed
   - Side-by-side comparison of original and cleaned text
   - Detailed list of removed content
5. Download the cleaned document

### Processing Google Drive Folders

1. Click on the "📁 Google Drive" tab
2. Paste the Google Drive folder URL
3. Click "Process Drive Folder"
4. View results for all documents in the folder
5. Download cleaned versions individually

## API Endpoints

The application provides the following REST API endpoints:

- `GET /` - Main web interface
- `POST /upload` - Upload and process a single document
- `POST /process-drive` - Process documents from Google Drive folder
- `POST /download-cleaned` - Download a cleaned document
- `GET /health` - Health check endpoint

## Development

### Project Structure

```
clean-yiddish-transcripts/
├── app.py                  # Flask application
├── cleaner.py             # Text cleaning logic
├── document_processor.py  # Document processing
├── drive_downloader.py    # Google Drive integration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── templates/
│   └── index.html        # Web UI
├── uploads/              # Temporary upload storage
└── temp/                 # Temporary file storage
```

### Customizing Cleaning Rules

To modify what content gets removed, edit the `removal_patterns` list in `cleaner.py`. Each pattern is a tuple of:
- Regular expression pattern
- Description of what it matches

Example:
```python
(r'\[.*?\]', 'bracketed notes'),
```

## Deployment Options

### Cloud Platforms

**Heroku**
```bash
heroku create your-app-name
git push heroku main
```

**Google Cloud Run**
```bash
gcloud run deploy --source .
```

**AWS Elastic Beanstalk**
```bash
eb init -p python-3.11 yiddish-transcript-cleaner
eb create yiddish-transcript-env
eb deploy
```

### Environment Variables

For production deployment, set:
- `FLASK_ENV=production`
- `SECRET_KEY=your-secret-key`

## Security Notes

- The application does not store uploaded files permanently
- Files are deleted immediately after processing
- Google Drive credentials are stored locally and not shared
- For production use, ensure proper authentication and HTTPS

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.