"""
Document processor for handling Word documents.
"""
import os
from docx import Document
from cleaner import TranscriptCleaner, DEFAULT_PROFILE


class DocumentProcessor:
    """Processes Word documents and extracts text."""
    
    def __init__(self):
        self.cleaner = TranscriptCleaner()
    
    def extract_text_from_docx(self, file_path):
        """
        Extract text from a .docx file.
        
        Args:
            file_path: Path to the .docx file
            
        Returns:
            str: Extracted text
        """
        try:
            doc = Document(file_path)
            paragraphs = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            
            return '\n'.join(paragraphs)
        except Exception as e:
            raise Exception(f"Error reading document: {str(e)}")
    
    def process_document(self, file_path, filename, profile=None):
        """
        Process a document: extract text, clean it, and return results.
        
        Args:
            file_path: Path to the document file
            filename: Original filename
            profile: Cleaning profile to use (default: DEFAULT_PROFILE constant)
            
        Returns:
            dict: Processing results including original, cleaned, removed items, and stats
        """
        # Use default profile if none specified
        if profile is None:
            profile = DEFAULT_PROFILE
            
        # Extract text
        original_text = self.extract_text_from_docx(file_path)
        
        # Clean text with selected profile
        cleaned_text, removed_items, profile_used = self.cleaner.clean_text(original_text, profile)
        
        # Get statistics
        stats = self.cleaner.get_statistics(original_text, cleaned_text)
        
        return {
            'filename': filename,
            'original_text': original_text,
            'cleaned_text': cleaned_text,
            'removed_items': removed_items,
            'statistics': stats,
            'profile': profile_used,
            'success': True
        }
    
    def save_cleaned_document(self, cleaned_text, output_path):
        """
        Save cleaned text to a new Word document.
        
        Args:
            cleaned_text: The cleaned text to save
            output_path: Path where to save the document
        """
        doc = Document()
        
        # Split text into paragraphs and add to document
        paragraphs = cleaned_text.split('\n')
        for para_text in paragraphs:
            if para_text.strip():
                doc.add_paragraph(para_text)
        
        doc.save(output_path)
        return output_path
