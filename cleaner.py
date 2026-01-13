"""
Text cleaning utilities for Yiddish transcripts.
Removes titles, headings, narrator notes, redactor notes, etc.
"""
import re


class TranscriptCleaner:
    """Cleans Yiddish transcripts by removing non-transcript content."""
    
    def __init__(self):
        # Define cleaning profiles
        self.profiles = {
            'titles_only': {
                'description': 'Removes only titles, headings, and speaker labels',
                'patterns': [
                    # Common heading patterns
                    (r'^[A-Z\s]+:.*$', 'headings with colon'),
                    (r'^Chapter \d+.*$', 'chapter headings'),
                    (r'^Section \d+.*$', 'section headings'),
                    
                    # Time stamps
                    (r'\d{1,2}:\d{2}:\d{2}', 'timestamps'),
                    (r'\[\d{1,2}:\d{2}\]', 'bracketed timestamps'),
                    
                    # Speaker labels (common in transcripts)
                    (r'^Speaker \d+:.*$', 'speaker labels'),
                    (r'^Interviewer:.*$', 'interviewer labels'),
                    (r'^Narrator:.*$', 'narrator labels'),
                    
                    # Multiple consecutive newlines (leave at most 2)
                    (r'\n{3,}', 'excessive newlines'),
                    
                    # Lines with only special characters or numbers
                    (r'^[\d\s\-_=]+$', 'separator lines'),
                    
                    # Page numbers
                    (r'^\s*Page \d+\s*$', 'page numbers'),
                    (r'^\s*\d+\s*$', 'standalone numbers'),
                ]
            },
            'titles_and_parentheses': {
                'description': 'Removes titles/headings AND all bracketed/parenthetical content',
                'patterns': [
                    # Bracketed content (narrator/redactor notes) - process before special pattern
                    (r'\[.*?\]', 'bracketed notes'),
                    (r'\(.*?\)', 'parenthetical notes'),
                    
                    # Bracketed content followed by parenthetical content (special case)
                    # Note: This pattern is placed after individual patterns to avoid conflicts
                    (r'\[.*?\]\s*\(.*?\)', 'brackets followed by parentheses'),
                    
                    # Common heading patterns
                    (r'^[A-Z\s]+:.*$', 'headings with colon'),
                    (r'^Chapter \d+.*$', 'chapter headings'),
                    (r'^Section \d+.*$', 'section headings'),
                    
                    # Time stamps
                    (r'\d{1,2}:\d{2}:\d{2}', 'timestamps'),
                    (r'\[\d{1,2}:\d{2}\]', 'bracketed timestamps'),
                    
                    # Speaker labels (common in transcripts)
                    (r'^Speaker \d+:.*$', 'speaker labels'),
                    (r'^Interviewer:.*$', 'interviewer labels'),
                    (r'^Narrator:.*$', 'narrator labels'),
                    
                    # Multiple consecutive newlines (leave at most 2)
                    (r'\n{3,}', 'excessive newlines'),
                    
                    # Lines with only special characters or numbers
                    (r'^[\d\s\-_=]+$', 'separator lines'),
                    
                    # Page numbers
                    (r'^\s*Page \d+\s*$', 'page numbers'),
                    (r'^\s*\d+\s*$', 'standalone numbers'),
                ]
            }
        }
        
        # Default profile for backwards compatibility
        self.removal_patterns = self.profiles['titles_and_parentheses']['patterns']
        
        # Characters to remove
        self.chars_to_remove = [
            '\u200b',  # Zero-width space
            '\ufeff',  # BOM
        ]
    
    def get_available_profiles(self):
        """
        Get list of available cleaning profiles.
        
        Returns:
            dict: Dictionary of profile names to their descriptions
        """
        return {
            name: profile['description'] 
            for name, profile in self.profiles.items()
        }
    
    def clean_text(self, text, profile='titles_and_parentheses'):
        """
        Clean the transcript text and return both cleaned text and removed content.
        
        Args:
            text: The original transcript text
            profile: The cleaning profile to use (default: 'titles_and_parentheses')
                     Available profiles: 'titles_only', 'titles_and_parentheses'
            
        Returns:
            tuple: (cleaned_text, removed_items, profile_name)
                - cleaned_text: The text after cleaning
                - removed_items: List of dicts with 'pattern', 'matches', and 'count'
                - profile_name: The name of the profile used
        """
        # Validate profile
        if profile not in self.profiles:
            profile = 'titles_and_parentheses'  # Default fallback
        
        # Get patterns for the selected profile
        removal_patterns = self.profiles[profile]['patterns']
        
        removed_items = []
        cleaned = text
        
        # Remove special characters
        for char in self.chars_to_remove:
            if char in cleaned:
                count = cleaned.count(char)
                cleaned = cleaned.replace(char, '')
                removed_items.append({
                    'pattern': f'Special character (unicode {ord(char):04x})',
                    'matches': [char] * count,
                    'count': count
                })
        
        # Apply regex patterns
        for pattern, description in removal_patterns:
            matches = re.findall(pattern, cleaned, re.MULTILINE | re.IGNORECASE)
            if matches:
                # Store what was removed
                removed_items.append({
                    'pattern': description,
                    'matches': matches[:10],  # Limit to first 10 for display
                    'count': len(matches)
                })
                
                # Remove the matches
                if description == 'excessive newlines':
                    cleaned = re.sub(pattern, '\n\n', cleaned, flags=re.MULTILINE)
                else:
                    cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned = re.sub(r' +', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'\n ', '\n', cleaned)  # Space after newline
        cleaned = re.sub(r' \n', '\n', cleaned)  # Space before newline
        cleaned = cleaned.strip()
        
        return cleaned, removed_items, profile
    
    def get_statistics(self, original_text, cleaned_text):
        """
        Get statistics about the cleaning process.
        
        Args:
            original_text: The original text
            cleaned_text: The cleaned text
            
        Returns:
            dict: Statistics about the cleaning
        """
        original_lines = len(original_text.split('\n'))
        cleaned_lines = len(cleaned_text.split('\n'))
        original_words = len(original_text.split())
        cleaned_words = len(cleaned_text.split())
        
        # Calculate reduction percentage
        if len(original_text) > 0:
            reduction_percentage = round((1 - len(cleaned_text) / len(original_text)) * 100, 2)
        else:
            reduction_percentage = 0.0
        
        return {
            'original_chars': len(original_text),
            'cleaned_chars': len(cleaned_text),
            'removed_chars': len(original_text) - len(cleaned_text),
            'original_lines': original_lines,
            'cleaned_lines': cleaned_lines,
            'removed_lines': original_lines - cleaned_lines,
            'original_words': original_words,
            'cleaned_words': cleaned_words,
            'removed_words': original_words - cleaned_words,
            'reduction_percentage': reduction_percentage
        }
