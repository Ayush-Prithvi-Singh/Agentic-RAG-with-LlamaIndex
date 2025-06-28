"""
Utility functions for the Agentic RAG system.
"""
import os
import re
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_extension(file_path: str) -> str:
    """Get the file extension from a file path."""
    return Path(file_path).suffix.lower()

def is_supported_file(file_path: str, supported_extensions: set) -> bool:
    """Check if a file is supported based on its extension."""
    return get_file_extension(file_path) in supported_extensions

def generate_file_hash(file_path: str) -> str:
    """Generate a hash for a file to track changes."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    return text.strip()

def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings
            sentence_endings = ['.', '!', '?', '\n\n']
            for ending in sentence_endings:
                last_ending = text.rfind(ending, start, end)
                if last_ending > start + chunk_size // 2:  # Only break if it's not too early
                    end = last_ending + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from filename."""
    path = Path(filename)
    return {
        'filename': path.name,
        'extension': path.suffix,
        'size': path.stat().st_size if path.exists() else 0,
        'created': path.stat().st_ctime if path.exists() else None,
        'modified': path.stat().st_mtime if path.exists() else None
    }

def format_response(response: str, max_length: int = 1000) -> str:
    """Format and truncate response if necessary."""
    if len(response) <= max_length:
        return response
    
    # Try to truncate at a sentence boundary
    truncated = response[:max_length]
    last_sentence = truncated.rfind('.')
    if last_sentence > max_length * 0.8:  # If we can find a sentence ending in the last 20%
        return truncated[:last_sentence + 1] + "..."
    
    return truncated + "..."

def create_summary(text: str, max_words: int = 100) -> str:
    """Create a summary of text by taking the first few sentences."""
    sentences = re.split(r'[.!?]+', text)
    summary = ""
    word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_words = len(sentence.split())
        if word_count + sentence_words <= max_words:
            summary += sentence + ". "
            word_count += sentence_words
        else:
            break
    
    return summary.strip()

def validate_file_path(file_path: str) -> bool:
    """Validate if a file path exists and is readable."""
    try:
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
    except (OSError, ValueError):
        return False

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        return 0.0

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    return sanitized or 'unnamed_file' 