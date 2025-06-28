"""
Document loader for the Agentic RAG system.
Supports multiple file formats and provides intelligent text extraction.
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from llama_index.core import Document, SimpleDirectoryReader
import pandas as pd

from .utils import (
    is_supported_file, 
    clean_text, 
    extract_metadata_from_filename,
    validate_file_path,
    get_file_size_mb,
    sanitize_filename
)
from config import Config

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading and processing of various document formats."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.supported_extensions = self.config.SUPPORTED_EXTENSIONS
        
    def load_document(self, file_path: str) -> Optional[Document]:
        """Load a single document and return a Document object."""
        if not validate_file_path(file_path):
            logger.error(f"Invalid file path: {file_path}")
            return None
            
        if not is_supported_file(file_path, self.supported_extensions):
            logger.error(f"Unsupported file type: {file_path}")
            return None
            
        try:
            extension = Path(file_path).suffix.lower()
            
            if extension == '.pdf':
                return self._load_pdf(file_path)
            elif extension in ['.docx', '.doc']:
                return self._load_docx(file_path)
            elif extension == '.csv':
                return self._load_csv(file_path)
            elif extension == '.json':
                return self._load_json(file_path)
            elif extension in ['.txt', '.md']:
                return self._load_text(file_path)
            else:
                # Try using SimpleDirectoryReader as fallback
                return self._load_generic(file_path)
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return None
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        documents = []
        
        if not os.path.isdir(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return documents
            
        try:
            # Use SimpleDirectoryReader for batch processing
            reader = SimpleDirectoryReader(
                input_dir=directory_path,
                filename_as_id=True,
                recursive=True
            )
            documents = reader.load_data()
            
            # Filter for supported files and add metadata
            filtered_documents = []
            for doc in documents:
                file_path = doc.metadata.get('file_path', '')
                if is_supported_file(file_path, self.supported_extensions):
                    # Enhance metadata
                    doc.metadata.update(self._extract_enhanced_metadata(file_path))
                    filtered_documents.append(doc)
                    
            logger.info(f"Loaded {len(filtered_documents)} documents from {directory_path}")
            return filtered_documents
            
        except Exception as e:
            logger.error(f"Error loading documents from directory {directory_path}: {str(e)}")
            return documents
    
    def _load_pdf(self, file_path: str) -> Optional[Document]:
        """Load PDF document."""
        try:
            # For now, use generic loader for PDF
            return self._load_generic(file_path)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
        return None
    
    def _load_docx(self, file_path: str) -> Optional[Document]:
        """Load DOCX document."""
        try:
            # For now, use generic loader for DOCX
            return self._load_generic(file_path)
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {str(e)}")
        return None
    
    def _load_csv(self, file_path: str) -> Optional[Document]:
        """Load CSV document."""
        try:
            df = pd.read_csv(file_path)
            text_content = df.to_string(index=False)
            metadata = self._extract_enhanced_metadata(file_path)
            metadata['columns'] = list(df.columns)
            metadata['rows'] = len(df)
            
            return Document(
                text=text_content,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {str(e)}")
        return None
    
    def _load_json(self, file_path: str) -> Optional[Document]:
        """Load JSON document."""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable text
            text_content = json.dumps(data, indent=2, ensure_ascii=False)
            metadata = self._extract_enhanced_metadata(file_path)
            
            return Document(
                text=text_content,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {str(e)}")
        return None
    
    def _load_text(self, file_path: str) -> Optional[Document]:
        """Load text document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Clean the text
            text_content = clean_text(text_content)
            metadata = self._extract_enhanced_metadata(file_path)
            
            return Document(
                text=text_content,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
        return None
    
    def _load_generic(self, file_path: str) -> Optional[Document]:
        """Generic document loader using SimpleDirectoryReader."""
        try:
            reader = SimpleDirectoryReader(
                input_files=[file_path],
                filename_as_id=True
            )
            documents = reader.load_data()
            if documents:
                doc = documents[0]
                doc.metadata.update(self._extract_enhanced_metadata(file_path))
                return doc
        except Exception as e:
            logger.error(f"Error in generic loader for {file_path}: {str(e)}")
        return None
    
    def _extract_enhanced_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract enhanced metadata from file."""
        metadata = extract_metadata_from_filename(file_path)
        metadata.update({
            'file_size_mb': get_file_size_mb(file_path),
            'file_path': file_path,
            'loader_type': 'document_loader'
        })
        return metadata
    
    def validate_documents(self, documents: List[Document]) -> List[Document]:
        """Validate and filter documents."""
        valid_documents = []
        
        for doc in documents:
            if doc and doc.text and len(doc.text.strip()) > 0:
                # Clean the text
                doc.text = clean_text(doc.text)
                valid_documents.append(doc)
            else:
                logger.warning(f"Skipping document with empty or invalid content")
                
        return valid_documents
    
    def get_document_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate a summary of loaded documents."""
        if not documents:
            return {"total_documents": 0}
            
        total_size = sum(doc.metadata.get('file_size_mb', 0) for doc in documents)
        total_text_length = sum(len(doc.text) for doc in documents)
        
        file_types = {}
        for doc in documents:
            ext = doc.metadata.get('extension', 'unknown')
            file_types[ext] = file_types.get(ext, 0) + 1
            
        return {
            "total_documents": len(documents),
            "total_size_mb": round(total_size, 2),
            "total_text_length": total_text_length,
            "file_types": file_types,
            "average_text_length": round(total_text_length / len(documents), 2)
        } 