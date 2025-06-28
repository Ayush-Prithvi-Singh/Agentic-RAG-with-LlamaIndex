"""
Configuration settings for the Agentic RAG system.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Agentic RAG system."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Vector Store Configuration
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "agentic_rag")
    
    # Document Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    
    # Agent Configuration
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "3"))
    
    # File Paths
    DATA_DIR = "data"
    VECTOR_STORE_DIR = "vector_store"
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.pdf', '.docx', '.doc', 
        '.csv', '.json', '.xml', '.html', '.htm'
    }
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        
        # Create necessary directories
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(cls.CHROMA_PERSIST_DIRECTORY, exist_ok=True) 