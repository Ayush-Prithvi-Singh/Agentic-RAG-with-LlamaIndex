"""
Vector store management for the Agentic RAG system.
Handles document indexing, storage, and retrieval using ChromaDB.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
import logging

from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb

from .utils import clean_text, split_text_into_chunks
from config import Config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations for document indexing and retrieval."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.chroma_client = None
        self.vector_store = None
        self.index = None
        self.service_context = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ChromaDB client and vector store."""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIRECTORY
            )
            
            # Get or create collection
            collection = self.chroma_client.get_or_create_collection(
                name=self.config.CHROMA_COLLECTION_NAME
            )
            
            # Initialize vector store
            self.vector_store = ChromaVectorStore(
                chroma_collection=collection
            )
            
            # Initialize embedding model
            embedding_model = OpenAIEmbedding(
                model=self.config.OPENAI_EMBEDDING_MODEL
            )
            
            # Initialize service context
            self.service_context = ServiceContext.from_defaults(
                embed_model=embedding_model,
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            
            logger.info("Vector store components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store components: {str(e)}")
            raise
    
    def create_index_from_documents(self, documents: List) -> bool:
        """Create vector index from documents."""
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return False
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                service_context=self.service_context,
                show_progress=True
            )
            
            logger.info(f"Successfully created index from {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index from documents: {str(e)}")
            return False
    
    def add_documents_to_index(self, documents: List) -> bool:
        """Add new documents to existing index."""
        try:
            if not self.index:
                logger.warning("No existing index found. Creating new index.")
                return self.create_index_from_documents(documents)
            
            if not documents:
                logger.warning("No documents provided for indexing")
                return False
            
            # Insert documents into existing index
            for doc in documents:
                self.index.insert(doc)
            
            logger.info(f"Successfully added {len(documents)} documents to index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {str(e)}")
            return False
    
    def query_index(self, query: str, **kwargs) -> Optional[str]:
        """Query the vector index."""
        try:
            if not self.index:
                logger.error("No index available for querying")
                return None
            
            # Set default query parameters
            query_kwargs = {
                'similarity_top_k': self.config.SIMILARITY_TOP_K,
                'response_mode': 'compact',
                'verbose': True
            }
            query_kwargs.update(kwargs)
            
            # Create query engine
            query_engine = self.index.as_query_engine(**query_kwargs)
            
            # Execute query
            response = query_engine.query(query)
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error querying index: {str(e)}")
            return None
    
    def get_similar_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get similar documents for a query."""
        try:
            if not self.index:
                logger.error("No index available for similarity search")
                return []
            
            # Create retriever
            retriever = self.index.as_retriever(
                similarity_top_k=top_k
            )
            
            # Get similar nodes
            nodes = retriever.retrieve(query)
            
            # Format results
            results = []
            for node in nodes:
                results.append({
                    'text': node.text,
                    'score': node.score,
                    'metadata': node.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar documents: {str(e)}")
            return []
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the index."""
        try:
            if not self.index:
                logger.error("No index available for deletion")
                return False
            
            # Delete documents by ID
            self.index.delete_nodes(doc_ids)
            
            logger.info(f"Successfully deleted {len(doc_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            if not self.index:
                return {"total_documents": 0, "index_exists": False}
            
            # Get collection stats
            collection = self.chroma_client.get_collection(
                name=self.config.CHROMA_COLLECTION_NAME
            )
            
            count = collection.count()
            
            return {
                "total_documents": count,
                "index_exists": True,
                "collection_name": self.config.CHROMA_COLLECTION_NAME,
                "persist_directory": self.config.CHROMA_PERSIST_DIRECTORY
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"total_documents": 0, "index_exists": False, "error": str(e)}
    
    def clear_index(self) -> bool:
        """Clear all documents from the index."""
        try:
            if not self.index:
                logger.warning("No index to clear")
                return True
            
            # Delete the collection
            self.chroma_client.delete_collection(
                name=self.config.CHROMA_COLLECTION_NAME
            )
            
            # Recreate the collection
            collection = self.chroma_client.create_collection(
                name=self.config.CHROMA_COLLECTION_NAME
            )
            
            # Reinitialize vector store
            self.vector_store = ChromaVectorStore(
                chroma_collection=collection
            )
            
            # Reset index
            self.index = None
            
            logger.info("Successfully cleared index")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
            return False
    
    def update_document(self, doc_id: str, new_text: str, metadata: Optional[Dict] = None) -> bool:
        """Update a document in the index."""
        try:
            if not self.index:
                logger.error("No index available for update")
                return False
            
            # For now, delete and re-add the document
            # In a production system, you might want more sophisticated update logic
            self.delete_documents([doc_id])
            
            # Create new document
            from llama_index import Document
            new_doc = Document(
                text=new_text,
                metadata=metadata or {}
            )
            
            # Add to index
            self.add_documents_to_index([new_doc])
            
            logger.info(f"Successfully updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False
    
    def save_index(self) -> bool:
        """Save the index to disk."""
        try:
            if not self.index:
                logger.warning("No index to save")
                return True
            
            # ChromaDB automatically persists data
            logger.info("Index saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self) -> bool:
        """Load existing index from disk."""
        try:
            # Check if collection exists
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.config.CHROMA_COLLECTION_NAME not in collection_names:
                logger.info("No existing index found")
                return False
            
            # Get collection
            collection = self.chroma_client.get_collection(
                name=self.config.CHROMA_COLLECTION_NAME
            )
            
            if collection.count() == 0:
                logger.info("Existing collection is empty")
                return False
            
            # Reinitialize vector store
            self.vector_store = ChromaVectorStore(
                chroma_collection=collection
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Load index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                service_context=self.service_context
            )
            
            logger.info("Successfully loaded existing index")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False 