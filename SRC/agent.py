"""
Main Agentic RAG agent implementation.
Combines document loading, vector storage, and tools for intelligent question answering.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool, BaseTool

from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager
from .tools import ToolManager
from .utils import clean_text, format_response, create_summary
from config import Config

logger = logging.getLogger(__name__)

class AgenticRAG:
    """Main Agentic RAG system that combines all components."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Validate configuration
        self.config.validate()
        
        # Initialize components
        self.document_loader = DocumentLoader(self.config)
        self.vector_store = VectorStoreManager(self.config)
        self.tool_manager = ToolManager()
        
        # Initialize LLM
        self.llm = OpenAI(
            model=self.config.OPENAI_MODEL,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS
        )
        
        # Initialize agent
        self.agent = None
        self._initialize_agent()
        
        # Load existing index if available
        self._load_existing_index()
        
        logger.info("Agentic RAG system initialized successfully")
    
    def _initialize_agent(self):
        """Initialize the OpenAI agent with custom tools."""
        try:
            # Create function tools from our custom tools
            tools = self._create_function_tools()
            
            # Initialize agent
            self.agent = OpenAIAgent.from_tools(
                tools=tools,
                llm=self.llm,
                verbose=True,
                system_prompt=self._get_system_prompt()
            )
            
            logger.info("Agent initialized with custom tools")
            
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise
    
    def _create_function_tools(self) -> List[BaseTool]:
        """Create function tools from custom tools."""
        tools = []
        
        # Web search tool
        def web_search(query: str, max_results: int = 5) -> str:
            """Search the web for information."""
            results = self.tool_manager.web_search.search(query, max_results)
            return format_response(str(results))
        
        tools.append(FunctionTool.from_defaults(
            fn=web_search,
            name="web_search",
            description="Search the web for current information about a topic"
        ))
        
        # Document analysis tool
        def analyze_document(text: str) -> str:
            """Analyze a document and extract insights."""
            analysis = self.tool_manager.doc_analysis.analyze_document(text)
            return format_response(str(analysis))
        
        tools.append(FunctionTool.from_defaults(
            fn=analyze_document,
            name="analyze_document",
            description="Analyze a document and extract key insights, keywords, and statistics"
        ))
        
        # Keyword extraction tool
        def extract_keywords(text: str, num_keywords: int = 10) -> str:
            """Extract keywords from text."""
            keywords = self.tool_manager.doc_analysis.extract_keywords(text, num_keywords)
            return format_response(str(keywords))
        
        tools.append(FunctionTool.from_defaults(
            fn=extract_keywords,
            name="extract_keywords",
            description="Extract important keywords from text"
        ))
        
        # Document comparison tool
        def compare_documents(text1: str, text2: str) -> str:
            """Compare two documents and find similarities."""
            comparison = self.tool_manager.doc_analysis.compare_documents(text1, text2)
            return format_response(str(comparison))
        
        tools.append(FunctionTool.from_defaults(
            fn=compare_documents,
            name="compare_documents",
            description="Compare two documents and find similarities and differences"
        ))
        
        # Data processing tool
        def process_csv_data(csv_content: str) -> str:
            """Process CSV data and extract insights."""
            stats = self.tool_manager.data_processing.process_csv_data(csv_content)
            return format_response(str(stats))
        
        tools.append(FunctionTool.from_defaults(
            fn=process_csv_data,
            name="process_csv_data",
            description="Process CSV data and extract statistics and insights"
        ))
        
        # File system tools
        def list_files(directory: str) -> str:
            """List files in a directory."""
            files = self.tool_manager.file_system.list_files(directory)
            return format_response(str(files))
        
        tools.append(FunctionTool.from_defaults(
            fn=list_files,
            name="list_files",
            description="List files in a directory"
        ))
        
        def read_file(file_path: str) -> str:
            """Read a file and return its contents."""
            content = self.tool_manager.file_system.read_file(file_path)
            return content if content else "File not found or could not be read"
        
        tools.append(FunctionTool.from_defaults(
            fn=read_file,
            name="read_file",
            description="Read a file and return its contents"
        ))
        
        return tools
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are an intelligent AI assistant with access to a comprehensive knowledge base and various tools. Your capabilities include:

1. **Document Analysis**: You can analyze documents, extract keywords, and compare documents for similarities.
2. **Web Search**: You can search the web for current information when needed.
3. **Data Processing**: You can process and analyze structured data like CSV files.
4. **File System Access**: You can read files and list directory contents.
5. **Knowledge Base**: You have access to a vector database containing indexed documents.

When answering questions:
- First, try to answer using your knowledge base (the indexed documents)
- If you need current information, use web search
- Use document analysis tools to provide deeper insights
- Be thorough but concise in your responses
- Always cite your sources when possible
- If you're unsure about something, say so rather than guessing

Your goal is to provide accurate, helpful, and well-reasoned responses to user queries."""
    
    def _load_existing_index(self):
        """Load existing index if available."""
        try:
            if self.vector_store.load_index():
                logger.info("Successfully loaded existing index")
            else:
                logger.info("No existing index found")
        except Exception as e:
            logger.warning(f"Could not load existing index: {str(e)}")
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the knowledge base."""
        try:
            documents = []
            
            for file_path in file_paths:
                if os.path.isfile(file_path):
                    doc = self.document_loader.load_document(file_path)
                    if doc:
                        documents.append(doc)
                elif os.path.isdir(file_path):
                    dir_docs = self.document_loader.load_documents_from_directory(file_path)
                    documents.extend(dir_docs)
            
            if not documents:
                return {"success": False, "message": "No valid documents found"}
            
            # Validate documents
            valid_documents = self.document_loader.validate_documents(documents)
            
            if not valid_documents:
                return {"success": False, "message": "No valid documents after validation"}
            
            # Add to vector store
            success = self.vector_store.add_documents_to_index(valid_documents)
            
            if success:
                summary = self.document_loader.get_document_summary(valid_documents)
                return {
                    "success": True,
                    "message": f"Successfully added {len(valid_documents)} documents",
                    "summary": summary
                }
            else:
                return {"success": False, "message": "Failed to add documents to index"}
                
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def query(self, question: str, use_agent: bool = True) -> Dict[str, Any]:
        """Query the knowledge base with a question."""
        try:
            start_time = datetime.now()
            
            if use_agent and self.agent:
                # Use agentic approach
                response = self.agent.chat(question)
                answer = str(response)
                method = "agentic"
            else:
                # Use direct vector search
                answer = self.vector_store.query_index(question)
                method = "vector_search"
            
            if not answer:
                return {
                    "success": False,
                    "answer": "I couldn't find a relevant answer to your question.",
                    "method": method,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Get similar documents for context
            similar_docs = self.vector_store.get_similar_documents(question, top_k=3)
            
            return {
                "success": True,
                "answer": answer,
                "method": method,
                "similar_documents": similar_docs,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return {
                "success": False,
                "answer": f"An error occurred while processing your question: {str(e)}",
                "method": "error",
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def get_similar_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get similar documents for a query."""
        return self.vector_store.get_similar_documents(query, top_k)
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze a document using the analysis tool."""
        return self.tool_manager.doc_analysis.analyze_document(text)
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information."""
        return self.tool_manager.web_search.search(query, max_results)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            index_stats = self.vector_store.get_index_stats()
            tool_info = self.tool_manager.get_available_tools()
            
            return {
                "index_stats": index_stats,
                "available_tools": tool_info,
                "config": {
                    "model": self.config.OPENAI_MODEL,
                    "embedding_model": self.config.OPENAI_EMBEDDING_MODEL,
                    "chunk_size": self.config.CHUNK_SIZE,
                    "chunk_overlap": self.config.CHUNK_OVERLAP
                },
                "system_status": "operational"
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from the knowledge base."""
        return self.vector_store.clear_index()
    
    def save_knowledge_base(self) -> bool:
        """Save the knowledge base to disk."""
        return self.vector_store.save_index()
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get a summary of all documents in the knowledge base."""
        try:
            stats = self.vector_store.get_index_stats()
            return {
                "total_documents": stats.get("total_documents", 0),
                "index_exists": stats.get("index_exists", False),
                "collection_name": stats.get("collection_name", ""),
                "persist_directory": stats.get("persist_directory", "")
            }
        except Exception as e:
            logger.error(f"Error getting document summary: {str(e)}")
            return {"error": str(e)}
    
    def process_query_with_tools(self, query: str) -> Dict[str, Any]:
        """Process a query using available tools."""
        try:
            # This is a more advanced method that can use multiple tools
            # based on the query content
            
            result = {
                "query": query,
                "tools_used": [],
                "results": {},
                "final_answer": ""
            }
            
            # Check if query needs web search
            if any(keyword in query.lower() for keyword in ['current', 'latest', 'recent', 'today', 'now']):
                web_results = self.search_web(query, max_results=3)
                result["tools_used"].append("web_search")
                result["results"]["web_search"] = web_results
            
            # Check if query is about document analysis
            if any(keyword in query.lower() for keyword in ['analyze', 'summary', 'keywords', 'statistics']):
                # This would need document content to analyze
                result["tools_used"].append("document_analysis")
                result["results"]["document_analysis"] = "Document analysis tool available"
            
            # Get vector search results
            vector_answer = self.vector_store.query_index(query)
            if vector_answer:
                result["results"]["vector_search"] = vector_answer
            
            # Combine results
            if result["results"]:
                result["final_answer"] = self._combine_results(result["results"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query with tools: {str(e)}")
            return {"error": str(e)}
    
    def _combine_results(self, results: Dict[str, Any]) -> str:
        """Combine results from different tools into a coherent answer."""
        combined = []
        
        if "vector_search" in results:
            combined.append(f"From knowledge base: {results['vector_search']}")
        
        if "web_search" in results:
            web_info = results["web_search"]
            if web_info:
                combined.append("Additional web information:")
                for item in web_info[:2]:  # Limit to 2 results
                    combined.append(f"- {item.get('title', 'Unknown')}: {item.get('snippet', '')[:200]}...")
        
        return "\n\n".join(combined) if combined else "No relevant information found." 