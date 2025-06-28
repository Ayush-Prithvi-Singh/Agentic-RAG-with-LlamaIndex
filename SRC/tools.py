"""
Custom tools for the Agentic RAG system.
Provides enhanced capabilities for the agent.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .utils import clean_text, create_summary, format_response

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Tool for performing web searches."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform a web search and return results."""
        try:
            # This is a simplified web search implementation
            # In production, you might want to use a proper search API
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract relevant information from DuckDuckGo response
            if 'AbstractText' in data and data['AbstractText']:
                results.append({
                    'title': data.get('AbstractSource', 'DuckDuckGo'),
                    'snippet': data['AbstractText'],
                    'url': data.get('AbstractURL', ''),
                    'source': 'duckduckgo'
                })
            
            # Add related topics
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('FirstURL', '').split('/')[-1],
                        'snippet': topic['Text'],
                        'url': topic.get('FirstURL', ''),
                        'source': 'duckduckgo'
                    })
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            return []
    
    def get_current_time(self) -> str:
        """Get current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class DocumentAnalysisTool:
    """Tool for analyzing documents and extracting insights."""
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze a document and extract key insights."""
        try:
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Basic analysis
            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)
            sentence_count = len([s for s in cleaned_text.split('.') if s.strip()])
            
            # Extract key phrases (simple implementation)
            words = cleaned_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Create summary
            summary = create_summary(cleaned_text, max_words=50)
            
            return {
                'word_count': word_count,
                'character_count': char_count,
                'sentence_count': sentence_count,
                'top_words': top_words,
                'summary': summary,
                'average_sentence_length': round(word_count / sentence_count, 2) if sentence_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {}
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        try:
            # Simple keyword extraction based on frequency
            words = clean_text(text).lower().split()
            word_freq = {}
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in keywords[:num_keywords]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def compare_documents(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two documents and find similarities."""
        try:
            # Extract keywords from both documents
            keywords1 = set(self.extract_keywords(text1))
            keywords2 = set(self.extract_keywords(text2))
            
            # Find common keywords
            common_keywords = keywords1.intersection(keywords2)
            
            # Calculate similarity score
            total_keywords = len(keywords1.union(keywords2))
            similarity_score = len(common_keywords) / total_keywords if total_keywords > 0 else 0
            
            return {
                'similarity_score': round(similarity_score, 3),
                'common_keywords': list(common_keywords),
                'unique_to_doc1': list(keywords1 - keywords2),
                'unique_to_doc2': list(keywords2 - keywords1)
            }
            
        except Exception as e:
            logger.error(f"Error comparing documents: {str(e)}")
            return {}

class DataProcessingTool:
    """Tool for processing and analyzing data."""
    
    def process_csv_data(self, csv_content: str) -> Dict[str, Any]:
        """Process CSV data and extract insights."""
        try:
            lines = csv_content.strip().split('\n')
            if len(lines) < 2:
                return {'error': 'Invalid CSV format'}
            
            # Parse headers
            headers = lines[0].split(',')
            
            # Parse data rows
            data_rows = []
            for line in lines[1:]:
                if line.strip():
                    row = line.split(',')
                    if len(row) == len(headers):
                        data_rows.append(row)
            
            # Basic statistics
            stats = {
                'total_rows': len(data_rows),
                'total_columns': len(headers),
                'headers': headers,
                'sample_data': data_rows[:5] if data_rows else []
            }
            
            # Try to identify data types
            if data_rows:
                column_types = []
                for i in range(len(headers)):
                    sample_values = [row[i] for row in data_rows[:10]]
                    # Simple type detection
                    try:
                        [float(val) for val in sample_values if val.strip()]
                        column_types.append('numeric')
                    except ValueError:
                        column_types.append('text')
                
                stats['column_types'] = column_types
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing CSV data: {str(e)}")
            return {'error': str(e)}
    
    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from text."""
        try:
            # Look for common patterns
            data = {}
            
            # Email addresses
            import re
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                data['emails'] = emails
            
            # Phone numbers
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phones = re.findall(phone_pattern, text)
            if phones:
                data['phone_numbers'] = phones
            
            # URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, text)
            if urls:
                data['urls'] = urls
            
            # Dates
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
            dates = re.findall(date_pattern, text)
            if dates:
                data['dates'] = dates
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            return {}

class FileSystemTool:
    """Tool for file system operations."""
    
    def list_files(self, directory: str) -> List[Dict[str, Any]]:
        """List files in a directory."""
        try:
            if not os.path.exists(directory):
                return []
            
            files = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    stat = os.stat(item_path)
                    files.append({
                        'name': item,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'path': item_path
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read a file and return its contents."""
        try:
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return None
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to a file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing file: {str(e)}")
            return False

class ToolManager:
    """Manages all available tools."""
    
    def __init__(self):
        self.web_search = WebSearchTool()
        self.doc_analysis = DocumentAnalysisTool()
        self.data_processing = DataProcessingTool()
        self.file_system = FileSystemTool()
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about all available tools."""
        return {
            'web_search': {
                'description': 'Search the web for information',
                'methods': ['search', 'get_current_time']
            },
            'document_analysis': {
                'description': 'Analyze documents and extract insights',
                'methods': ['analyze_document', 'extract_keywords', 'compare_documents']
            },
            'data_processing': {
                'description': 'Process and analyze data',
                'methods': ['process_csv_data', 'extract_structured_data']
            },
            'file_system': {
                'description': 'Perform file system operations',
                'methods': ['list_files', 'read_file', 'write_file']
            }
        }
    
    def execute_tool(self, tool_name: str, method_name: str, **kwargs) -> Any:
        """Execute a specific tool method."""
        try:
            if tool_name == 'web_search':
                tool = self.web_search
            elif tool_name == 'document_analysis':
                tool = self.doc_analysis
            elif tool_name == 'data_processing':
                tool = self.data_processing
            elif tool_name == 'file_system':
                tool = self.file_system
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            if not hasattr(tool, method_name):
                raise ValueError(f"Unknown method {method_name} for tool {tool_name}")
            
            method = getattr(tool, method_name)
            return method(**kwargs)
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}.{method_name}: {str(e)}")
            return None 