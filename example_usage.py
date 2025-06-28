"""
Example usage of the Agentic RAG system.
Demonstrates how to use the system programmatically.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from src.agent import AgenticRAG
from config import Config

def main():
    """Main example function."""
    print("🤖 Agentic RAG with LlamaIndex - Example Usage")
    print("=" * 50)
    
    # Initialize the system
    print("\n1. Initializing Agentic RAG system...")
    try:
        config = Config()
        rag = AgenticRAG(config)
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        print("Please make sure you have set up your OpenAI API key in the .env file")
        return
    
    # Add sample documents
    print("\n2. Adding sample documents...")
    sample_doc_path = "data/sample_document.txt"
    
    if os.path.exists(sample_doc_path):
        result = rag.add_documents([sample_doc_path])
        if result.get("success"):
            print("✅ Sample document added successfully!")
            if "summary" in result:
                print(f"   - {result['summary']['total_documents']} documents processed")
                print(f"   - Total size: {result['summary']['total_size_mb']} MB")
        else:
            print(f"❌ Failed to add documents: {result.get('message', 'Unknown error')}")
    else:
        print("⚠️  Sample document not found, skipping document addition")
    
    # Example queries
    print("\n3. Testing queries...")
    
    example_queries = [
        "What is artificial intelligence?",
        "What are the different types of AI?",
        "What are the ethical considerations of AI?",
        "How does machine learning work?",
        "What are the applications of AI in healthcare?"
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n   Query {i}: {query}")
        print("   " + "-" * 40)
        
        try:
            result = rag.query(query, use_agent=True)
            
            if result.get("success"):
                print(f"   ✅ Answer ({result.get('method', 'Unknown')}):")
                answer = result.get("answer", "")
                # Truncate long answers for display
                if len(answer) > 300:
                    answer = answer[:300] + "..."
                print(f"   {answer}")
                print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            else:
                print(f"   ❌ Failed: {result.get('answer', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Test document analysis
    print("\n4. Testing document analysis...")
    
    sample_text = """
    Artificial Intelligence is transforming the world around us. From voice assistants 
    to autonomous vehicles, AI technologies are becoming increasingly prevalent in our 
    daily lives. Machine learning algorithms can now process vast amounts of data to 
    identify patterns and make predictions with remarkable accuracy.
    """
    
    try:
        analysis = rag.analyze_document(sample_text)
        if analysis:
            print("✅ Document analysis completed!")
            print(f"   - Word count: {analysis.get('word_count', 0)}")
            print(f"   - Character count: {analysis.get('character_count', 0)}")
            print(f"   - Sentence count: {analysis.get('sentence_count', 0)}")
            print(f"   - Top keywords: {[word for word, freq in analysis.get('top_words', [])[:5]]}")
        else:
            print("❌ Document analysis failed")
    except Exception as e:
        print(f"❌ Error in document analysis: {str(e)}")
    
    # Test web search
    print("\n5. Testing web search...")
    
    try:
        web_results = rag.search_web("latest developments in AI", max_results=2)
        if web_results:
            print("✅ Web search completed!")
            for i, result in enumerate(web_results, 1):
                print(f"   Result {i}: {result.get('title', 'Unknown')}")
                snippet = result.get('snippet', '')[:100] + "..." if len(result.get('snippet', '')) > 100 else result.get('snippet', '')
                print(f"   {snippet}")
        else:
            print("❌ No web search results found")
    except Exception as e:
        print(f"❌ Error in web search: {str(e)}")
    
    # Test similar documents
    print("\n6. Testing similar document retrieval...")
    
    try:
        similar_docs = rag.get_similar_documents("machine learning", top_k=2)
        if similar_docs:
            print("✅ Similar documents found!")
            for i, doc in enumerate(similar_docs, 1):
                print(f"   Document {i} (Score: {doc.get('score', 0):.3f}):")
                text_preview = doc.get('text', '')[:150] + "..." if len(doc.get('text', '')) > 150 else doc.get('text', '')
                print(f"   {text_preview}")
        else:
            print("❌ No similar documents found")
    except Exception as e:
        print(f"❌ Error retrieving similar documents: {str(e)}")
    
    # System statistics
    print("\n7. System statistics...")
    
    try:
        stats = rag.get_system_stats()
        if "error" not in stats:
            print("✅ System statistics:")
            print(f"   - Total documents: {stats.get('index_stats', {}).get('total_documents', 0)}")
            print(f"   - Available tools: {len(stats.get('available_tools', {}))}")
            print(f"   - Model: {stats.get('config', {}).get('model', 'Unknown')}")
            print(f"   - Status: {stats.get('system_status', 'Unknown')}")
        else:
            print(f"❌ Error getting system stats: {stats.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error getting system statistics: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 Example usage completed!")
    print("\nTo run the web interface, use: streamlit run app.py")

def interactive_mode():
    """Interactive mode for testing queries."""
    print("\n🔍 Interactive Query Mode")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    try:
        config = Config()
        rag = AgenticRAG(config)
        print("✅ System ready for interactive queries!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        return
    
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("🤔 Processing...")
            result = rag.query(query, use_agent=True)
            
            if result.get("success"):
                print(f"\n✅ Answer ({result.get('method', 'Unknown')}):")
                print(result.get("answer", ""))
                print(f"\n⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
            else:
                print(f"\n❌ Failed: {result.get('answer', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic RAG Example Usage")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        main() 