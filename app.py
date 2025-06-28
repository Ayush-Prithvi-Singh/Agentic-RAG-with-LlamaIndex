"""
Streamlit web application for the Agentic RAG system.
Provides a user-friendly interface for document management and querying.
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

# Import our custom modules
from src.agent import AgenticRAG
from config import Config

# Page configuration
st.set_page_config(
    page_title="Agentic RAG with LlamaIndex",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the RAG system."""
    try:
        config = Config()
        rag = AgenticRAG(config)
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic RAG with LlamaIndex</h1>', unsafe_allow_html=True)
    
    # Initialize RAG system using session state
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = initialize_rag_system()
    
    rag = st.session_state.rag_system
    if rag is None:
        st.error("Failed to initialize the RAG system. Please check your configuration.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page",
            ["🏠 Dashboard", "📚 Document Management", "❓ Query System", "🔧 System Tools", "📊 Analytics"]
        )
        
        st.header("System Status")
        try:
            stats = rag.get_system_stats()
            if "error" not in stats:
                st.success("✅ System Operational")
                st.metric("Documents", stats.get("index_stats", {}).get("total_documents", 0))
            else:
                st.error("❌ System Error")
        except Exception as e:
            st.error(f"❌ System Error: {str(e)}")
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard(rag)
    elif page == "📚 Document Management":
        show_document_management(rag)
    elif page == "❓ Query System":
        show_query_system(rag)
    elif page == "🔧 System Tools":
        show_system_tools(rag)
    elif page == "📊 Analytics":
        show_analytics(rag)

def show_dashboard(rag):
    """Show the main dashboard."""
    st.header("🏠 Dashboard")
    
    # System overview
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        stats = rag.get_system_stats()
        
        with col1:
            st.metric(
                "Total Documents",
                stats.get("index_stats", {}).get("total_documents", 0)
            )
        
        with col2:
            st.metric(
                "Available Tools",
                len(stats.get("available_tools", {}))
            )
        
        with col3:
            st.metric(
                "Model",
                stats.get("config", {}).get("model", "Unknown")
            )
        
        with col4:
            st.metric(
                "System Status",
                "🟢 Operational" if "error" not in stats else "🔴 Error"
            )
    
    except Exception as e:
        st.error(f"Error loading system stats: {str(e)}")
    
    # Quick actions
    st.header("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Add Documents", use_container_width=True):
            st.session_state.show_upload = True
            st.rerun()
    
    with col2:
        if st.button("❓ Ask Question", use_container_width=True):
            st.session_state.show_query = True
            st.rerun()
    
    with col3:
        if st.button("🔧 System Tools", use_container_width=True):
            st.session_state.show_tools = True
            st.rerun()
    
    # Recent activity (placeholder)
    st.header("Recent Activity")
    st.info("Recent activity tracking will be implemented in future versions.")
    
    # System information
    st.header("System Information")
    
    try:
        config_info = stats.get("config", {})
        st.json(config_info)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")

def show_document_management(rag):
    """Show document management interface."""
    st.header("📚 Document Management")
    
    # File upload
    st.subheader("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['txt', 'pdf', 'docx', 'csv', 'json', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files:")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")
        
        if st.button("📥 Add to Knowledge Base"):
            with st.spinner("Processing documents..."):
                # Save uploaded files temporarily
                temp_files = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_files.append(tmp_file.name)
                
                # Add documents to RAG system
                result = rag.add_documents(temp_files)
                
                # Clean up temporary files
                for temp_file in temp_files:
                    os.unlink(temp_file)
                
                if result.get("success"):
                    st.success(result.get("message", "Documents added successfully!"))
                    if "summary" in result:
                        st.json(result["summary"])
                else:
                    st.error(result.get("message", "Failed to add documents"))
    
    # Directory upload
    st.subheader("Upload from Directory")
    
    directory_path = st.text_input("Enter directory path:")
    if directory_path and st.button("📁 Add Directory"):
        if os.path.exists(directory_path):
            with st.spinner("Processing directory..."):
                result = rag.add_documents([directory_path])
                if result.get("success"):
                    st.success(result.get("message", "Directory processed successfully!"))
                    if "summary" in result:
                        st.json(result["summary"])
                else:
                    st.error(result.get("message", "Failed to process directory"))
        else:
            st.error("Directory does not exist")
    
    # Document management
    st.subheader("Manage Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.checkbox("I understand this will delete all documents"):
                with st.spinner("Clearing documents..."):
                    success = rag.clear_knowledge_base()
                    if success:
                        st.success("All documents cleared successfully!")
                    else:
                        st.error("Failed to clear documents")
    
    with col2:
        if st.button("💾 Save Knowledge Base"):
            with st.spinner("Saving..."):
                success = rag.save_knowledge_base()
                if success:
                    st.success("Knowledge base saved successfully!")
                else:
                    st.error("Failed to save knowledge base")
    
    # Document summary
    st.subheader("Document Summary")
    try:
        summary = rag.get_document_summary()
        st.json(summary)
    except Exception as e:
        st.error(f"Error loading document summary: {str(e)}")

def show_query_system(rag):
    """Show the query interface."""
    st.header("❓ Query System")
    
    # Query input
    query = st.text_area("Enter your question:", height=100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_agent = st.checkbox("Use Agentic Approach", value=True, help="Use the AI agent for more intelligent responses")
    
    with col2:
        if st.button("🔍 Ask Question", type="primary"):
            if query.strip():
                with st.spinner("Processing your question..."):
                    result = rag.query(query, use_agent=use_agent)
                    
                    if result.get("success"):
                        st.success("✅ Answer Generated")
                        
                        # Display answer
                        st.subheader("Answer")
                        st.write(result.get("answer", ""))
                        
                        # Display metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Method", result.get("method", "Unknown"))
                        with col2:
                            st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                        with col3:
                            st.metric("Similar Documents", len(result.get("similar_documents", [])))
                        
                        # Show similar documents
                        similar_docs = result.get("similar_documents", [])
                        if similar_docs:
                            st.subheader("Similar Documents")
                            for i, doc in enumerate(similar_docs[:3]):
                                with st.expander(f"Document {i+1} (Score: {doc.get('score', 0):.3f})"):
                                    st.write(doc.get('text', '')[:500] + "...")
                                    st.json(doc.get('metadata', {}))
                    else:
                        st.error("❌ Failed to generate answer")
                        st.write(result.get("answer", "An error occurred"))
            else:
                st.warning("Please enter a question")
    
    # Advanced query options
    st.subheader("Advanced Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Get Similar Documents"):
            if query.strip():
                with st.spinner("Finding similar documents..."):
                    similar_docs = rag.get_similar_documents(query, top_k=5)
                    if similar_docs:
                        st.subheader("Similar Documents")
                        for i, doc in enumerate(similar_docs):
                            with st.expander(f"Document {i+1} (Score: {doc.get('score', 0):.3f})"):
                                st.write(doc.get('text', '')[:300] + "...")
                    else:
                        st.info("No similar documents found")
            else:
                st.warning("Please enter a query first")
    
    with col2:
        if st.button("🌐 Web Search"):
            if query.strip():
                with st.spinner("Searching the web..."):
                    web_results = rag.search_web(query, max_results=3)
                    if web_results:
                        st.subheader("Web Search Results")
                        for i, result in enumerate(web_results):
                            with st.expander(f"Result {i+1}: {result.get('title', 'Unknown')}"):
                                st.write(result.get('snippet', ''))
                                if result.get('url'):
                                    st.write(f"URL: {result['url']}")
                    else:
                        st.info("No web results found")
            else:
                st.warning("Please enter a query first")

def show_system_tools(rag):
    """Show system tools and utilities."""
    st.header("🔧 System Tools")
    
    # Tool categories
    tab1, tab2, tab3, tab4 = st.tabs(["Document Analysis", "Data Processing", "File System", "System Info"])
    
    with tab1:
        st.subheader("Document Analysis")
        
        text_to_analyze = st.text_area("Enter text to analyze:", height=200)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Analyze Document"):
                if text_to_analyze.strip():
                    with st.spinner("Analyzing document..."):
                        analysis = rag.analyze_document(text_to_analyze)
                        if analysis:
                            st.subheader("Analysis Results")
                            st.json(analysis)
                        else:
                            st.error("Failed to analyze document")
                else:
                    st.warning("Please enter text to analyze")
        
        with col2:
            if st.button("🔑 Extract Keywords"):
                if text_to_analyze.strip():
                    with st.spinner("Extracting keywords..."):
                        keywords = rag.tool_manager.doc_analysis.extract_keywords(text_to_analyze)
                        if keywords:
                            st.subheader("Keywords")
                            st.write(keywords)
                        else:
                            st.error("Failed to extract keywords")
                else:
                    st.warning("Please enter text first")
    
    with tab2:
        st.subheader("Data Processing")
        
        csv_data = st.text_area("Enter CSV data:", height=200, help="Paste CSV data here")
        
        if st.button("📈 Process CSV"):
            if csv_data.strip():
                with st.spinner("Processing CSV data..."):
                    stats = rag.tool_manager.data_processing.process_csv_data(csv_data)
                    if "error" not in stats:
                        st.subheader("CSV Statistics")
                        st.json(stats)
                    else:
                        st.error(f"Error processing CSV: {stats['error']}")
            else:
                st.warning("Please enter CSV data")
    
    with tab3:
        st.subheader("File System")
        
        directory_path = st.text_input("Enter directory path:")
        
        if st.button("📁 List Files"):
            if directory_path:
                with st.spinner("Listing files..."):
                    files = rag.tool_manager.file_system.list_files(directory_path)
                    if files:
                        st.subheader("Files in Directory")
                        df = pd.DataFrame(files)
                        st.dataframe(df)
                    else:
                        st.info("No files found or directory doesn't exist")
            else:
                st.warning("Please enter a directory path")
    
    with tab4:
        st.subheader("System Information")
        
        try:
            stats = rag.get_system_stats()
            st.json(stats)
        except Exception as e:
            st.error(f"Error loading system stats: {str(e)}")

def show_analytics(rag):
    """Show analytics and insights."""
    st.header("📊 Analytics")
    
    # System performance metrics
    st.subheader("System Performance")
    
    try:
        stats = rag.get_system_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Documents",
                stats.get("index_stats", {}).get("total_documents", 0)
            )
        
        with col2:
            st.metric(
                "Available Tools",
                len(stats.get("available_tools", {}))
            )
        
        with col3:
            st.metric(
                "System Status",
                "🟢 Operational" if "error" not in stats else "🔴 Error"
            )
        
        # Configuration details
        st.subheader("Configuration Details")
        config = stats.get("config", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Configuration:**")
            st.write(f"- LLM Model: {config.get('model', 'Unknown')}")
            st.write(f"- Embedding Model: {config.get('embedding_model', 'Unknown')}")
        
        with col2:
            st.write("**Processing Configuration:**")
            st.write(f"- Chunk Size: {config.get('chunk_size', 'Unknown')}")
            st.write(f"- Chunk Overlap: {config.get('chunk_overlap', 'Unknown')}")
        
        # Available tools
        st.subheader("Available Tools")
        tools = stats.get("available_tools", {})
        
        for tool_name, tool_info in tools.items():
            with st.expander(f"🔧 {tool_name.title()}"):
                st.write(f"**Description:** {tool_info.get('description', 'No description')}")
                st.write(f"**Methods:** {', '.join(tool_info.get('methods', []))}")
    
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")
    
    # Usage statistics (placeholder)
    st.subheader("Usage Statistics")
    st.info("Usage statistics tracking will be implemented in future versions.")

if __name__ == "__main__":
    main() 