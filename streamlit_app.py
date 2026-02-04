import streamlit as st
from pathlib import Path
import sys
import time

#Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

#Page Configuration
st.set_page_config(
    page_title="RAG Search",
    page_icon="📚",
    layout="centered"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>        
""",unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG sysrem cached"""
    try:
        #Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()

        urls = Config.DEFAULT_URLS #use default urls

        #process documents
        documents=doc_processor.load_from_url(urls)

        #create vector store
        vector_store.create_vectorstore(documents)

        #Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder,len(documents)
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None,0
    
def main():
    """Main application"""
    init_session_state()

    #Title
    st.title("📚 RAG document search")
    st.markdown("Ask questions based on the loaded documents.")

    #initialize RAG system
    if not st.session_state.initialized:
        with st.spinner("Initializing RAG system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"RAG system initialized with {num_chunks} documents.")
    
    st.markdown("---")

    #Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:", 
            placeholder="What would like to know?"
            )
        submit = st.form_submit_button("Search")

    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()

                #Get answer
                response = st.session_state.rag_system.run(question)

                elapsed_time = time.time() - start_time

                #Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': response['answer'],
                    'elapsed_time': elapsed_time
                })
                #Display answer
                st.markdown(f"**Answer:**")
                st.success(response['answer'])

                with st.expander("Source documents"):
                    for i,doc in enumerate(response['retrieve_docs'],1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...", #show first 300 chars
                            height=100,
                            disabled=True
                        )
                st.caption(f"Response time: {elapsed_time:.2f} seconds")

    #show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Search History")
        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.caption(f"Response time: {item['elapsed_time']:.2f} seconds")
            st.markdown("---")

if __name__ == "__main__":
    main()


