import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

# Remove the dotenv loading since we'll get API keys from user input
# from dotenv import load_dotenv
# load_dotenv()
# os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
# os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
# groq_api_key=os.getenv('GROQ_API_KEY')

# Set page config
st.set_page_config(page_title="RAG Document Q&A Chatbot", layout="wide")

# Title
st.title("📚 RAG Document Q&A Chatbot")
st.markdown("Upload your PDF documents and ask questions about them!")

# Sidebar for API keys
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # API Keys input
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key for embeddings")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key for the LLM")
    
    # Model selection
    st.header("🤖 Model Configuration")
    groq_model = st.selectbox(
        "Select Groq Model",
        ["Llama3-8b-8192", "Llama3-70b-8192", "Mixtral-8x7b-32768", "Gemma2-9b-it"],
        help="Choose the Groq model for text generation"
    )

# Check if API keys are provided
if not openai_api_key or not groq_api_key:
    st.warning("⚠️ Please provide both OpenAI and Groq API keys in the sidebar to continue.")
    st.stop()

# Initialize LLM with user-provided API key
llm = ChatGroq(groq_api_key=groq_api_key, model=groq_model)

prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    If the question is not related to the context, please say "I don't know".
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:"""
)

def create_vector_embedding(uploaded_files):
    """Create vector embeddings from uploaded PDF files"""
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
        return False
    
    try:
        with st.spinner("Processing PDF documents..."):
            # Create temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to temporary directory
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load documents
                loader = PyPDFDirectoryLoader(temp_dir)
                docs = loader.load()
                
                if not docs:
                    st.error("No text could be extracted from the uploaded PDFs.")
                    return False
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                final_documents = text_splitter.split_documents(docs)
                
                # Create embeddings
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vector_store = FAISS.from_documents(final_documents, embeddings)
                
                # Store in session state
                st.session_state.vector = vector_store
                st.session_state.docs_processed = len(docs)
                st.session_state.chunks_created = len(final_documents)
                
        return True
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

# File upload section
st.header("📄 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF files", 
    type=['pdf'], 
    accept_multiple_files=True,
    help="Upload one or more PDF files to analyze"
)

# Process documents button
if uploaded_files and st.button("🔍 Process Documents", type="primary"):
    if create_vector_embedding(uploaded_files):
        st.success(f"✅ Successfully processed {st.session_state.docs_processed} documents into {st.session_state.chunks_created} chunks!")
        st.session_state.documents_processed = True

# Chat interface
st.header("💬 Ask Questions")

# Check if documents have been processed
if 'documents_processed' not in st.session_state or not st.session_state.documents_processed:
    st.info("👆 Please upload and process your PDF documents first before asking questions.")
else:
    # User input
    user_prompt = st.text_input(
        "Enter your question about the uploaded documents:",
        placeholder="e.g., What are the main findings in the research papers?"
    )
    
    if user_prompt and st.button("🚀 Get Answer", type="primary"):
        if 'vector' in st.session_state:
            try:
                with st.spinner("Generating answer..."):
                    # Create chains
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = st.session_state.vector.as_retriever(search_kwargs={"k": 3})
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    
                    # Get response
                    response = retrieval_chain.invoke({'input': user_prompt})
                    
                    # Display answer
                    st.subheader("🤖 Answer")
                    st.write(response['answer'])
                    
                    # Display source documents
                    with st.expander("📖 Source Documents"):
                        st.write("**Retrieved document chunks:**")
                        for i, doc in enumerate(response['context'], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.write(doc.page_content)
                            st.markdown("---")
                            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
        else:
            st.error("Vector store not found. Please process documents again.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and Groq")
