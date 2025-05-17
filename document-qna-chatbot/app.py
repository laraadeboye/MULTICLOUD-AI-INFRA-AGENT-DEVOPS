import os
import tempfile
import gradio as gr
from typing import List, Dict, Any
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Load environment variables
load_dotenv()

# Configure the LlamaIndex settings
def configure_llama_index():
    """Configure LlamaIndex settings with the appropriate models."""
    # Set up Groq LLM with updated parameters
    # Updated initialization for newer Groq API
    try:
        # Try with model parameter (newer versions)
        llm = Groq(model="llama3-70b-8192", api_key=os.environ.get("GROQ_API_KEY"))
    except TypeError:
        try:
            # Try with model_name parameter (older versions)
            llm = Groq(model_name="llama3-70b-8192", api_key=os.environ.get("GROQ_API_KEY"))
        except TypeError:
            # Fallback to minimal parameters if needed
            llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            print("Warning: Groq initialized with minimal parameters. You may need to set the model manually.")
    
    # Set up Sentence Transformers embedding model
    embed_model = HuggingFaceEmbedding(        
        model_name="all-MiniLM-L6-v2"  # Fast lightweight model with good performance
    )

    
    # Update global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    
    return llm, embed_model

# Process documents and create index
def process_documents(file_paths: List[str]) -> VectorStoreIndex:
    """Process documents using SimpleDirectoryReader and create a vector store index."""
    documents = []
    
    # Handle both files and directories
    input_files = []
    input_dirs = []
    
    for file_path in file_paths:
        if os.path.isdir(file_path):
            input_dirs.append(file_path)
        else:
            input_files.append(file_path)
    
    try:
        # Process directories
        for dir_path in input_dirs:
            reader = SimpleDirectoryReader(input_dir=dir_path)
            docs = reader.load_data()
            documents.extend(docs)
        
        # Process individual files
        if input_files:
            reader = SimpleDirectoryReader(input_files=input_files)
            docs = reader.load_data()
            documents.extend(docs)
            
    except Exception as e:
        print(f"Error processing files: {e}")
    
    if not documents:
        # Create a dummy document if no documents were processed
        documents = [Document(text="No valid documents were processed.")]
    
    # Create index from documents
    index = VectorStoreIndex.from_documents(documents)
    return index

# Query engine setup
def setup_query_engine(index: VectorStoreIndex):
    """Set up the query engine with the vector store index."""
    return index.as_query_engine(
        similarity_top_k=3,  # Retrieve top 3 most relevant chunks
        response_mode="compact"  # Generate a compact response
    )

# Main query function for RAG
def query_with_context(query_text: str, index: VectorStoreIndex) -> str:
    """Query the RAG system with the given text."""
    query_engine = setup_query_engine(index)
    response = query_engine.query(query_text)
    return str(response)

# State management for Gradio
class AppState:
    def __init__(self):
        self.index = None
        self.files_processed = False
        self.configure_llama_index()
    
    def configure_llama_index(self):
        self.llm, self.embed_model = configure_llama_index()
    
    def process_files(self, files: List[str]):
        try:
            self.index = process_documents(files)
            self.files_processed = True
            return "Files processed successfully. You can now ask questions."
        except Exception as e:
            return f"Error processing files: {str(e)}"
    
    def reset(self):
        self.index = None
        self.files_processed = False
        return "Application reset. Please upload new files."

# Create Gradio interface
def create_interface():
    """Create the Gradio interface for the application."""
    state = AppState()
    
    with gr.Blocks(title="Document Question and Answer Chatbot") as interface:
        gr.Markdown("# Document Question and Answer Chatbot")
        gr.Markdown("Upload documents and ask questions based on their content.")
        
        with gr.Row():
            with gr.Column(scale=1):
                files_input = gr.File(
                    file_count="multiple",
                    label="Upload Documents"
                )
                process_button = gr.Button("Process Documents")
                status_output = gr.Textbox(
                    label="Status",
                    value="Upload and process documents to begin.",
                    interactive=False
                )
                reset_button = gr.Button("Reset")
            
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about your documents...",
                    lines=2
                )
                query_button = gr.Button("Ask")
                response_output = gr.Textbox(
                    label="Response",
                    interactive=False,
                    lines=10
                )
        
        # Event handlers
        def process_uploaded_files(files):
            if not files:
                return "No files uploaded. Please upload at least one file."
            
            # Save uploaded files to temp directory and get their paths
            file_paths = []
            for file in files:
                file_paths.append(file.name)
            
            return state.process_files(file_paths)
        
        def query_documents(query):
            if not state.files_processed or state.index is None:
                return "Please process documents first before asking questions."
            
            if not query.strip():
                return "Please enter a question."
            
            try:
                response = query_with_context(query, state.index)
                return response
            except Exception as e:
                return f"Error processing query: {str(e)}"
        
        # Connect event handlers
        process_button.click(
            process_uploaded_files,
            inputs=[files_input],
            outputs=[status_output]
        )
        
        query_button.click(
            query_documents,
            inputs=[query_input],
            outputs=[response_output]
        )
        
        reset_button.click(
            state.reset,
            outputs=[status_output]
        )
    
    return interface

def main():
    """Main function to launch the Gradio interface."""
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()
