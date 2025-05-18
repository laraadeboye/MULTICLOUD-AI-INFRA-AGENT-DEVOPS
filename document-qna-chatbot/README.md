# Document Question and Answer LLM Application

A context-aware LLM application that uses Gradio Blocks, Groq Cloud, and LlamaIndex to provide accurate and relevant responses to user queries based on uploaded documents.

## Architecture

This application implements a Retrieval-Augmented Generation (RAG) pipeline with the following components:

1. **User Interface**: Gradio Blocks for a clean, intuitive interface
2. **Document Processing**:  Uses LlamaIndex to parse, chunk, and index document content
3. **Embeddings**: Utilizes HuggingFace's Sentence Transformers for vector representations
4. **LLM Responses**: Groq Cloud for fast, high-quality responses
5. **RAG Pipeline**: LlamaIndex for orchestrating the AI services

## Prerequisites

- Python 3.10
- Docker (optional, for containerized deployment)
- API keys for:
  - Groq Cloud
  - LlamaCloud

## Setup

### Local Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/context-aware-llm-app.git
   cd context-aware-llm-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   cp .env-example .env
   ```
   Then edit the `.env` file with your actual API keys.

5. Run the application:
   ```
   python3 app.py
   ```

6. Open your browser and navigate to http://localhost:7860 to use the application.

### Docker Setup

1. Build the Docker image:
   ```
   docker build -t document-qna-chatbot .
   ```

2. Run the Docker container:
   ```
   docker run -p 7860:7860 --env-file .env document-qna-chatbot
   ```

3. Open your browser and navigate to http://localhost:7860 to use the application.

## Usage

1. Upload one or more documents using the file upload section.
2. Click "Process Documents" to extract text and create embeddings.
3. Type your question in the query box and click "Ask".
4. The application will retrieve relevant context from your documents and provide an informed response.
5. Use the "Reset" button to clear the current session and upload new documents.

## Supported Document Types

Llamaindex [SimpleDirectoryReader](https://docs.llamaindex.ai/en/v0.10.18/module_guides/loading/simpledirectoryreader.html) supports various document types, including:
- PDF
- DOCX
- PPTX
- TXT
- CSV
- And more

## Limitations

- Large documents may take longer to process.
- The accuracy of responses depends on the quality and relevance of the uploaded documents.
- Document Format Support: The implementation relies on LlamaIndex's SimpleDirectoryReader, which supports common document formats but may have limitations with specialized formats
- Scalability: The in-memory vector store is suitable for moderate document volumes but may need optimization for large-scale deployments
- Error Handling: Basic error handling is implemented, but more robust error management could be added for production use
- Security: The application uses environment variables for API keys, but additional security measures would be necessary for a production deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
