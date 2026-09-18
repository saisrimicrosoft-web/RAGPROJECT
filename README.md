# RAG-Based Document Q&A System

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF, TXT, DOCX, Markdown) and ask questions about their content. It uses an LLM to generate answers based solely on the retrieved context from the document.

## Architecture / Workflow
1. **Document Ingestion:** Extract text from user-uploaded documents (PDF, TXT, DOCX, MD).
2. **Chunking:** Split the extracted text into smaller, overlapping chunks to fit within LLM context limits and improve retrieval precision.
3. **Embeddings:** Convert text chunks into vector representations using a local HuggingFace embedding model (`all-MiniLM-L6-v2`).
4. **Vector Database / Indexing:** Store the generated embeddings and their metadata in ChromaDB.
5. **Similarity Search (Retrieval):** When a user asks a question, it is embedded, and the vector database is queried to find the most similar chunks (context).
6. **LLM Generation:** The retrieved context and user question are passed to the Gemini LLM with a strict prompt to answer using only the provided context.

## Technologies Used
- **UI:** Terminal / CLI
- **Framework:** LangChain
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** ChromaDB
- **LLM:** Google Gemini (`gemini-1.5-flash` via `langchain-google-genai`)
- **Document Loaders:** PyPDF2, python-docx

## Setup Instructions

1. Clone or download the repository.
2. Ensure you have Python installed (3.8+ recommended).
3. Open a terminal and navigate to the project directory.
4. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on Mac/Linux:
   source venv/bin/activate
   ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Sample Document Used for Testing
The application uses `sample_policy.txt` as a sample document for testing. It contains a mock employee policy with information about working hours and remote work.

## How to Run the Application

You can provide your Google Gemini API key by either setting the `GOOGLE_API_KEY` environment variable, adding it to a `.env` file, passing it via `--api-key`, or entering it interactively.

### 1. Interactive Mode
Run the following command to start the interactive Q&A loop:
```bash
python main.py --file sample_policy.txt
```
The system will extract, chunk, and embed the document. You can then type your questions in the interactive terminal prompt. Type `exit` to quit.

### 2. Direct Query Mode
You can pass arguments directly for a one-off query without entering the interactive loop:
```bash
python main.py --file sample_policy.txt --query "What is the work from home policy?"
```

### Advanced Options
You can configure chunking behavior using the flags:
```bash
python main.py --file sample_policy.txt --chunk_size 1000 --chunk_overlap 200
```

## Explanations

### What Embeddings Are
Embeddings are numerical representations (vectors) of text. They capture the semantic meaning of the text in a high-dimensional space. Words or sentences with similar meanings will have vectors that are closer together. This allows computers to understand the context and relationships between text rather than just matching exact keywords.

### Why a Vector Database is Required
A vector database is required to efficiently store and query these high-dimensional embeddings. Standard relational databases are not designed for similarity searches (finding the "closest" vectors). A vector DB uses specialized algorithms to rapidly compare the question's embedding against millions of document chunk embeddings.

### How Similarity Search Works
When you ask a question, the system converts it into an embedding. It then compares this question vector against all the chunk vectors stored in the vector database using a distance metric (like Cosine Similarity). The database returns the chunks with vectors closest to the question vector, which represent the most semantically relevant text.

### What Chunk Size and Overlap You Selected
- **Chunk Size:** 1000 characters
- **Chunk Overlap:** 200 characters
**Why:** A chunk size of 1000 provides enough context for the LLM to understand a full paragraph or concept without overwhelming the context window. An overlap of 200 ensures that if a sentence or thought spans across a chunk boundary, it isn't lost, allowing the retriever to maintain context continuity.

### How RAG Differs from Simply Asking an LLM a Question
When you simply ask an LLM a question, it relies entirely on its pre-trained, static, general knowledge. It might hallucinate or not know about private/recent data.
In RAG (Retrieval-Augmented Generation), you first retrieve factual, highly relevant information from your own specific documents and inject that context into the LLM's prompt. The LLM acts purely as a reasoning engine to read the context and formulate an answer, ensuring accuracy and citing the source document.

## Screenshots

*(When editing this file on GitHub, you can delete the placeholder text below and simply drag-and-drop your screenshot images directly into the editor!)*

### Document Ingestion
![Document Ingestion Screenshot](replace_this_with_image_url)

### Embedding/Indexing Process
![Embedding/Indexing Screenshot](replace_this_with_image_url)

### Vector Database/Search
![Vector Search Screenshot](replace_this_with_image_url)

### User Query
![User Query Screenshot](replace_this_with_image_url)

### Retrieved Context
![Retrieved Context Screenshot](replace_this_with_image_url)

### Final Generated Answer
![Final Answer Screenshot](replace_this_with_image_url)
