import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

class RAGPipeline:
    def __init__(self, google_api_key=None, chunk_size=1000, chunk_overlap=200):
        self.google_api_key = google_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Embeddings (HuggingFace local model)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Vector Store and Retriever
        self.vector_store = None
        self.retriever = None

        # LLM
        self.model_name = os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash")
        if self.google_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, 
                google_api_key=self.google_api_key,
                temperature=0
            )
        else:
            self.llm = None

    def update_api_key(self, api_key):
        self.google_api_key = api_key
        if self.google_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, 
                google_api_key=self.google_api_key,
                temperature=0
            )

    def process_document(self, file_path):
        """Extract text and process a document into chunks."""
        suffix = os.path.splitext(file_path)[1]
        
        docs = []
        if suffix.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif suffix.lower() == '.docx':
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        elif suffix.lower() in ['.txt', '.md']:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Chunking: split the extracted text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        splits = text_splitter.split_documents(docs)
        return splits

    def create_vector_store(self, splits):
        """Create a vector database from document chunks."""
        # Generate embeddings and store in ChromaDB
        self.vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        return self.vector_store

    def ask_question(self, question):
        """Query the vector database and generate an answer."""
        if not self.vector_store:
            return "Please upload and process a document first.", []
        if not self.llm:
            return "Please provide a Google API Key via environment variable or CLI argument to generate an answer.", []

        # 1. Similarity Search: Retrieve most relevant chunks based on question embedding
        docs = self.retriever.invoke(question)
        
        # 2. Setup Prompt based on assignment requirement
        prompt_template = """
        You are a helpful assistant for answering questions based on the provided document context.
        
        Instructions:
        Answer the question using the provided context. If the answer cannot be found in the 
        context, clearly state that the information is not available in the document.

        Context:
        {context}

        Question:
        {input}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "input"]
        )

        # Create chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

        # 3. Pass retrieved content to LLM and get final answer
        response = retrieval_chain.invoke({"input": question})
        
        return response['answer'], docs
