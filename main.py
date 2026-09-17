import os
import sys
import argparse
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="RAG Document Q&A System")
    parser.add_argument("--file", type=str, help="Path to the document (PDF, TXT, DOCX, MD)")
    parser.add_argument("--api-key", type=str, help="Google Gemini API Key (overrides env var)")
    parser.add_argument("--query", type=str, help="Direct question to ask (bypasses interactive loop)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap for text splitting")
    args = parser.parse_args()

    print("="*50)
    print("RAG-Based Document Q&A System (CLI)")
    print("="*50)
    
    # 1. Get API Key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Note: An LLM API Key is needed to generate answers.")
        api_key = input("Please enter your Google Gemini API Key: ").strip()
        if not api_key:
            print("API Key is required to generate answers. Exiting.")
            return

    # 2. Get Document File
    document_path = args.file
    if not document_path:
        document_path = input("Enter the path to your document (e.g., sample_policy.txt): ").strip()
        
    if not os.path.exists(document_path):
        print(f"\nError: File not found at '{document_path}'")
        return

    # 3. Initialize Pipeline
    print(f"\n[1/3] Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        google_api_key=api_key, 
        chunk_size=args.chunk_size, 
        chunk_overlap=args.chunk_overlap
    )

    # 4. Process Document
    print(f"[2/3] Extracting text and chunking document: {document_path}")
    splits = pipeline.process_document(document_path)
    print(f"      -> Created {len(splits)} chunks.")

    # 5. Generate Embeddings & Index
    print(f"[3/3] Generating embeddings and indexing in Vector DB (Chroma)...")
    pipeline.create_vector_store(splits)
    print("      -> Ready!")

    # 6. Execute Query
    if args.query:
        print(f"\nQuestion: {args.query}")
        print("\nSearching and thinking...")
        answer, docs = pipeline.ask_question(args.query)
        print(f"\nAnswer:\n{answer}")
        
        print("\n--- Retrieved Context (Top 4 Chunks) ---")
        for i, doc in enumerate(docs):
            print(f"\n[Chunk {i+1}]")
            print(doc.page_content.strip())
            print("-" * 40)
        return

    # 7. Interactive Q&A Loop
    print("\n" + "="*50)
    print("Document loaded successfully. You can now ask questions!")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if not question:
            continue

        print("\nSearching and thinking...")
        answer, docs = pipeline.ask_question(question)
        
        print(f"\nAnswer:\n{answer}")
        
        print("\n--- Retrieved Context (Top 4 Chunks) ---")
        for i, doc in enumerate(docs):
            print(f"\n[Chunk {i+1}]")
            print(doc.page_content.strip())
            print("-" * 40)

if __name__ == "__main__":
    main()
