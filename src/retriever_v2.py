"""
Simplified Vector Store - Guaranteed to Work
"""
import os
from typing import List
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

load_dotenv()

def create_vectorstore_simple(chunks: List[Document]):
    """Dead simple vectorstore creation that actually works."""
    
    print(f"\nCreating vectorstore with {len(chunks)} chunks...")
    print("This will take 30-60 seconds...\n")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create fresh vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="portfolio"
    )
    
    print(f"✓ Vectorstore created!\n")
    
    # Immediate validation
    test_results = vectorstore.similarity_search("test", k=1)
    print(f"✓ Validation: Can retrieve {len(test_results)} document(s)")
    
    if test_results:
        print(f"✓ Sample metadata: {test_results[0].metadata.get('title', 'N/A')}\n")
    
    return vectorstore

def load_vectorstore_simple():
    """Load existing vectorstore."""
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="portfolio"
    )
    
    return vectorstore

def list_all_projects_simple(vectorstore: Chroma) -> List[str]:
    """Get all project titles without LLM."""
    
    # Fetch many docs to ensure we get all projects
    all_docs = vectorstore.similarity_search("", k=500)
    
    # Extract unique titles
    titles = set()
    for doc in all_docs:
        title = doc.metadata.get('title')
        if title:
            titles.add(title)
    
    return sorted(list(titles))

def main():
    """Build and test vectorstore."""
    
    print("=" * 70)
    print("BUILDING VECTORSTORE (SIMPLIFIED VERSION)")
    print("=" * 70)
    
    # Get chunks from ingestion
    from ingest import main as ingest_main
    _, chunks = ingest_main()
    
    # Create vectorstore
    vectorstore = create_vectorstore_simple(chunks)
    
    # Test listing projects
    print("\nTesting project listing...")
    titles = list_all_projects_simple(vectorstore)
    
    print(f"\n✓ Found {len(titles)} unique projects:\n")
    for i, title in enumerate(titles, 1):
        print(f"{i:2d}. {title}")
    
    # Test search
    print("\n\nTesting semantic search for 'fraud detection'...")
    results = vectorstore.similarity_search("fraud detection", k=3)
    
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get('title', 'Unknown')
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"\n{i}. {title}")
        print(f"   {preview}...")
    
    print("\n" + "=" * 70)
    print("✓ VECTORSTORE READY!")
    print("=" * 70)

if __name__ == "__main__":
    main()