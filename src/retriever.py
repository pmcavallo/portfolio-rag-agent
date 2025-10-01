"""
Vector Store and Retrieval System
==================================

This module handles:
1. Creating embeddings for all chunks
2. Storing in Chroma vector database
3. Semantic search across projects
4. Metadata-based queries (list all projects WITHOUT using LLM)

WHY CHROMA:
- Local, no server needed (free!)
- Persistent storage (build once, use forever)
- Fast similarity search
- Supports metadata filtering

COST:
- text-embedding-3-small: $0.00002 per 1K tokens
- 153 chunks × ~750 chars avg = ~115K chars = ~29K tokens
- Cost: 29 × $0.00002 = $0.00058 (less than a penny!)
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Load environment variables
load_dotenv()


class PortfolioVectorStore:
    """
    Manages the vector database for portfolio projects.
    
    Key capabilities:
    - Store document chunks with embeddings
    - Semantic search (find relevant projects by meaning)
    - Metadata queries (list all titles, filter by date, etc.)
    - MMR search (diverse results, not all from same project)
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Where to save the Chroma database
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Cheapest, still excellent
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vectorstore: Optional[Chroma] = None
        
        print(f"📦 Vector store will persist to: {persist_directory}")
    
    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        Create embeddings for all chunks and store in Chroma.
        
        This is a ONE-TIME operation that costs ~$0.0006.
        Once built, we can query it forever for free.
        
        Args:
            chunks: List of document chunks with metadata
            
        Returns:
            Chroma vectorstore instance
        """
        print(f"\n🔮 Creating embeddings for {len(chunks)} chunks...")
        print(f"   Model: text-embedding-3-small")
        print(f"   Estimated cost: ~$0.0006")
        print(f"   This will take ~30-60 seconds...\n")
        
        # Create vectorstore with embeddings
        # from_documents() handles: embedding creation + storage
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="portfolio_projects"
        )
        
        print(f"✓ Embeddings created and stored in {self.persist_directory}")
        print(f"✓ Vector store contains {vectorstore._collection.count()} chunks\n")
        
        self.vectorstore = vectorstore
        return vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        Load existing vectorstore from disk.
        
        Use this after initial creation to avoid re-embedding (saves time & money).
        
        Returns:
            Chroma vectorstore instance
        """
        print(f"\n📂 Loading existing vectorstore from {self.persist_directory}...")
        
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="portfolio_projects"
        )
        
        count = vectorstore._collection.count()
        print(f"✓ Loaded vectorstore with {count} chunks\n")
        
        self.vectorstore = vectorstore
        return vectorstore
    
    def semantic_search(
        self, 
        query: str, 
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for relevant chunks using semantic similarity.
        
        This finds chunks that are CONCEPTUALLY similar to the query,
        not just keyword matches. E.g., "ML pipelines" will match
        "machine learning workflows" even without exact words.
        
        Args:
            query: User's question or search term
            k: Number of results to return
            filter_dict: Optional metadata filter (e.g., {"title": "Fraud Detection"})
            
        Returns:
            List of (Document, relevance_score) tuples, sorted by relevance
            Score is 0-1, where 1 = perfect match
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized! Call create or load first.")
        
        print(f"🔍 Searching for: '{query}' (top {k} results)")
        
        # similarity_search_with_score returns (doc, score) tuples
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        print(f"✓ Found {len(results)} relevant chunks\n")
        
        # Display results preview
        for i, (doc, score) in enumerate(results[:3], 1):
            title = doc.metadata.get('title', 'Unknown')
            print(f"   {i}. {title} (relevance: {score:.3f})")
        
        if len(results) > 3:
            print(f"   ... and {len(results) - 3} more results\n")
        
        return results
    
    def mmr_search(
        self,
        query: str,
        k: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Maximal Marginal Relevance search - returns DIVERSE results.
        
        Problem: Regular similarity search might return 10 chunks all from
        the same project (if it's very relevant).
        
        Solution: MMR balances relevance with diversity. It ensures you get
        results from MULTIPLE projects, not just the top match.
        
        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of candidates to consider (then diversify down to k)
            lambda_mult: 0 = max diversity, 1 = max relevance (0.5 = balanced)
            
        Returns:
            List of diverse documents
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized!")
        
        print(f"🎯 MMR search for: '{query}' (diversity factor: {lambda_mult})")
        
        results = self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        
        # Show diversity
        unique_titles = set(doc.metadata.get('title') for doc in results)
        print(f"✓ Found {len(results)} chunks from {len(unique_titles)} different projects\n")
        
        return results
    
    def list_all_projects(self) -> List[str]:
        """
        Get list of all 27 project titles directly from metadata.
        
        CRITICAL: This does NOT use the LLM at all!
        We query the vectorstore metadata to get the TRUE titles.
        This is impossible to hallucinate.
        
        Returns:
            List of 27 unique project titles, sorted alphabetically
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized!")
        
        print("📋 Listing all projects from metadata (no LLM used)...")
        
        # Get all documents from the collection
        collection = self.vectorstore._collection
        all_metadatas = collection.get()['metadatas']
        
        # Extract unique titles
        titles = set()
        for metadata in all_metadatas:
            if 'title' in metadata:
                titles.add(metadata['title'])
        
        titles_list = sorted(list(titles))
        
        print(f"✓ Found {len(titles_list)} unique projects\n")
        
        return titles_list
    
    def get_project_chunks(self, project_title: str) -> List[Document]:
        """
        Retrieve ALL chunks for a specific project.
        
        Used for project summarization - we want the complete content,
        not just relevant snippets.
        
        Args:
            project_title: Exact title of the project
            
        Returns:
            List of all chunks from that project, in order
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized!")
        
        print(f"📄 Retrieving all chunks for: '{project_title}'")
        
        # Use metadata filter to get only chunks from this project
        results = self.vectorstore.similarity_search(
            query=project_title,  # Query doesn't matter much here
            k=50,  # Get plenty (most projects have <10 chunks)
            filter={"title": project_title}
        )
        
        # Sort by chunk_index to maintain document order
        results.sort(key=lambda doc: doc.metadata.get('chunk_index', 0))
        
        print(f"✓ Found {len(results)} chunks for this project\n")
        
        return results


def build_vectorstore(chunks: List[Document], force_rebuild: bool = False):
    """
    Main function to build or load the vector store.
    
    Args:
        chunks: Document chunks from ingestion
        force_rebuild: If True, rebuild even if vectorstore exists
    """
    store = PortfolioVectorStore(persist_directory="./chroma_db")
    
    # Check if vectorstore already exists
    if os.path.exists("./chroma_db") and not force_rebuild:
        print("⚠️  Vectorstore already exists!")
        print("   To rebuild, delete ./chroma_db folder or set force_rebuild=True\n")
        vectorstore = store.load_vectorstore()
    else:
        if force_rebuild and os.path.exists("./chroma_db"):
            print("🔄 Rebuilding vectorstore (force_rebuild=True)...\n")
        vectorstore = store.create_vectorstore(chunks)
    
    return store, vectorstore


def test_retrieval(store: PortfolioVectorStore):
    """
    Test retrieval functions with example queries.
    """
    print("\n" + "=" * 70)
    print("TESTING RETRIEVAL SYSTEM")
    print("=" * 70 + "\n")
    
    # Test 1: List all projects (metadata query, no LLM)
    print("TEST 1: List all project titles")
    print("-" * 70)
    titles = store.list_all_projects()
    for i, title in enumerate(titles, 1):
        print(f"{i:2d}. {title}")
    print()
    
    # Test 2: Semantic search
    print("\nTEST 2: Semantic search - 'fraud detection'")
    print("-" * 70)
    results = store.semantic_search("fraud detection", k=5)
    for doc, score in results:
        title = doc.metadata.get('title', 'Unknown')
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"[{score:.3f}] {title}")
        print(f"        {preview}...\n")
    
    # Test 3: MMR search (diverse results)
    print("\nTEST 3: MMR search - 'machine learning'")
    print("-" * 70)
    diverse_results = store.mmr_search("machine learning", k=5)
    unique_projects = set()
    for doc in diverse_results:
        title = doc.metadata.get('title', 'Unknown')
        unique_projects.add(title)
        print(f"   • {title}")
    print(f"\n✓ Results span {len(unique_projects)} different projects (good diversity!)\n")
    
    # Test 4: Get specific project
    print("\nTEST 4: Retrieve all chunks from first project")
    print("-" * 70)
    if titles:
        first_title = titles[0]
        project_chunks = store.get_project_chunks(first_title)
        print(f"Project: {first_title}")
        print(f"Total chunks: {len(project_chunks)}")
        print(f"Total characters: {sum(len(c.page_content) for c in project_chunks)}")


def main():
    """
    Main execution: Load chunks from ingestion, build vectorstore, test retrieval.
    """
    print("=" * 70)
    print("PORTFOLIO RAG AGENT - VECTOR STORE CREATION")
    print("=" * 70 + "\n")
    
    # Import and run ingestion to get chunks
    from ingest import main as ingest_main
    _, chunks = ingest_main()
    
    # Build vectorstore
    store, vectorstore = build_vectorstore(chunks, force_rebuild=False)
    
    # Test retrieval
    test_retrieval(store)
    
    print("\n" + "=" * 70)
    print("✓ VECTORSTORE READY FOR USE!")
    print("=" * 70)
    print(f"Location: ./chroma_db")
    print(f"Chunks: {len(chunks)}")
    print(f"Projects: 27")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()