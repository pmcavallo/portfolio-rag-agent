"""
Document Ingestion Pipeline
============================

This module handles fetching, parsing, and chunking the 27 portfolio project 
markdown files from GitHub. The critical task is extracting YAML frontmatter 
metadata (especially the project title) and attaching it to every chunk.

WHY THIS MATTERS:
- Flowise had no way to extract true titles, so the LLM invented fake ones
- We store titles as metadata, allowing us to list projects without LLM calls
- This prevents hallucination at the architectural level, not just prompt level
"""

import os
import re
import yaml
import requests
from typing import List, Dict, Any
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document


class PortfolioDocumentLoader:
    """
    Fetches and parses markdown files from Paulo's GitHub portfolio.
    
    Each markdown file has YAML frontmatter:
    ---
    layout: post
    title: "Actual Project Title"
    date: YYYY-MM-DD
    ---
    
    We MUST extract this metadata before chunking, otherwise we lose the 
    connection between chunks and their source project titles.
    """
    
    def __init__(self, github_raw_base_url: str):
        """
        Args:
            github_raw_base_url: Base URL for raw GitHub content
                Example: https://raw.githubusercontent.com/pmcavallo/pmcavallo.github.io/master/_posts
        """
        self.base_url = github_raw_base_url
        self.documents: List[Document] = []
        
    def fetch_file_list(self) -> List[str]:
        """
        Get list of all markdown files in the _posts directory.
        
        For simplicity, we'll provide the list manually since we know there are 27.
        In production, you might use GitHub API to auto-discover files.
        
        Returns:
            List of filenames (e.g., ["2025-09-28-fraud-rt-sandbox.md", ...])
        """
        # TODO: For this MVP, we'll fetch a known list
        # In Phase 2, could use GitHub API: api.github.com/repos/{owner}/{repo}/contents/{path}
        
        # Placeholder - we'll need the actual 27 filenames
        # For now, let's use the API approach
        api_url = "https://api.github.com/repos/pmcavallo/pmcavallo.github.io/contents/_posts"
        
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            files = response.json()
            
            # Filter for markdown files only
            md_files = [f['name'] for f in files if f['name'].endswith('.md')]
            print(f"✓ Found {len(md_files)} markdown files in repository")
            return md_files
            
        except Exception as e:
            print(f"✗ Error fetching file list: {e}")
            return []
    
    def parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """
        Extract YAML frontmatter from markdown content.
        
        FIXED: Convert datetime objects to strings for Chroma compatibility.
        """
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(frontmatter_pattern, content, re.DOTALL)
        
        if match:
            yaml_content = match.group(1)
            body_content = match.group(2)
            
            try:
                metadata = yaml.safe_load(yaml_content)
                
                # CRITICAL FIX: Convert any datetime objects to strings
                # Chroma only accepts str, int, float, bool in metadata
                for key, value in metadata.items():
                    if hasattr(value, 'isoformat'):  # It's a date/datetime
                        metadata[key] = value.isoformat()  # Convert to string
                
                return metadata, body_content
            except yaml.YAMLError as e:
                print(f"✗ YAML parsing error: {e}")
                return {}, content
        else:
            print("⚠ Warning: No frontmatter found in document")
            return {}, content
        
    def fetch_document(self, filename: str) -> Document:
        """
        Fetch a single markdown file and parse it into a LangChain Document.
        
        The Document object contains:
        - page_content: The actual markdown text
        - metadata: Dict with title, date, filename, source
        
        Args:
            filename: Name of the markdown file (e.g., "2025-09-28-fraud-rt-sandbox.md")
            
        Returns:
            LangChain Document with metadata attached
        """
        url = f"{self.base_url}/{filename}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            content = response.text
            
            # Parse frontmatter to extract metadata
            metadata, body = self.parse_frontmatter(content)
            
            # Enrich metadata with additional fields
            metadata['filename'] = filename
            metadata['source'] = url
            metadata['fetch_date'] = datetime.now().isoformat()
            
            # Create LangChain Document
            # We keep the full content (including frontmatter context) for chunking
            doc = Document(
                page_content=body,
                metadata=metadata
            )
            
            print(f"✓ Fetched: {metadata.get('title', 'Untitled')} ({filename})")
            return doc
            
        except Exception as e:
            print(f"✗ Error fetching {filename}: {e}")
            return None
    
    def load_all_documents(self) -> List[Document]:
        """
        Fetch and parse all 27 portfolio markdown files.
        
        Returns:
            List of LangChain Documents, each with metadata attached
        """
        filenames = self.fetch_file_list()
        
        if not filenames:
            raise ValueError("No markdown files found! Check GitHub URL.")
        
        print(f"\n📥 Fetching {len(filenames)} documents from GitHub...\n")
        
        documents = []
        for filename in filenames:
            doc = self.fetch_document(filename)
            if doc:
                documents.append(doc)
        
        print(f"\n✓ Successfully loaded {len(documents)} documents")
        
        # Validation: Do we have 27 documents?
        if len(documents) != 27:
            print(f"⚠ WARNING: Expected 27 documents, got {len(documents)}")
        
        self.documents = documents
        return documents


class DocumentChunker:
    """
    Splits documents into smaller chunks for embedding and retrieval.
    
    CHUNKING STRATEGY:
    - Chunk size: 3000 characters (preserves context for project descriptions)
    - Overlap: 500 characters (ensures no information loss at boundaries)
    - Separator hierarchy: paragraphs > sentences > words (semantic preservation)
    
    WHY THESE PARAMETERS:
    - Portfolio projects are narrative descriptions (not just facts)
    - Users ask about project summaries, not specific line-item details
    - With only 27 documents, we can afford larger chunks
    - Larger chunks = better context = better answers
    
    EXPECTED OUTPUT:
    - 27 documents × ~1-3 chunks each = 38-50 total chunks
    """
    
    def __init__(self, chunk_size: int = 3000, chunk_overlap: int = 500):
        """
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        # RecursiveCharacterTextSplitter tries to split on semantic boundaries
        # Priority: \n\n (paragraphs) > \n (lines) > . (sentences) > space > characters
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        CRITICAL: Each chunk must retain the parent document's metadata
        (especially the title) so we can track which project it came from.
        
        Args:
            documents: List of full documents with metadata
            
        Returns:
            List of chunked documents, each with:
            - page_content: Chunk text
            - metadata: {title, filename, date, chunk_index, ...}
        """
        print(f"\n✂️  Chunking {len(documents)} documents...")
        print(f"   Chunk size: {self.splitter._chunk_size} chars")
        print(f"   Overlap: {self.splitter._chunk_overlap} chars\n")
        
        chunked_docs = []
        
        for doc in documents:
            # Split this document into chunks
            chunks = self.splitter.split_documents([doc])
            
            # Add chunk index to metadata (useful for debugging)
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                chunked_docs.append(chunk)
            
            title = doc.metadata.get('title', 'Untitled')
            print(f"✓ {title}: {len(chunks)} chunks")
        
        print(f"\n✓ Total chunks created: {len(chunked_docs)}")
        
        return chunked_docs


def validate_documents(documents: List[Document]) -> None:
    """
    Validate that all documents have required metadata.
    This catches problems early before we waste money on embeddings.
    
    Checks:
    - All documents have 'title' metadata
    - No duplicate titles (each project should be unique)
    - All titles are non-empty strings
    - All documents have content
    """
    print("\n🔍 Validating documents...\n")
    
    titles = []
    issues = []
    
    for i, doc in enumerate(documents):
        # Check for title
        if 'title' not in doc.metadata:
            issues.append(f"Document {i}: Missing 'title' in metadata")
        else:
            title = doc.metadata['title']
            if not title or not isinstance(title, str):
                issues.append(f"Document {i}: Invalid title: {title}")
            else:
                titles.append(title)
        
        # Check for content
        if not doc.page_content or len(doc.page_content.strip()) == 0:
            issues.append(f"Document {i}: Empty content")
        
        # Check for filename
        if 'filename' not in doc.metadata:
            issues.append(f"Document {i}: Missing 'filename' in metadata")
    
    # Check for duplicates
    unique_titles = set(titles)
    if len(titles) != len(unique_titles):
        duplicates = [t for t in titles if titles.count(t) > 1]
        issues.append(f"Duplicate titles found: {set(duplicates)}")
    
    # Report
    if issues:
        print("✗ VALIDATION FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        raise ValueError("Document validation failed!")
    else:
        print(f"✓ All {len(documents)} documents validated successfully")
        print(f"✓ Found {len(unique_titles)} unique project titles")
        print(f"\n📋 Project Titles:")
        for title in sorted(titles):
            print(f"   • {title}")


def main():
    """
    Main ingestion pipeline.
    
    Steps:
    1. Fetch 27 markdown files from GitHub
    2. Parse YAML frontmatter (extract titles!)
    3. Validate documents
    4. Chunk documents
    5. Validate chunks
    
    Run this script first to verify data quality before creating embeddings.
    """
    print("=" * 70)
    print("PORTFOLIO RAG AGENT - DOCUMENT INGESTION")
    print("=" * 70)
    
    # Configuration
    GITHUB_BASE_URL = "https://raw.githubusercontent.com/pmcavallo/pmcavallo.github.io/master/_posts"
    
    # Step 1: Load documents from GitHub
    loader = PortfolioDocumentLoader(GITHUB_BASE_URL)
    documents = loader.load_all_documents()
    
    # Step 2: Validate full documents
    validate_documents(documents)
    
    # Step 3: Chunk documents
    chunker = DocumentChunker(chunk_size=3000, chunk_overlap=500)
    chunks = chunker.chunk_documents(documents)
    
    # Step 4: Validate chunks
    print("\n🔍 Validating chunks...")
    titles_in_chunks = set(chunk.metadata.get('title') for chunk in chunks)
    print(f"✓ Chunks contain {len(titles_in_chunks)} unique project titles")
    
    if len(titles_in_chunks) != 27:
        print(f"⚠ WARNING: Expected 27 unique titles, found {len(titles_in_chunks)}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Average chunks per document: {len(chunks) / len(documents):.1f}")
    print(f"Unique titles: {len(titles_in_chunks)}")
    print("=" * 70)
    
    return documents, chunks


if __name__ == "__main__":
    docs, chunks = main()