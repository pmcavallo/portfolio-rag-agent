"""Quick diagnostic to check what's in the vectorstore"""
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

print("Checking vectorstore...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="portfolio_projects"
)

# Try to get count
try:
    # Method 1: Try _collection
    count = vectorstore._collection.count()
    print(f"Collection count: {count}")
except Exception as e:
    print(f"Error getting count via _collection: {e}")

# Method 2: Try getting all docs
try:
    all_docs = vectorstore.get()
    print(f"Documents via get(): {len(all_docs.get('ids', []))}")
    if all_docs.get('metadatas'):
        print(f"First metadata: {all_docs['metadatas'][0]}")
except Exception as e:
    print(f"Error via get(): {e}")

# Method 3: Try similarity search
try:
    results = vectorstore.similarity_search("test", k=5)
    print(f"Similarity search returned: {len(results)} docs")
    if results:
        print(f"First result metadata: {results[0].metadata}")
except Exception as e:
    print(f"Error in similarity search: {e}")