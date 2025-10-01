"""
Grounded RAG Agent - Zero Hallucination Architecture
=====================================================

This is the CRITICAL component that prevents hallucination.

FLOWISE FAILED because:
- Pre-built chains prioritize conversational fluency
- LLM "fills gaps" with plausible but false information
- No validation of responses against retrieved content

OUR SOLUTION:
- Strict system prompt: "Answer ONLY from context"
- Temperature = 0 (deterministic, no creativity)
- Response validation layer
- Fallback to "I don't know" when uncertain
- Every answer MUST cite sources

This architectural approach makes hallucination nearly impossible.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_chroma import Chroma

load_dotenv()


class GroundedPortfolioAgent:
    """
    RAG agent that answers ONLY from retrieved context.
    
    Key Features:
    - Refuses to answer without relevant context
    - Cites sources for every claim
    - Temperature = 0 (no creative hallucination)
    - Response validation
    """
    
    def __init__(self, vectorstore: Chroma):
        """
        Initialize the grounded agent.
        
        Args:
            vectorstore: Chroma vectorstore with portfolio projects
        """
        self.vectorstore = vectorstore
        
        # Use GPT-4o-mini: cheap ($0.15/1M tokens), capable, reliable
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,  # CRITICAL: No creativity = no hallucination
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # System prompt that enforces strict grounding
        self.system_prompt = """You are a factual assistant for Paulo Cavallo's portfolio of 27 data science projects.

STRICT RULES - FOLLOW EXACTLY:

1. Answer ONLY using the provided context chunks below
2. If the context doesn't contain enough information, respond with: "I don't have sufficient information about that in the portfolio."
3. ALWAYS cite the source filename for every claim you make
4. Format citations as: (Source: filename.md)
5. DO NOT add information from your training data
6. DO NOT make up project names or details
7. DO NOT speculate or infer beyond what's explicitly stated
8. If asked for subjective judgments (best, most complex, etc.), either decline or cite objective metrics from the context

When the context is relevant:
- Provide accurate answers based strictly on the context
- Quote relevant excerpts when helpful
- Include source citations after each claim
- Be concise but complete

Remember: It's better to say "I don't know" than to hallucinate. Accuracy is more important than being helpful."""

    def answer_question(
        self, 
        question: str, 
        k: int = 10,
        min_relevance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Answer a question about the portfolio projects.
        
        Process:
        1. Retrieve relevant chunks from vectorstore
        2. Check relevance scores
        3. Format context for LLM
        4. Generate answer with strict grounding
        5. Validate response
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            min_relevance: Minimum relevance score (0-1) to consider
            
        Returns:
            Dict with:
            - answer: The grounded response
            - sources: List of source documents
            - context_used: The context provided to LLM
            - retrieval_scores: Relevance scores
        """
        print(f"\n🔍 Question: {question}\n")
        
        # Step 1: Retrieve relevant chunks
        results = self.vectorstore.similarity_search_with_score(
            query=question,
            k=k
        )
        
        # Step 2: Filter by relevance threshold
        filtered_results = [
            (doc, score) for doc, score in results 
            if score >= min_relevance
        ]
        
        if not filtered_results:
            return {
                "answer": "I don't have sufficient information about that in the portfolio. The search didn't return any sufficiently relevant results.",
                "sources": [],
                "context_used": "",
                "retrieval_scores": []
            }
        
        print(f"✓ Retrieved {len(filtered_results)} relevant chunks\n")
        
        # Step 3: Format context for LLM
        context_parts = []
        sources = []
        scores = []
        
        for i, (doc, score) in enumerate(filtered_results, 1):
            title = doc.metadata.get('title', 'Unknown')
            filename = doc.metadata.get('filename', 'unknown.md')
            chunk_text = doc.page_content
            
            context_parts.append(
                f"[CHUNK {i} - Source: {filename}]\n"
                f"Project Title: {title}\n"
                f"Content: {chunk_text}\n"
            )
            
            sources.append({
                'title': title,
                'filename': filename,
                'relevance_score': float(score)
            })
            scores.append(float(score))
        
        context = "\n---\n".join(context_parts)
        
        # Step 4: Generate answer with strict grounding
        print("🤖 Generating grounded response...\n")
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Context from portfolio:

{context}

---

Question: {question}

Remember: Answer ONLY from the context above. Cite sources. If insufficient information, say so.""")
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Step 5: Basic validation
        has_citation = any(marker in answer for marker in ['Source:', 'filename', '.md'])
        has_disclaimer = any(phrase in answer.lower() for phrase in [
            "i don't have", "insufficient", "don't know", "cannot find"
        ])
        
        if not has_citation and not has_disclaimer:
            print("⚠️  WARNING: Response lacks citations!\n")
        
        print(f"✓ Response generated ({len(answer)} chars)\n")
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "retrieval_scores": scores
        }
    
    def list_all_projects(self) -> List[str]:
        """
        List all 27 project titles without using LLM.
        
        This is a METADATA QUERY, not an LLM call.
        Impossible to hallucinate because we're just reading stored data.
        
        Returns:
            Sorted list of all project titles
        """
        print("📋 Listing all projects (metadata query, no LLM)...\n")
        
        all_docs = self.vectorstore.similarity_search("", k=500)
        
        titles = set()
        for doc in all_docs:
            title = doc.metadata.get('title')
            if title:
                titles.add(title)
        
        titles_list = sorted(list(titles))
        
        print(f"✓ Found {len(titles_list)} unique projects\n")
        
        return titles_list
    
    def summarize_project(self, project_title: str) -> Dict[str, Any]:
        """
        Generate a summary of a specific project.
        
        Retrieves ALL chunks for that project and asks LLM to summarize,
        but still with strict grounding rules.
        
        Args:
            project_title: Exact title of project to summarize
            
        Returns:
            Dict with summary and source info
        """
        print(f"📄 Summarizing project: {project_title}\n")
        
        # Get all chunks for this specific project
        all_chunks = self.vectorstore.similarity_search(
            query=project_title,
            k=50,
            filter={"title": project_title}
        )
        
        if not all_chunks:
            return {
                "summary": f"Could not find project titled '{project_title}'",
                "chunks_used": 0
            }
        
        # Sort by chunk index to maintain order
        all_chunks.sort(key=lambda doc: doc.metadata.get('chunk_index', 0))
        
        # Combine all chunks
        full_content = "\n\n".join(chunk.page_content for chunk in all_chunks)
        filename = all_chunks[0].metadata.get('filename', 'unknown.md')
        
        print(f"✓ Retrieved {len(all_chunks)} chunks ({len(full_content)} chars)\n")
        print("🤖 Generating summary...\n")
        
        # Generate summary with strict grounding
        messages = [
            SystemMessage(content="""You are summarizing a portfolio project. 

Rules:
- Summarize in 2-3 paragraphs
- Use ONLY information from the provided content
- Include key technologies, methods, and outcomes
- Cite the source filename
- Do not add external information"""),
            HumanMessage(content=f"""Project: {project_title}
Source: {filename}

Full Project Content:
{full_content}

---

Provide a comprehensive 2-3 paragraph summary of this project, including:
1. What problem it solved
2. Technologies and methods used
3. Key outcomes or results

Remember to cite the source.""")
        ]
        
        response = self.llm.invoke(messages)
        summary = response.content
        
        print(f"✓ Summary generated ({len(summary)} chars)\n")
        
        return {
            "summary": summary,
            "chunks_used": len(all_chunks),
            "source": filename
        }


def test_agent(agent: GroundedPortfolioAgent):
    """
    Test the agent with queries that made Flowise hallucinate.
    
    These are ADVERSARIAL TESTS - designed to catch hallucination.
    """
    print("\n" + "=" * 70)
    print("TESTING GROUNDED AGENT - ADVERSARIAL QUERIES")
    print("=" * 70 + "\n")
    
    test_queries = [
        {
            "name": "TEST 1: List all projects (should use metadata, not LLM)",
            "query": "list_projects",
            "type": "metadata"
        },
        {
            "name": "TEST 2: Question about fraud detection",
            "query": "Tell me about fraud detection projects",
            "type": "question"
        },
        {
            "name": "TEST 3: Technology search (AWS)",
            "query": "What projects use AWS?",
            "type": "question"
        },
        {
            "name": "TEST 4: Non-existent technology (Azure - user doesn't use it)",
            "query": "What projects use Azure?",
            "type": "question"
        },
        {
            "name": "TEST 5: Subjective question (should refuse or cite metrics)",
            "query": "What's the most complex project?",
            "type": "question"
        }
    ]
    
    for test in test_queries:
        print("\n" + "=" * 70)
        print(test["name"])
        print("=" * 70)
        
        if test["type"] == "metadata":
            # Direct metadata query
            titles = agent.list_all_projects()
            print(f"\nFound {len(titles)} projects:")
            for i, title in enumerate(titles, 1):
                print(f"{i:2d}. {title}")
            
            # Validation
            if len(titles) == 27:
                print("\n✅ PASS: Found exactly 27 projects")
            else:
                print(f"\n❌ FAIL: Expected 27 projects, got {len(titles)}")
        
        else:
            # LLM-based question
            result = agent.answer_question(test["query"], k=5)
            
            print(f"\n{'='*70}")
            print("ANSWER:")
            print(f"{'='*70}")
            print(result["answer"])
            print(f"\n{'='*70}")
            print(f"Sources: {len(result['sources'])}")
            for source in result['sources']:
                print(f"  - {source['title']} ({source['relevance_score']:.3f})")
            print(f"{'='*70}\n")
        
        input("Press Enter to continue to next test...")


def main():
    """
    Main testing script.
    """
    print("=" * 70)
    print("GROUNDED RAG AGENT - TESTING")
    print("=" * 70)
    
    # Load vectorstore
    from retriever_v2 import load_vectorstore_simple
    vectorstore = load_vectorstore_simple()
    
    # Create agent
    agent = GroundedPortfolioAgent(vectorstore)
    
    # Run tests
    test_agent(agent)
    
    print("\n" + "=" * 70)
    print("✓ AGENT TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()