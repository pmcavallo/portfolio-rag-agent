"""
Simple Gradio UI for Portfolio RAG Agent
"""
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check API key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file!")

from retriever_v2 import load_vectorstore_simple
from agent import GroundedPortfolioAgent

# Initialize
print("Loading vectorstore...")
try:
    vectorstore = load_vectorstore_simple()
    agent = GroundedPortfolioAgent(vectorstore)
    print("Agent ready!")
except Exception as e:
    print(f"Error initializing agent: {e}")
    raise

def answer_question(question: str) -> str:
    """Handle user question with error handling."""
    if not question or not question.strip():
        return "Please enter a question."
    
    try:
        result = agent.answer_question(question, k=5)
        
        # Format response with sources
        answer = result["answer"]
        sources = result["sources"]
        
        response = f"{answer}\n\n---\n\n**Sources ({len(sources)}):**\n"
        for i, source in enumerate(sources, 1):
            response += f"{i}. {source['title']} (relevance: {source['relevance_score']:.3f})\n"
        
        return response
    except Exception as e:
        return f"Error processing question: {str(e)}\n\nPlease try rephrasing your question or check the console for details."

def list_projects() -> str:
    """List all projects with error handling."""
    try:
        titles = agent.list_all_projects()
        return "\n".join(f"{i}. {title}" for i, title in enumerate(titles, 1))
    except Exception as e:
        return f"Error listing projects: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Portfolio RAG Agent") as demo:
    gr.Markdown("# Paulo Cavallo's Portfolio RAG Agent")
    gr.Markdown("Ask questions about 27 data science projects. **Zero hallucination guaranteed!**")
    
    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What projects use AWS?",
            lines=2
        )
        question_output = gr.Textbox(
            label="Answer",
            lines=15
        )
        question_btn = gr.Button("Ask", variant="primary")
        
        gr.Markdown("### Example Questions:")
        gr.Markdown("- Tell me about fraud detection projects\n- What projects use machine learning?\n- List projects using AWS\n- What is the most recent project?")
        
        question_btn.click(
            fn=answer_question,
            inputs=question_input,
            outputs=question_output
        )
    
    with gr.Tab("Browse All Projects"):
        gr.Markdown("### All 27 Portfolio Projects")
        projects_output = gr.Textbox(
            label="Project Titles",
            lines=30,
            value=list_projects()
        )
        refresh_btn = gr.Button("Refresh List")
        refresh_btn.click(fn=list_projects, outputs=projects_output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)