"""Streamlit web interface for the CHT Documentation Q&A Chatbot."""

import streamlit as st
import asyncio
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.rag_chain import RAGChain
from src.utils import load_config

# Configure page
st.set_page_config(
    page_title="CHT Documentation Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .source-table {
        font-size: 0.8em;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .assistant-message {
        background-color: #f0f0f0;
    }
    .source-link {
        color: #0066cc;
        text-decoration: none;
    }
    .source-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_chain() -> RAGChain:
    """Get or create RAG chain instance.
    
    Returns:
        RAGChain instance.
    """
    return RAGChain()

def display_sources(sources: list):
    """Display source information in a table.
    
    Args:
        sources: List of source dictionaries.
    """
    if not sources:
        return
        
    st.markdown("### Sources")
    
    # Create a DataFrame for better display
    source_data = []
    for source in sources:
        source_data.append({
            "Title": source['title'],
            "URL": f"[Link]({source['url']})",
            "Relevance Score": f"{source['score']:.2f}"
        })
    
    # Display as table
    st.markdown(
        """
        <div class="source-table">
        """,
        unsafe_allow_html=True
    )
    st.table(source_data)
    st.markdown("</div>", unsafe_allow_html=True)

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling.
    
    Args:
        role: 'user' or 'assistant'
        content: Message content
    """
    class_name = f"{role}-message chat-message"
    st.markdown(
        f"""
        <div class="{class_name}">
            <strong>{role.title()}:</strong><br>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )

async def process_question(
    rag_chain: RAGChain,
    question: str,
    history: list
) -> Dict[str, Any]:
    """Process a question and get response.
    
    Args:
        rag_chain: RAGChain instance
        question: User's question
        history: Conversation history
    
    Returns:
        Response dictionary
    """
    return await rag_chain.answer_question(question)

def main():
    """Main Streamlit application."""
    # Title
    st.title("CHT Documentation Assistant")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            """
            This assistant helps you find information in the Community Health Toolkit
            documentation. Ask questions about CHT, and I'll provide answers with
            relevant source links.
            """
        )
        
        st.markdown("### Tips")
        st.markdown(
            """
            - Be specific in your questions
            - Check the sources for more context
            - You can ask follow-up questions
            - Click source links to read more
            """
        )
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.sources = []
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Display latest sources
    if st.session_state.sources:
        display_sources(st.session_state.sources)
    
    # Question input
    question = st.text_input("Ask a question about CHT:", key="question_input")
    
    if question:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        # Show user message
        display_chat_message("user", question)
        
        # Get response
        try:
            rag_chain = get_rag_chain()
            
            with st.spinner("Thinking..."):
                # Run async code
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    process_question(
                        rag_chain,
                        question,
                        st.session_state.messages
                    )
                )
                loop.close()
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"]
            })
            
            # Update sources
            st.session_state.sources = response["sources"]
            
            # Show assistant message
            display_chat_message("assistant", response["answer"])
            
            # Show sources
            display_sources(response["sources"])
            
            # Clear input
            st.session_state.question_input = ""
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    try:
        # Verify API keys
        load_config()
        main()
    except Exception as e:
        st.error(f"Configuration Error: {str(e)}")
