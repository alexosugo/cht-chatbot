"""RAG chain implementation for the CHT Documentation Q&A Chatbot."""

from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from typing import Dict, Any
from utils import load_config


class RAGChain:
    """Implements the Retrieval Augmented Generation chain using LangChain."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        top_k: int = 5,
    ):
        """Initialize the RAG chain.

        Args:
            model_name: Name of the VertexAI model to use.
            temperature: Sampling temperature for generation.
            top_k: Number of similar documents to retrieve.
        """
        # Load configuration
        config = load_config()

        # Initialize Pinecone
        pinecone = Pinecone(
            api_key=config["PINECONE_API_KEY"],
        )

        # Initialize LangChain components
        self.embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

        self.llm = VertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
        )

        # Get Pinecone index
        self.index_name = "cht-docs"
        if self.index_name not in pinecone.list_indexes():
            raise ValueError(f"Index '{self.index_name}' not found in Pinecone")

        # Initialize vector store with Langchain's PineconeVectorStore
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            text_key="text",  # Key for the document text in metadata
            namespace="",  # Optional namespace for the vectors
        )

        # Setup memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Setup prompt templates
        self.qa_template = PromptTemplate(
            template="""You are a helpful assistant for the Community Health
            Toolkit (CHT). Use the following pieces of context to answer the
            question at the end. If you don't know the answer, just say that
            you don't know, don't try to make up an answer. Always cite your
            sources by mentioning the title and URL of the documentation pages
            you used.

            Context: {context}

            Chat History: {chat_history}

            Question: {question}

            Answer: Let me help you with that.""",
            input_variables=["context", "chat_history", "question"],
        )

        # Initialize the RAG chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": top_k}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.qa_template},
            return_source_documents=True,
        )

    async def answer_question(
        self, question: str, use_history: bool = True
    ) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline.

        Args:
            question: User's question.
            use_history: Whether to include conversation history.

        Returns:
            Dictionary containing the answer and metadata.
        """
        # Get response from chain with proper chat history handling
        chat_history = (
            [] if not use_history else self.memory.chat_memory.messages
        )
        response = await self.chain.ainvoke(
            {"question": question, "chat_history": chat_history}
        )

        # Extract source documents
        sources = []
        for doc in response["source_documents"]:
            sources.append(
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "url": doc.metadata.get("url", ""),
                    "section": doc.metadata.get("section", ""),
                    "score": doc.metadata.get("score", 0.0),
                }
            )

        # Prepare response
        result = {
            "question": question,
            "answer": response["answer"],
            "sources": sources,
        }

        return result

    def clear_history(self):
        """Clear the conversation history."""
        self.memory.clear()


async def main():
    """Main function to test the RAG chain."""

    # Initialize RAG chain
    rag_chain = RAGChain()

    # Test questions
    questions = [
        "What is the Community Health Toolkit?",
        "How do I install it?",
        "What are the system requirements?",
    ]

    # Answer questions
    for question in questions:
        print(f"\nQuestion: {question}")
        response = await rag_chain.answer_question(question)
        print(f"\nAnswer: {response['answer']}")
        print("\nSources:")
        for source in response["sources"]:
            print(f"- {source['title']} ({source['url']})")
            print(f"  Relevance score: {source['score']:.4f}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
