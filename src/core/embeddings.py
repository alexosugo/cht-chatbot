"""Embeddings and vector store management for the CHT Documentation Q&A Chatbot.

Handles embedding generation and vector store operations using Pinecone."""

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from typing import Dict, List, Any, Optional
from utils import load_config


class EmbeddingsManager:
    """Manages document embeddings and vector store operations.

    Uses Langchain's PineconeVectorStore for vector storage and retrieval.
    """

    def __init__(
        self,
        index_name: str = "cht-docs",
        dimension: int = 768,  # VertexAI embedding dimension
        metric: str = "cosine",
        namespace: Optional[str] = None,
    ):
        """Initialize the embeddings manager.

        Args:
            index_name: Name of the Pinecone index.
            dimension: Dimension of the embedding vectors.
            metric: Distance metric for vector similarity.
            namespace: Optional namespace for vector store operations.
        """
        # Load configuration
        config = load_config()

        # Initialize VertexAI embeddings
        self.embedding_model = VertexAIEmbeddings(
            model_name="text-embedding-005"
        )

        # Initialize Pinecone
        pinecone = Pinecone(api_key=config["PINECONE_API_KEY"])

        index = None

        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            index = pinecone.Index(index_name)

        # Initialize PineconeVectorStore
        self.vector_store = PineconeVectorStore(
            index=index, embedding=self.embedding_model, namespace=namespace
        )

    async def batch_generate_embeddings(
        self, chunks: List[Dict[str, Any]], batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for multiple text chunks.

        Args:
            chunks: List of document chunks with text and metadata.
            batch_size: Number of chunks to process in each batch.

        Returns:
            List of chunks with added embeddings.
        """
        embedded_chunks = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Generate embeddings for the batch
            texts = [chunk["text"] for chunk in batch]
            embeddings = await self.embedding_model.aembed_documents(texts)

            # Add embeddings to chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding
                embedded_chunks.append(chunk)

            print(f"Processed {len(embedded_chunks)}/{len(chunks)} chunks")

        return embedded_chunks

    def upsert_to_pinecone(
        self, chunks: List[Dict[str, Any]], batch_size: int = 100
    ):
        """Upload document chunks to Pinecone using Langchain's PineconeVectorStore.

        Args:
            chunks: List of document chunks with embeddings and metadata.
            batch_size: Number of vectors to upsert in each batch.
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Prepare texts and metadata for the batch
            texts = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            ids = [chunk["chunk_id"] for chunk in batch]

            # Add texts to vector store
            self.vector_store.add_texts(
                texts=texts, metadatas=metadatas, ids=ids
            )

            processed = min(i + batch_size, len(chunks))
            print(f"Uploaded {processed}/{len(chunks)} vectors")

    async def query_similar(
        self, query: str, top_k: int = 5, filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Query similar documents using the query text.

        Args:
            query: Query text to find similar documents for.
            top_k: Number of similar documents to return.
            filter: Optional metadata filter to apply to the search.

        Returns:
            List of similar documents with scores and metadata.
        """
        # Use PineconeVectorStore's similarity search
        results = self.vector_store.similarity_search(
            query=query, k=top_k, filter=filter
        )

        # Format results to maintain compatibility
        formatted_results = []
        for doc in results:
            formatted_results.append(
                {
                    "metadata": doc.metadata,
                    "score": doc.metadata.get(
                        "score", 1.0
                    ),  # Score if available
                    "text": doc.page_content,
                }
            )

        return formatted_results


async def main():
    """Main function to test embeddings and vector store."""
    from document_processor import DocumentProcessor

    # Initialize managers
    doc_processor = DocumentProcessor()
    embeddings_manager = EmbeddingsManager()

    # Load and process documents
    docs = doc_processor.load_scraped_docs()
    chunks = doc_processor.process_documents(docs)
    print(f"Processing {len(chunks)} chunks")

    # Generate embeddings
    embedded_chunks = await embeddings_manager.batch_generate_embeddings(chunks)
    print("Generated embeddings")

    # Upload to Pinecone
    embeddings_manager.upsert_to_pinecone(embedded_chunks)
    print("Uploaded to Pinecone")

    # Test query
    results = await embeddings_manager.query_similar(
        "How do I install CHT?", top_k=3
    )
    print("\nTest Query Results:")
    for match in results:
        print(f"Score: {match.get('score', 1.0):.4f}")
        print(f"Title: {match['metadata'].get('title')}")
        print(f"URL: {match['metadata'].get('url')}\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
