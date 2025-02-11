"""Embeddings and vector store management for the CHT Documentation Q&A Chatbot."""

import google.generativeai as genai
import pinecone
from typing import Dict, List, Any
from utils import load_config

class EmbeddingsManager:
    """Manages document embeddings and vector store operations."""
    
    def __init__(
        self,
        index_name: str = "cht-docs",
        dimension: int = 768,  # Gemini embedding dimension
        metric: str = "cosine"
    ):
        """Initialize the embeddings manager.
        
        Args:
            index_name: Name of the Pinecone index.
            dimension: Dimension of the embedding vectors.
            metric: Distance metric for vector similarity.
        """
        # Load configuration
        config = load_config()
        
        # Initialize Gemini
        genai.configure(api_key=config['GOOGLE_API_KEY'])
        self.embedding_model = genai.GenerativeModel('embedding-001')
        
        # Initialize Pinecone
        pinecone.init(
            api_key=config['PINECONE_API_KEY'],
            environment=config['PINECONE_ENVIRONMENT']
        )
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric
            )
        
        self.index = pinecone.Index(index_name)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text using Gemini.
        
        Args:
            text: Text to generate embedding for.
        
        Returns:
            Embedding vector as a list of floats.
        """
        result = await self.embedding_model.embed_content(text=text)
        return result.embedding
    
    async def batch_generate_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100
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
            batch = chunks[i:i + batch_size]
            
            # Generate embeddings for the batch
            embeddings = await asyncio.gather(*[
                self.generate_embedding(chunk['text'])
                for chunk in batch
            ])
            
            # Add embeddings to chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk['embedding'] = embedding
                embedded_chunks.append(chunk)
            
            print(f"Processed {len(embedded_chunks)}/{len(chunks)} chunks")
        
        return embedded_chunks
    
    def upsert_to_pinecone(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Upload document chunks to Pinecone.
        
        Args:
            chunks: List of document chunks with embeddings and metadata.
            batch_size: Number of vectors to upsert in each batch.
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare vectors for the batch
            vectors = [
                (
                    chunk['chunk_id'],
                    chunk['embedding'],
                    chunk['metadata']
                )
                for chunk in batch
            ]
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
            
            print(f"Uploaded {min(i + batch_size, len(chunks))}/{len(chunks)} vectors")
    
    async def query_similar(
        self,
        query: str,
        top_k: int = 5,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Query similar documents using the query text.
        
        Args:
            query: Query text to find similar documents for.
            top_k: Number of similar documents to return.
            include_metadata: Whether to include document metadata.
        
        Returns:
            List of similar documents with scores and metadata.
        """
        # Generate query embedding
        query_embedding = await self.generate_embedding(query)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata
        )
        
        return results.matches

async def main():
    """Main function to test embeddings and vector store."""
    import asyncio
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
        "How do I install CHT?",
        top_k=3
    )
    print("\nTest Query Results:")
    for match in results:
        print(f"Score: {match.score:.4f}")
        print(f"Title: {match.metadata.get('title')}")
        print(f"URL: {match.metadata.get('url')}\n")

if __name__ == "__main__":
    asyncio.run(main())
