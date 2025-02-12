"""Document processing module for the CHT Documentation Q&A Chatbot."""

import json
from typing import Dict, List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import clean_text, get_scraped_docs_dir
import os


class DocumentProcessor:
    """Handles document processing and chunking for the RAG system."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
    ):
        """Initialize the document processor.

        Args:
            chunk_size: The size of text chunks to create.
            chunk_overlap: The amount of overlap between chunks.
            length_function: Function to measure text length.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_scraped_docs(self, filename: Optional[str] = None) -> List[Dict]:
        """Load scraped documents from JSON file.

        Args:
            filename: Specific JSON file to load. If None, loads the most recent.

        Returns:
            List of document dictionaries.
        """
        docs_dir = get_scraped_docs_dir()

        if filename is None:
            # Get the most recent file
            json_files = [
                f for f in os.listdir(docs_dir) if f.endswith(".json")
            ]
            if not json_files:
                raise FileNotFoundError("No scraped document files found")
            filename = max(json_files)  # Most recent by filename

        file_path = os.path.join(docs_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process documents into chunks suitable for embedding.

        Args:
            documents: List of document dictionaries from the scraper.

        Returns:
            List of processed document chunks with metadata.
        """
        processed_chunks = []

        for doc in documents:
            # Clean the content
            content = clean_text(doc["content"])

            # Skip empty documents
            if not content:
                continue

            # Split into chunks
            chunks = self.text_splitter.split_text(content)

            # Create chunk documents with metadata
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "chunk_id": f"{doc['url']}_{i}",
                    "text": chunk,
                    "metadata": {
                        "url": doc["url"],
                        "title": doc["title"],
                        "section": doc["section"],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                }
                processed_chunks.append(chunk_doc)

        return processed_chunks

    def save_processed_chunks(self, chunks: List[Dict], output_file: str):
        """Save processed chunks to a JSON file.

        Args:
            chunks: List of processed document chunks.
            output_file: Path to save the processed chunks.
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    """Main function to test document processing."""
    processor = DocumentProcessor()

    # Load the most recent scraped docs
    docs = processor.load_scraped_docs()
    print(f"Loaded {len(docs)} documents")

    # Process into chunks
    chunks = processor.process_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # Save processed chunks
    output_file = os.path.join(get_scraped_docs_dir(), "processed_chunks.json")
    processor.save_processed_chunks(chunks, output_file)
    print(f"Saved processed chunks to {output_file}")


if __name__ == "__main__":
    main()
