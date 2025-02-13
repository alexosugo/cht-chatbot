#!/usr/bin/env python3
"""Main script for running the CHT Documentation Q&A Chatbot pipeline."""

import asyncio
import argparse
import json
import os
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from scraper.scraper import CHTDocCrawler
from core.document_processor import DocumentProcessor
from core.embeddings import EmbeddingsManager

from utils import load_config, get_scraped_docs_dir

from honeyhive import HoneyHiveTracer, atrace

HoneyHiveTracer.init(
    api_key=load_config()["HONEY_HIVE_API_KEY"],
    project=load_config()["AGENT_ID"],
)

console = Console()


@atrace
async def crawl_docs(show_progress: bool = False) -> Optional[str]:
    """Scrape CHT documentation.

    Returns:
        Path to the scraped documents file, or None if scraping failed.
    """
    try:
        with console.status("[bold yellow]Scraping CHT documentation..."):
            crawler = CHTDocCrawler()
            results = crawler.crawl()

            # Get the latest file
            docs_dir = get_scraped_docs_dir()
            json_files = [f for f in os.listdir(docs_dir) if f.endswith(".json")]
            latest_file = max(json_files)  # Most recent by filename

            console.print(f"[green]✓[/] Scraped {len(results)} documents")
            return os.path.join(docs_dir, latest_file)

    except Exception as e:
        console.print(f"[bold red]Error scraping documentation:[/] {str(e)}")
        return None


@atrace
async def process_documents(docs_file: str) -> Optional[str]:
    """Process scraped documents into chunks.

    Args:
        docs_file: Path to the scraped documents file.

    Returns:
        Path to the processed chunks file, or None if processing failed.
    """
    try:
        with console.status("[bold yellow]Processing documents..."):
            processor = DocumentProcessor()

            # Load and process documents
            docs = processor.load_scraped_docs(os.path.basename(docs_file))
            chunks = processor.process_documents(docs)

            # Save processed chunks
            output_file = os.path.join(get_scraped_docs_dir(), "processed_chunks.json")
            processor.save_processed_chunks(chunks, output_file)

            console.print(f"[green]✓[/] Created {len(chunks)} chunks")
            return output_file

    except Exception as e:
        console.print(f"[bold red]Error processing documents:[/] {str(e)}")
        return None


@atrace
async def generate_embeddings(chunks_file: str) -> bool:
    """Generate embeddings and store in Pinecone.

    Args:
        chunks_file: Path to the processed chunks file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with console.status("[bold yellow]Generating embeddings..."):
            # Load chunks
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            # Initialize embeddings manager
            embeddings_manager = EmbeddingsManager()

            # Generate embeddings and upload to Pinecone
            embedded_chunks = await embeddings_manager.batch_generate_embeddings(chunks)
            embeddings_manager.upsert_to_pinecone(embedded_chunks)

            msg = (
                f"[green]✓[/] Generated and stored embeddings for {len(chunks)} chunks"
            )
            console.print(msg)
            return True

    except Exception as e:
        console.print(f"[bold red]Error generating embeddings:[/] {str(e)}")
        return False


def run_cli():
    """Run the CLI interface."""
    from cli.interface import main as cli_main

    cli_main()


def run_web():
    """Run the web interface."""
    import subprocess

    subprocess.run(["streamlit", "run", "src/web/app.py"])


async def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description="CHT Documentation Q&A Chatbot")
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="cli",
        help="Interface mode (cli or web)",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape and process new documentation",
    )
    args = parser.parse_args()

    try:
        # Verify configuration
        load_config()

        if args.scrape:
            # Run the complete pipeline
            docs_file = await crawl_docs(show_progress=True)
            if not docs_file:
                return

            chunks_file = await process_documents(docs_file)
            if not chunks_file:
                return

            success = await generate_embeddings(chunks_file)
            if not success:
                return

        # Run the selected interface
        if args.mode == "web":
            run_web()
        else:
            run_cli()

    except Exception as e:
        console.print(f"[bold red]Fatal error:[/] {str(e)}")


if __name__ == "__main__":
    # Print welcome message
    console.print(
        Panel.fit(
            "[bold blue]CHT Documentation Q&A Chatbot[/]\n\n"
            "A RAG-based chatbot for answering questions about the "
            "Community Health Toolkit documentation.",
            title="Welcome",
        )
    )

    # Run main function
    asyncio.run(main())
