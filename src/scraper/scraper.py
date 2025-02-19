#!/usr/bin/env python3
"""
CHT documentation scraper using Firecrawl.
"""
from typing import Dict, List, Any
import asyncio
import json
import os
from datetime import datetime, timezone
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from dotenv import load_dotenv

load_dotenv()


class CHTDocCrawler:
    """Scraper for Community Health Toolkit documentation."""

    def __init__(self, base_url: str = "https://docs.communityhealthtoolkit.org"):
        """Initialize the scraper.

        Args:
            base_url: The base URL of the CHT documentation.
        """
        self.base_url = base_url

        # Create data directory if it doesn't exist
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "scraped_docs",
        )
        os.makedirs(self.data_dir, exist_ok=True)

    def crawl(self) -> List[Dict]:
        """Crawl the CHT documentation.

        Returns:
            List of dictionaries containing scraped content and metadata.
        """
        loader = FireCrawlLoader(
            api_key=os.getenv("FIRECRAWL_API_KEY"),
            url=self.base_url,
            mode="crawl",
            params={
                "limit": 500,
                "scrapeOptions": {
                    "onlyMainContent": True,
                },
            },
        )

        # Start the crawl
        data = loader.load()

        # Extract the scraped data
        scraped_data = self._process_crawl_result(data)

        # Save the scraped data
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.data_dir, f"cht_docs_{timestamp}.json")

        with open(output_file, "w") as f:
            json.dump(scraped_data, f, indent=2)

        return scraped_data

    def _process_crawl_result(
        self, crawl_result: List[Any], show_progress: bool = True
    ) -> List[Dict]:
        """Process the crawl result to extract useful content.

        Args:
            crawl_result: Raw crawl result containing Document objects
            show_progress: Whether to show progress updates

        Returns:
            List of processed documents with extracted content
        """
        processed_docs = []
        total_pages = len(crawl_result)

        if show_progress:
            print(f"\nProcessing {total_pages} pages...")

        for i, doc in enumerate(crawl_result, 1):
            # Extract content and metadata from the Document object
            metadata = doc.metadata
            content = doc.page_content

            # Create a document with metadata and content
            processed_doc = {
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            processed_docs.append(processed_doc)

            if show_progress:
                if i == 1:
                    print("\nStarting content extraction...")
                if i % 5 == 0:  # Show progress more frequently
                    print(
                        f"\rProcessing content: {i}/{total_pages} pages ({(i/total_pages)*100:.1f}%) - Current: {processed_doc['title']}",
                        end="",
                    )

        if show_progress:
            print(f"\rProcessed {total_pages}/{total_pages} pages (100%)")

        return processed_docs


async def main():
    """Main function to run the scraper."""
    scraper = CHTDocCrawler()
    results = scraper.crawl()

    # Show final summary
    print(f"\nSaved {len(results)} documents to {scraper.data_dir}")


if __name__ == "__main__":
    asyncio.run(main())
