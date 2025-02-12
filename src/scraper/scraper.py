#!/usr/bin/env python3
"""
CHT documentation scraper using Firecrawl.
"""
from typing import Dict, List
import asyncio
from firecrawl import FirecrawlApp
import json
import os
from datetime import datetime, timezone
from utils import load_config


class CHTDocScraper:
    """Scraper for Community Health Toolkit documentation."""

    def __init__(
        self, base_url: str = "https://docs.communityhealthtoolkit.org"
    ):
        """Initialize the scraper.

        Args:
            base_url: The base URL of the CHT documentation.
        """
        config = load_config()

        self.base_url = base_url
        self.crawler = FirecrawlApp(api_key=config["FIRECRAWL_API_KEY"])

        # Create data directory if it doesn't exist
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "scraped_docs",
        )
        os.makedirs(self.data_dir, exist_ok=True)

    async def scrape(self) -> List[Dict]:
        """Scrape the CHT documentation.

        Returns:
            List of dictionaries containing scraped content and metadata.
        """
        # Configure crawling rules
        self.crawler.allow_patterns(
            [
                r"/docs/.*",  # Only crawl documentation pages
            ]
        )

        # Configure content extraction
        self.crawler.extract(
            {
                "title": ".//h1",  # Main title
                "content": "//article",  # Main content
                "section": '//nav[@aria-label="Breadcrumb"]',  # Navigation breadcrumb
                "last_updated": "//time",  # Last updated timestamp if available
            }
        )

        # Start crawling
        results = await self.crawler.run()

        # Process and save results
        processed_results = []
        for result in results:
            processed_doc = {
                "url": result.url,
                "title": result.extracted.get("title", ""),
                "content": result.extracted.get("content", ""),
                "section": result.extracted.get("section", ""),
                "last_updated": result.extracted.get("last_updated", ""),
                "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            processed_results.append(processed_doc)

        # Save to file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.data_dir, f"cht_docs_{timestamp}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_results, f, ensure_ascii=False, indent=2)

        return processed_results


async def main():
    """Main function to run the scraper."""
    scraper = CHTDocScraper()
    results = await scraper.scrape()
    print(f"Scraped {len(results)} documents")


if __name__ == "__main__":
    asyncio.run(main())
