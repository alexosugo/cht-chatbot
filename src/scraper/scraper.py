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

    async def scrape(self, show_progress: bool = True) -> List[Dict]:
        """Scrape the CHT documentation.

        Args:
            show_progress: Whether to show progress updates during scraping

        Returns:
            List of dictionaries containing scraped content and metadata.
        """
        if show_progress:
            print("Initializing crawler with parameters:")
            print("  - Base URL:", self.base_url)
            print("  - Page limit: 100")
            print("  - Formats: markdown, html")
            print("\nStarting crawl...")

        # Start crawling the documentation
        crawl_result = self.crawler.async_crawl_url(
            self.base_url,
            params = {
                'scrapeOptions': {'formats': ['markdown', 'html']},
                'limit': 1,
            }
        )
        
        # Get the crawl ID
        crawl_id = crawl_result['id']
        
        # Poll for completion
        progress_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        progress_idx = 0
        pages_found = 0

        while True:
            try:
                status = self.crawler.check_crawl_status(crawl_id)
                print('Crawl status:', status)

                if show_progress:
                    print("\nReceived status:", status)  # Debug line
                
                current_status = status.get('status', '')
                result = status.get('data')
                
                if result is None:
                    if show_progress:
                        print("Warning: No 'data' field in status response")
                    current_pages = 0
                else:
                    current_pages = len(result)

                if current_status == 'completed':
                    if show_progress:
                        print(f"\rCrawl completed! Found {current_pages} pages.")
                    if not result:
                        raise Exception("Crawl completed but no results were returned")
                    scraped_data = self._process_crawl_result(result, show_progress)
                    break
                elif current_status == 'failed':
                    if show_progress:
                        print("\nCrawl failed!")
                    raise Exception(f"Crawl failed: {status.get('error')}")
            except Exception as e:
                print(f"\nError checking crawl status: {str(e)}")
                raise
            
            if show_progress:
                # Always update the progress indicator
                print(f"\r{progress_chars[progress_idx]} Scraping CHT documentation... Status: {current_status}, Pages found: {current_pages}", end='')
                progress_idx = (progress_idx + 1) % len(progress_chars)
                
                # Show new pages found
                if current_pages > pages_found:
                    print(f"\n  → Found {current_pages - pages_found} new pages")
                    pages_found = current_pages
            
            await asyncio.sleep(2)  # Wait before checking again
        
        # Save the scraped data
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.data_dir, f'cht_docs_{timestamp}.json')
        
        with open(output_file, 'w') as f:
            json.dump(scraped_data, f, indent=2)
            
        return scraped_data

    def _process_crawl_result(self, crawl_result: List[Dict], show_progress: bool = True) -> List[Dict]:
        """Process the crawl result to extract useful content.

        Args:
            crawl_result: Raw crawl result from Firecrawl
            show_progress: Whether to show progress updates

        Returns:
            List of processed documents with extracted content
        """
        processed_docs = []
        total_pages = len(crawl_result)
        
        if show_progress:
            print(f"\nProcessing {total_pages} pages...")
        
        for i, page in enumerate(crawl_result, 1):
            # Extract content from the page
            content = []
            if 'extracted' in page:
                extracted = page['extracted']
                
                # Add title if available
                if 'title' in extracted and extracted['title']:
                    content.append(extracted['title'][0])  # Usually one title per page
                
                # Add main content
                if 'content' in extracted:
                    content.extend(extracted['content'])
                
                # Add other text content
                if 'text' in extracted:
                    content.extend(extracted['text'])
            
            # Create a document with metadata and content
            doc = {
                'url': page.get('url', ''),
                'title': extracted.get('title', [''])[0] if 'extracted' in page and 'title' in extracted else '',
                'content': '\n'.join(content),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            processed_docs.append(doc)
            
            if show_progress:
                if i == 1:
                    print("\nStarting content extraction...")
                if i % 5 == 0:  # Show progress more frequently
                    print(f"\rProcessing content: {i}/{total_pages} pages ({(i/total_pages)*100:.1f}%) - Current: {doc['title']}", end='')
        
        if show_progress:
            print(f"\rProcessed {total_pages}/{total_pages} pages (100%)")
        
        return processed_docs

async def main():
    """Main function to run the scraper."""
    scraper = CHTDocScraper()
    results = await scraper.scrape(show_progress=True)
    
    # Show final summary
    print(f"\nSaved {len(results)} documents to {scraper.data_dir}")


if __name__ == "__main__":
    asyncio.run(main())
