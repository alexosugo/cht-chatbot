# CHT Documentation Q&A Chatbot

A RAG-based Q&A chatbot for the Community Health Toolkit (CHT) documentation.

## Features
- Semantic search using Pinecone
- Context-aware responses using Google Gemini
- Source citations for transparency
- Both CLI and web interfaces

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a .env file with your API keys:
```
PINECONE_API_KEY=your_key_here
PINECONE_ENVIRONMENT=your_environment_here
GOOGLE_API_KEY=your_key_here
```

3. Run the CLI interface:
```bash
python src/cli/interface.py
```

4. Run the web interface:
```bash
streamlit run src/web/app.py
```

## Setup

#### Scrape new documentation and start CLI interface
```bash
python src/main.py --scrape --mode cli
```

#### Start web interface with existing data
```bash
python src/main.py --mode web
```

#### Just scrape and process new documentation
```bash
python src/main.py --scrape
```

## Project Structure
```
rag-chatbot/
├── .env                  # Environment variables
├── requirements.txt      # Project dependencies
├── README.md            # Project documentation
├── src/
│   ├── scraper/         # Web scraping module
│   ├── core/            # Core RAG functionality
│   ├── cli/             # Command-line interface
│   ├── web/             # Web interface
│   └── utils.py         # Utility functions
└── data/                # Scraped and processed data
