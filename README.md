# CHT Documentation Q&A Chatbot

A RAG-based Q&A chatbot for the Community Health Toolkit (CHT) documentation.

## Features
- Semantic search using Pinecone
- Context-aware responses using Google Gemini
- Source citations for transparency
- Both CLI and web interfaces

## Setup

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/cht-chatbot.git
cd cht-chatbot
```

2. Install the project in development mode:
```bash
pip install -e .
```

3. Create a .env file with your API keys:
```
PINECONE_API_KEY=your_key_here
VERTEX_PROJECT=your_project_id_here
VERTEX_LOCATION=your_project_location_here
VERTEX_API_HOST=your_host_here
HONEY_HIVE_API_KEY=your_key_here (optional)
HELICONE_API_KEY=you_key_here (optional)
FIRECRAWL_API_KEY=your_key_here
```

## Running the Application

After installation, you can run the application in several ways:

1. Run the CLI interface:
```bash
python src/cli/interface.py
```

2. Run the web interface:
```bash
streamlit run src/web/app.py
```

## Usage

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
