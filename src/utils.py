"""Utility functions for the CHT Documentation Q&A Chatbot."""

import os
from typing import Dict, Any
from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables.

    Returns:
        Dictionary containing configuration values.
    """
    load_dotenv()

    required_vars = [
        "PINECONE_API_KEY",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "FIRECRAWL_API_KEY",
        "HONEY_HIVE_API_KEY",
        "AGENT_ID",
    ]

    config = {}
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            missing_vars.append(var)
        config[var] = value

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    return config


def clean_text(text: str) -> str:
    """Clean and normalize text content.

    Args:
        text: Raw text content.

    Returns:
        Cleaned and normalized text.
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove special characters that might interfere with processing
    text = text.replace("\u200b", "")  # Zero-width space
    text = text.replace("\u200e", "")  # Left-to-right mark

    return text.strip()


def get_project_root() -> str:
    """Get the absolute path to the project root directory.

    Returns:
        Absolute path to the project root.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_directory(path: str) -> str:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The absolute path to the directory.
    """
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def get_data_dir() -> str:
    """Get the path to the data directory, creating it if necessary.

    Returns:
        Absolute path to the data directory.
    """
    data_dir = os.path.join(get_project_root(), "data")
    return ensure_directory(data_dir)


def get_scraped_docs_dir() -> str:
    """Get the path to the scraped documents directory, creating it if necessary.

    Returns:
        Absolute path to the scraped documents' directory.
    """
    scraped_dir = os.path.join(get_data_dir(), "scraped_docs")
    return ensure_directory(scraped_dir)


_honeyhive_initialized = False

def init_honeyhive():
    """Initialize HoneyHive with configuration.
    
    This function ensures HoneyHive is only initialized once.
    """
    global _honeyhive_initialized
    
    if _honeyhive_initialized:
        return
        
    config = load_config()
    
    # Print debug info before initialization
    print("Initializing HoneyHive with:")
    print(f"API Key: {config['HONEY_HIVE_API_KEY'][:8]}...")  # Only show first 8 chars
    print(f"Project: {config['AGENT_ID']}")

    # Initialize HoneyHive tracing
    HoneyHiveTracer.init(
        api_key=config["HONEY_HIVE_API_KEY"],
        project=config["AGENT_ID"],
    )
    print("HoneyHive initialization successful")
    
    _honeyhive_initialized = True
