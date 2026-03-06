from typing import List, Dict, Any
from langchain_core.tools import tool
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

@tool
def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for a given topic.
    
    Args:
        query (str): The search query.
        max_results (int): Maximum number of search results to return.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing 'title', 'href', and 'body' of search results.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
        return results

@tool
def scrape_url(url: str) -> str:
    """
    Scrape text content from a specific URL.
    
    Args:
        url (str): The URL to scrape.
        
    Returns:
        str: The extracted text content from the body of the webpage.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        return text[:5000]
    except Exception as e:
        return f"Error scraping URL: {str(e)}"