import pytest
from src.tools.web_search import search_web, scrape_url

from unittest.mock import patch

def test_search_web_returns_results():
    """
    Test that web search returns a list of dictionaries with expected keys.
    """
    with patch('src.tools.web_search.DDGS') as mock_ddgs:
        mock_instance = mock_ddgs.return_value.__enter__.return_value
        mock_instance.text.return_value = [
            {"href": "https://langchain.com", "body": "LangChain framework", "title": "LangChain"}
        ]
        
        results = search_web.invoke({"query": "LangChain", "max_results": 2})
        assert isinstance(results, list)
        assert len(results) > 0
        assert "href" in results[0]
        assert "body" in results[0]
        assert "title" in results[0]

def test_scrape_url_valid():
    """
    Test that scraping a simple valid URL returns a string of text.
    """
    with patch('src.tools.web_search.requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"<html><body>Example Domain</body></html>"
        mock_get.return_value.raise_for_status = lambda: None
        
        result = scrape_url.invoke({"url": "https://example.com"})
        assert isinstance(result, str)
        assert "Example Domain" in result

def test_scrape_url_invalid():
    """
    Test that scraping an invalid URL handles the error gracefully.
    """
    result = scrape_url.invoke({"url": "https://invalid.example.invalid"})
    assert "Error scraping URL" in result