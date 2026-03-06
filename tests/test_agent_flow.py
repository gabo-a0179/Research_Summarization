import pytest
from unittest.mock import patch, MagicMock

from src.agents.state import AgentState
from src.agents.nodes import search_node, summarize_node
from src.agents.graph import build_graph

def test_agent_state_keys():
    """
    Test that AgentState includes topic, search_results, and summary keys.
    """
    state: AgentState = {
        "topic": "AI",
        "search_results": [{"title": "Test", "href": "http://test", "body": "Test body"}],
        "summary": "This is a summary."
    }
    assert "topic" in state
    assert "search_results" in state
    assert "summary" in state
    assert len(state["search_results"]) == 1

@patch("src.agents.nodes.search_web")
def test_search_node(mock_search_web):
    """
    Test the search_node to ensure it fetches search results correctly.
    """
    mock_search_web.invoke.return_value = [{"title": "AI News", "href": "http://ai.net", "body": "Latest AI news"}]
    
    initial_state = {"topic": "AI News", "search_results": [], "summary": ""}
    result_state = search_node(initial_state)
    
    mock_search_web.invoke.assert_called_once_with({"query": "AI News", "max_results": 3})
    assert "search_results" in result_state
    assert len(result_state["search_results"]) == 1
    assert result_state["search_results"][0]["title"] == "AI News"

@patch("src.agents.nodes.search_web")
@patch("src.agents.nodes.ChatOpenAI")
def test_graph_flow(mock_chat_openai, mock_search_web):
    """
    Test the entire StateGraph flow using LangGraph.
    """
    mock_search_web.invoke.return_value = [
        {"title": "AI in 2026", "href": "http://example.com/ai2026", "body": "AI in 2026 is great."}
    ]
    
    mock_llm_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Summary of AI in 2026 is great."
    mock_llm_instance.invoke.return_value = mock_response
    mock_chat_openai.return_value = mock_llm_instance
    
    app = build_graph()
    initial_state = {"topic": "AI Future", "search_results": [], "summary": ""}
    
    final_output = app.invoke(initial_state)
    
    assert "summary" in final_output
    assert "Summary of AI in 2026 is great." in final_output["summary"]
    assert len(final_output["search_results"]) == 1