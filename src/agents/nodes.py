import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.state import AgentState
from src.tools.web_search import search_web

def search_node(state: AgentState) -> AgentState:
    """
    Workflow node that executes a web search based on the topic.
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        AgentState: The updated state with search_results.
    """
    topic = state.get("topic", "")

    results = search_web.invoke({"query": topic, "max_results": 3})
    
    return {"search_results": results}