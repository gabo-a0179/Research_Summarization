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

def summarize_node(state: AgentState) -> AgentState:
    """
    Workflow node that summarizes the gathered search results.
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        AgentState: The updated state with the generated summary.
    """
    topic = state.get("topic", "")
    results = state.get("search_results", [])
    
    if not results:
        return {"summary": "No research content found to summarize."}
        
    # Format the results for the LLM context
    context = "\n\n".join([f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}" for r in results])
    
    prompt = f"Topic: {topic}\n\nSearch Results:\n{context}\n\nPlease provide a concise summary of the topic based ONLY on the search results."
    
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    messages = [
        SystemMessage(content="You are an expert research and summarization agent."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {"summary": str(response.content)}