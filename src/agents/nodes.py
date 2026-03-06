import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.state import AgentState
from src.tools.web_search import search_web
from src.memory.vector_store import retrieve_from_vector_store, save_to_vector_store

def retrieve_node(state: AgentState) -> AgentState:
    """
    Workflow node that queries ChromaDB for contextual background on the topic.
    """
    topic = state.get("topic", "")
    try:
        past_context = retrieve_from_vector_store(topic, k=2)
    except Exception as e:
        print(f"Warning: Failed to retrieve from vector store: {e}")
        past_context = []
        
    return {"past_context": past_context}

def save_node(state: AgentState) -> AgentState:
    """
    Workflow node that saves the generated summary back to ChromaDB.
    """
    topic = state.get("topic", "")
    summary = state.get("summary", "")
    
    if summary and summary != "No research content found to summarize.":
        try:
            save_to_vector_store(topic, summary)
        except Exception as e:
            print(f"Warning: Failed to save to vector store: {e}")
            
    return {}

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
    Workflow node that summarizes the gathered search results and incorporates past context..
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        AgentState: The updated state with the generated summary.
    """
    topic = state.get("topic", "")
    results = state.get("search_results", [])
    past_context = state.get("past_context", [])
    
    if not results:
        return {"summary": "No research content found to summarize."}

    # Format the past context
    past_context_str = ""
    if past_context:
        past_context_str = "Prior Context from Vector DB:\n" + "\n\n".join(past_context) + "\n\n"
        
    # Format the results for the LLM context
    context = "\n\n".join([f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}" for r in results])
    
    prompt = f"Topic: {topic}\n\n{past_context_str}Search Results:\n{context}\n\nPlease provide a concise summary of the topic based on the search results and any available prior context."

    
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    messages = [
        SystemMessage(content="You are an expert research and summarization agent."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {"summary": str(response.content)}