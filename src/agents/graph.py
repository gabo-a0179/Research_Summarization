from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.nodes import search_node, summarize_node, retrieve_node, save_node

def build_graph():
    """
    Build and compile the StateGraph for the research agent.
    
    Returns:
        CompiledGraph: The compiled LangGraph application.
    """
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("search_node", search_node)
    builder.add_node("summarize_node", summarize_node)
    builder.add_node("save_node", save_node)
    
    # Add edges
    # START -> retrieve -> search -> summarize -> save -> END
    builder.add_edge(START, "retrieve_node")
    builder.add_edge("retrieve_node", "search_node")
    builder.add_edge("search_node", "summarize_node")
    builder.add_edge("summarize_node", "save_node")
    builder.add_edge("save_node", END)
    
    # Compile the graph
    app = builder.compile()
    
    return app