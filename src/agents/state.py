from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    """
    Represents the state of the agent as it moves through the LangGraph nodes.
    
    Attributes:
        topic (str): The research topic provided by the user.
        past_context (List[str]): Past context retrieved from Vector Store.
        search_results (List[Dict[str, str]]): List of search results from the web.
        summary (str): The final generated summary.
    """
    topic: str
    past_context: List[str]
    search_results: List[Dict[str, str]]
    summary: str