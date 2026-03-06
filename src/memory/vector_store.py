import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Local directory for ChromaDB
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_db")

def get_vector_store():
    """
    Returns the ChromaDB vector store instance, initialized with OpenAI embeddings.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    os.makedirs(DB_DIR, exist_ok=True)
    
    vectorstore = Chroma(
        collection_name="research_summaries",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    return vectorstore

def save_to_vector_store(topic: str, summary: str):
    """
    Saves a generated topic and summary to the vector store.
    
    Args:
        topic (str): The researched topic.
        summary (str): The generated summary to index.
    """
    vectorstore = get_vector_store()
    document = f"Topic: {topic}\nSummary: {summary}"
    metadatas = [{"topic": topic}]
    
    vectorstore.add_texts(texts=[document], metadatas=metadatas)

def retrieve_from_vector_store(query: str, k: int = 2) -> list[str]:
    """
    Retrieves the top k most similar past summaries based on the query.
    
    Args:
        query (str): The topic or query to search context for.
        k (int): Returns the top k similar documents.
    
    Returns:
        list[str]: The retrieved context bodies.
    """
    vectorstore = get_vector_store()
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]