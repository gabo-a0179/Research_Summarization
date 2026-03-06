"""
Entry point for the Research & Summarization Agent workflow.
"""
import os
from dotenv import load_dotenv
from src.agents.graph import build_graph

def main():
    load_dotenv()
    
    print("Agent initialized.")
    app = build_graph()
    
    topic = input("Enter a topic to research: ")
    if not topic.strip():
        print("Empty topic, exiting.")
        return
        
    print(f"\nStarting research on: {topic}")
    
    initial_state = {"topic": topic, "search_results": [], "summary": ""}
    
    # Stream the graph execution
    for output in app.stream(initial_state):
        # Iterate over node outputs
        for key, value in output.items():
            print(f"\n--- Output from {key} ---")
            if "search_results" in value:
                count = len(value["search_results"])
                print(f"Found {count} references.")
            if "summary" in value:
                print(value["summary"])

    print("\nWorkflow completed.")

if __name__ == "__main__":
    main()