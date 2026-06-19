from config.configuration import get_settings
from config.logging_config import setup_logging
from core.graph.graph import build_graph
from core.state import ChatState

setup_logging(get_settings().log_level)

graph = build_graph()



def run_chat(user_input: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    # Only send the new message. reservation_fields / retrieved_chunks have no
    # reducer, so re-sending them each turn would overwrite the state the
    # checkpointer persists — wiping a reservation that's mid-collection.
    initial_state: ChatState = {"user_input": user_input}
    response = graph.invoke(
        initial_state,
        config=config,
    )
    print(f"Response: {response}\n")
    return response["response"]

if __name__ == "__main__":
    print("🅿️ Parking Assistant Ready! (type 'quit' to exit)\n")
    print("RAG Chatbot Skeleton — type 'quit' to exit.")
    print("Try inputs like:")
    print("  - 'where is the entrance?'        (info_query  → rag_agent)")
    print("  - 'is floor 2 available?'         (dynamic_query → dynamic_agent)")
    print("  - 'I want to book a spot'         (reservation → reservation_agent)")
    print("  - 'write me a poem'               (out_of_scope)")
    print("-" * 60)
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        try:
            response = run_chat(user_input)
            print(f"Bot: {response}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")
