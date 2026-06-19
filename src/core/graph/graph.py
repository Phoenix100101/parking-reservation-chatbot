from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph

from core.state import ChatState
from core.nodes.router_node import router_node
from core.nodes.retrieval_node import rag_agent_node
from core.nodes.dynamic_node import dynamic_agent_node
from core.nodes.reservation_node import reservation_agent_node
from core.nodes.out_of_scope_node import out_of_scope_node
from guardrails.input_filter import input_filter_node
from guardrails.output_filter import output_filter_node

# ---------------------------------------------------------------------------
# Node names — kept as constants so edge wiring stays consistent
# ---------------------------------------------------------------------------
INPUT_GUARDRAIL = "input_guardrail"
ROUTER = "router"
RAG_AGENT = "rag_agent"
DYNAMIC_AGENT = "dynamic_agent"
RESERVATION_AGENT = "reservation_agent"
OUTPUT_GUARDRAIL = "output_guardrail"
OUT_OF_SCOPE = "out_of_scope"

# -----------------------------------------------------------------------------
# Conditional Edge Functions
# -----------------------------------------------------------------------------
def route_after_input_guardrail(state: ChatState) -> str:
    return "router" if not state.get("input_blocked") else END


def route_after_router(state: ChatState) -> str:
    intent = state.get("intent")
    if intent == "info_query":
        return RAG_AGENT
    if intent == "dynamic_query":
        return DYNAMIC_AGENT
    if intent == "reservation":
        return RESERVATION_AGENT
    return OUT_OF_SCOPE






def build_graph():
    g = StateGraph(ChatState)

    g.add_node(INPUT_GUARDRAIL, input_filter_node)
    g.add_node(ROUTER, router_node)
    g.add_node(RAG_AGENT, rag_agent_node)
    g.add_node(DYNAMIC_AGENT, dynamic_agent_node)
    g.add_node(RESERVATION_AGENT, reservation_agent_node)
    g.add_node(OUT_OF_SCOPE, out_of_scope_node)
    g.add_node(OUTPUT_GUARDRAIL, output_filter_node)

    g.set_entry_point(INPUT_GUARDRAIL)

    # input_guardrail → router (or END if blocked)
    g.add_conditional_edges(
        INPUT_GUARDRAIL,
        route_after_input_guardrail,
        {ROUTER: ROUTER, END: END},
    )

    # router → one of four branches
    g.add_conditional_edges(
        ROUTER,
        route_after_router,
        {
            RAG_AGENT: RAG_AGENT,
            DYNAMIC_AGENT: DYNAMIC_AGENT,
            RESERVATION_AGENT: RESERVATION_AGENT,
            OUT_OF_SCOPE: OUT_OF_SCOPE,
        },
    )

    # all three agents converge on output_guardrail
    g.add_edge(RAG_AGENT, OUTPUT_GUARDRAIL)
    g.add_edge(DYNAMIC_AGENT, OUTPUT_GUARDRAIL)
    g.add_edge(RESERVATION_AGENT, OUTPUT_GUARDRAIL)
    g.add_edge(OUT_OF_SCOPE, OUTPUT_GUARDRAIL)

    # output_guardrail → END
    g.add_edge(OUTPUT_GUARDRAIL, END)

    # In-memory checkpointer so reservation_fields / history persist across
    # turns. Invoke with config={"configurable": {"thread_id": ...}} per chat.
    # NOTE: process-local and non-durable — swap for a persistent saver
    # (e.g. Postgres) for real deployments.
    return g.compile(checkpointer=MemorySaver())
