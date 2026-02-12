"""
Construction of the LangGraph state machine for the MoF(Metal-organic Framework) MLIP(Machine Learning interatomic Potential) Agent.

The graph orchestrates the following high-level steps:
1. intent        – parse the free-form query into a structured QueryIntent.
2. canonicalize  – rewrite that into a precise CanonicalQuery string.
3. retrieve      – pull external (arXiv) and local (PDF) context.
4. novelty       – decide if the proposal is novel enough to pursue.
5. spec          – generate a concrete ExperimentSpec JSON object.

Each node is implemented as a Python function that reads and updates an
AgentState dict. This module only deals with control flow; all LLM logic
is encapsulated in the chain builders from app.chains.*.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from app.schemas import QueryIntent, CanonicalQuery, NoveltyVerdict, ExperimentSpec
from app.chains.intent import build_intent_chain
from app.chains.canonicalize import build_canonicalize_chain, intent_to_jsonable
from app.chains.novelty import build_novelty_chain
from app.chains.specgen import build_spec_chain, novelty_to_jsonable
from app.tools.arxiv_tool import fetch_arxiv_docs, docs_to_compact_text
from app.tools.local_rag import local_rag_search

# Safety limits to avoid over-long prompts to the LLM. These caps are
# intentionally conservative and operate on character counts rather than
# exact tokens to keep implementation simple.
MAX_LITERATURE_CHARS = 15000
MAX_LOCAL_CTX_CHARS = 8000
# Tighter limits when fast_mode is enabled (smaller prompts = faster API round-trips)
MAX_LITERATURE_CHARS_FAST = 5000
MAX_LOCAL_CTX_CHARS_FAST = 3000


class AgentState(TypedDict):
    query_original: str
    memory_context: str

    intent: Optional[QueryIntent]
    canonical: Optional[CanonicalQuery]
    literature_text: str
    local_ctx: str
    novelty: Optional[NoveltyVerdict]
    exp_id: str
    spec: Optional[ExperimentSpec]
    reject_reason: str


def build_graph(llm: ChatOpenAI, arxiv_max_docs: int, fast_mode: bool = False):
    """
    Construct and compile the LangGraph state machine for the agent.

    Parameters
    ----------
    llm:
        ChatOpenAI instance used by all chains.
    arxiv_max_docs:
        Upper bound on the number of arXiv documents to load per run.
    fast_mode:
        If True, use smaller context limits and skip the novelty LLM call
        (go straight from retrieve to spec) for faster runs.
    """
    max_lit = MAX_LITERATURE_CHARS_FAST if fast_mode else MAX_LITERATURE_CHARS
    max_local = MAX_LOCAL_CTX_CHARS_FAST if fast_mode else MAX_LOCAL_CTX_CHARS

    intent_chain = build_intent_chain(llm)
    canon_chain = build_canonicalize_chain(llm)
    novelty_chain = build_novelty_chain(llm)
    spec_chain = build_spec_chain(llm)

    def step_intent(state: AgentState):
        """
        First node: infer the user's high-level intent from the raw query
        and any retrieved memory context.
        """
        intent = intent_chain.invoke({
            "query": state["query_original"],
            "memory_context": state["memory_context"],
        })
        return {"intent": intent}

    def step_canonicalize(state: AgentState):
        """
        Second node: canonicalise the free-form query into a precise,
        MLIP(Machine Learning interatomic Potential)-ready CanonicalQuery, taking the parsed intent into account.
        """
        intent = state["intent"]
        canonical = canon_chain.invoke({
            "query": state["query_original"],
            "intent_json": intent_to_jsonable(intent),
            "memory_context": state["memory_context"],
        })
        return {"canonical": canonical}

    def step_retrieve(state: AgentState):
        """
        Third node: retrieve external and local context relevant to the
        canonical question:
        - arXiv literature via LangChain loaders
        - local PDF context via the local_rag_search tool

        If the arXiv API fails (e.g. HTTP 500, 429 rate limit), we continue
        with empty literature so the pipeline still runs using local RAG
        and memory only.
        """
        canonical = state["canonical"].query_canonical

        # arXiv can return HTTP 500 (server error) or 429 (rate limit).
        # Use a short query to avoid huge URLs and rate limits; on failure
        # use empty literature so the run does not crash.
        lit = ""
        arxiv_query = (canonical[:200] + "...") if len(canonical) > 200 else canonical
        try:
            docs = fetch_arxiv_docs(arxiv_query, max_docs=arxiv_max_docs)
            lit = docs_to_compact_text(docs)
        except Exception:
            pass  # Continue with lit = ""; novelty will see "(no results)"

        if lit:
            lit = lit[:max_lit]

        local_ctx, _refs = local_rag_search(canonical)
        if local_ctx:
            local_ctx = local_ctx[:max_local]

        out: dict = {"literature_text": lit, "local_ctx": local_ctx}
        if fast_mode:
            # Skip novelty LLM call: inject a pass verdict so spec step has something to read
            out["novelty"] = NoveltyVerdict(
                status="pass",
                rationale="(fast mode: novelty check skipped)",
                top_refs=[],
            )
        return out

    def step_novelty(state: AgentState):
        """
        Fourth node: pass the canonical question and gathered context to
        the novelty chain, which decides whether to reject, pass, or mark
        the proposal as uncertain.
        """
        canonical = state["canonical"].query_canonical
        novelty = novelty_chain.invoke({
            "canonical": canonical,
            "memory_context": state["memory_context"],
            "lit": state["literature_text"] or "(no results)",
            "local_ctx": state["local_ctx"] or "(none)",
        })
        reject_reason = novelty.rationale if novelty.status == "reject" else ""
        return {"novelty": novelty, "reject_reason": reject_reason}

    def route_after_retrieve(state: AgentState):
        """In fast mode skip the novelty node and go straight to spec."""
        return "spec" if fast_mode else "novelty"

    def route_after_novelty(state: AgentState):
        """
        Small routing function used by LangGraph to decide whether the
        graph should terminate early (on rejection) or proceed to spec
        generation.
        """
        n = state["novelty"]
        if n and n.status == "reject":
            return "reject"
        return "spec"

    def step_spec(state: AgentState):
        """
        Final node: construct the concrete ExperimentSpec JSON object
        from the canonical query, novelty verdict, and past-run context.
        """
        novelty_json = novelty_to_jsonable(state["novelty"])
        spec = spec_chain.invoke({
            "query_original": state["query_original"],
            "query_canonical": state["canonical"].query_canonical,
            "memory_context": state["memory_context"],
            "novelty_json": novelty_json,
            "exp_id": state["exp_id"],
        })
        return {"spec": spec}

    # Define the graph structure by registering nodes and edges between them.
    g = StateGraph(AgentState)
    g.add_node("intent", step_intent)
    g.add_node("canonicalize", step_canonicalize)
    g.add_node("retrieve", step_retrieve)
    g.add_node("novelty", step_novelty)
    g.add_node("spec", step_spec)

    g.set_entry_point("intent")
    g.add_edge("intent", "canonicalize")
    g.add_edge("canonicalize", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"novelty": "novelty", "spec": "spec"})
    g.add_conditional_edges("novelty", route_after_novelty, {"reject": END, "spec": "spec"})
    g.add_edge("spec", END)

    return g.compile()
