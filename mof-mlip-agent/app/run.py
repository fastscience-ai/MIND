"""
CLI entrypoint for the MoF(Metal-organic Framework) MLIP(Machine Learning interatomic Potential) Agent.

[For students] This file is the main entry point when you run:
    python -m app.run "Your scientific research query"

It (1) reads your query from the command line, (2) loads config and memory,
(3) builds the LangGraph agent (intent → canonicalize → retrieve → novelty → spec),
(4) runs the graph once, and (5) either writes a JSON experiment spec to disk
or exits early if the novelty check rejects the idea. All steps are commented
below so you can follow the flow.
"""

import sys
import os
from datetime import datetime
from langchain_openai import ChatOpenAI

from app.config import load_config
from app.utils.ids import make_exp_id
from app.utils.io import ensure_dir, write_json
from app.graph import build_graph
from app.memory import MemoryStore


def main():
    """
    Main command-line entrypoint.

    Usage (from project root):
        python -m app.run "Your scientific research query"
    """
    # -------------------------------------------------------------------------
    # Step 1: Parse the user's query from the command line
    # -------------------------------------------------------------------------
    # sys.argv[0] is the program name (e.g. "app.run"), sys.argv[1] is the first
    # argument. We require exactly one argument: the natural-language query.
    if len(sys.argv) < 2:
        print("Usage: python -m app.run \"<your English query>\"")
        sys.exit(1)

    query_original = sys.argv[1].strip()
    if not query_original:
        print("Empty query.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 2: Load configuration from environment variables
    # -------------------------------------------------------------------------
    # cfg contains: openai_model, temperature, arxiv_max_docs, output_dir,
    # memory_file, memory_max_items, memory_retrieve_k, fast_mode.
    cfg = load_config()

    # -------------------------------------------------------------------------
    # Step 3: Check that the OpenAI API key is set (required for all LLM calls)
    # -------------------------------------------------------------------------
    # Without this key, the ChatOpenAI client would fail later with a confusing
    # error. We fail early with a clear message so the student knows what to do.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")

    # -------------------------------------------------------------------------
    # Step 4: Create the LLM (Large Language Model) client
    # -------------------------------------------------------------------------
    # This single client is passed into the graph and used by every chain
    # (intent, canonicalize, novelty, spec). temperature=0 keeps outputs deterministic.
    llm = ChatOpenAI(model=cfg.openai_model, temperature=cfg.temperature)

    # -------------------------------------------------------------------------
    # Step 5: Load "memory" from past runs and format it for the prompts
    # -------------------------------------------------------------------------
    # Memory is stored in a JSONL file. We retrieve the top-k runs that are most
    # similar to the current query (by simple keyword overlap), then format
    # them into one string (memory_context). The agent will see this string in
    # each LLM prompt so it can stay consistent with previous experiments.
    mem = MemoryStore(path=cfg.memory_file, max_items=cfg.memory_max_items)
    retrieved = mem.retrieve(query_original, k=cfg.memory_retrieve_k)
    memory_context = mem.format_context(retrieved)

    # -------------------------------------------------------------------------
    # Step 6: Generate a unique experiment ID and build the agent graph
    # -------------------------------------------------------------------------
    # exp_id is used in the final JSON spec (e.g. "mof-20260212-1234").
    # build_graph() returns a compiled LangGraph: a state machine that runs
    # the nodes "intent" → "canonicalize" → "retrieve" → "novelty" (or skip) → "spec".
    exp_id = make_exp_id("mof")
    app = build_graph(llm, arxiv_max_docs=cfg.arxiv_max_docs, fast_mode=cfg.fast_mode)

    # -------------------------------------------------------------------------
    # Step 7: Prepare the initial state for the graph
    # -------------------------------------------------------------------------
    # The graph runs by updating this state dict. We start with the user query
    # and memory context filled in; all other fields (intent, canonical,
    # literature_text, etc.) are filled by the graph nodes as they run.
    state = {
        "query_original": query_original,
        "memory_context": memory_context,

        "intent": None,
        "canonical": None,
        "literature_text": "",
        "local_ctx": "",
        "novelty": None,
        "exp_id": exp_id,
        "spec": None,
        "reject_reason": "",
    }

    # -------------------------------------------------------------------------
    # Step 8: Run the full graph once (all nodes in sequence)
    # -------------------------------------------------------------------------
    # invoke() runs the graph until it reaches the END node. The returned "out"
    # is the final state after every node has run (intent, canonical, novelty,
    # spec, etc. are now populated).
    out = app.invoke(state)

    novelty = out.get("novelty")
    intent = out.get("intent")
    canonical = out.get("canonical")

    # -------------------------------------------------------------------------
    # Step 9: If the novelty check rejected the experiment, print and save, then exit
    # -------------------------------------------------------------------------
    # The novelty node can return status "reject" if the idea is already in the
    # literature or local PDFs. We print the rationale and top references, append
    # a record to memory (so we remember this rejection next time), and exit
    # without writing a spec file.
    if novelty and novelty.status == "reject":
        print("\nNOVELTY VERDICT: REJECT")
        print(novelty.rationale)
        if novelty.top_refs:
            print("\nTop references:")
            for r in novelty.top_refs[:3]:
                print(f"- {r.title} ({r.id}) :: {r.why_relevant}")

        mem.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "exp_id": exp_id,
            "query_original": query_original,
            "query_canonical": canonical.query_canonical if canonical else "",
            "mof_name": intent.mof_name if intent else "",
            "task_type": intent.task_hint if intent else "",
            "verdict_status": "reject",
            "verdict_rationale": novelty.rationale,
        })
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Step 10: We passed the novelty check — get the generated spec and write it
    # -------------------------------------------------------------------------
    # The spec node should have produced an ExperimentSpec. If not, something
    # went wrong (e.g. the LLM failed); we exit with an error code.
    spec = out.get("spec")
    if spec is None:
        print("No spec generated (unexpected).")
        sys.exit(2)

    # Create the output directory if it does not exist, then write the spec
    # as a pretty-printed JSON file (e.g. outputs/mof-20260212-1234.json).
    ensure_dir(cfg.output_dir)
    path = os.path.join(cfg.output_dir, f"{spec.exp_id}.json")
    write_json(path, spec.model_dump())

    print("\nNOVELTY VERDICT:", novelty.status if novelty else "(none)")
    print("Wrote experiment spec to:", path)
    print("\nCanonical query:\n", spec.query_canonical)

    # -------------------------------------------------------------------------
    # Step 11: Save this run to memory for future runs
    # -------------------------------------------------------------------------
    # Even when we pass the novelty check, we append a summary of this run
    # (query, verdict, task type, etc.) to the memory store. Next time the
    # user asks something similar, the agent will see this run in memory_context
    # and can avoid repeating the same spec or stay consistent.
    mem.append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "exp_id": spec.exp_id,
        "query_original": spec.query_original,
        "query_canonical": spec.query_canonical,
        "mof_name": (intent.mof_name if intent else "") or spec.structure.get("id", ""),
        "task_type": spec.task.get("type", ""),
        "verdict_status": novelty.status if novelty else "pass",
        "verdict_rationale": novelty.rationale if novelty else "",
    })


if __name__ == "__main__":
    main()
