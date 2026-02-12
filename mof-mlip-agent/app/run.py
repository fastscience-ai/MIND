"""
CLI entrypoint for the MOF MLIP Agent.

This module:
- parses the user query from the command line,
- loads application configuration,
- initialises the OpenAI chat model,
- wires up the LangGraph-based agent,
- runs the full reasoning pipeline, and
- writes the final JSON experiment specification to disk while
  recording a memory entry for future runs.
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
    if len(sys.argv) < 2:
        print("Usage: python -m app.run \"<your English query>\"")
        sys.exit(1)

    # Take the first argument as the natural-language query the user typed.
    query_original = sys.argv[1].strip()
    if not query_original:
        print("Empty query.")
        sys.exit(1)

    # Load configuration (model name, temperature, I/O paths, etc.).
    cfg = load_config()

    # Guardrail: ensure the OpenAI API key is present before constructing
    # any client objects. This fails fast with a clear error message.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")

    # Create the chat model wrapper used by all chains.
    llm = ChatOpenAI(model=cfg.openai_model, temperature=cfg.temperature)

    # Persistent memory:
    # - retrieve: fetch top-k similar past runs to provide context
    # - format_context: render them as a compact string for prompts
    mem = MemoryStore(path=cfg.memory_file, max_items=cfg.memory_max_items)
    retrieved = mem.retrieve(query_original, k=cfg.memory_retrieve_k)
    memory_context = mem.format_context(retrieved)

    # Unique identifier for this experimental specification.
    exp_id = make_exp_id("mof")

    # Compile the LangGraph that encodes the agent's control flow.
    app = build_graph(llm, arxiv_max_docs=cfg.arxiv_max_docs)

    # Initial graph state. Nodes incrementally fill in these fields.
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

    # Run the full agent graph synchronously and collect the final state.
    out = app.invoke(state)

    novelty = out.get("novelty")
    intent = out.get("intent")
    canonical = out.get("canonical")

    # If the novelty gate rejects the experiment, print a human-readable
    # explanation and persist a memory record so future runs can recognise
    # that this hypothesis was already evaluated and rejected.
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

    # At this point we expect a concrete ExperimentSpec; if missing, treat as error.
    spec = out.get("spec")
    if spec is None:
        print("No spec generated (unexpected).")
        sys.exit(2)

    # Ensure the output directory exists, then write the JSON spec to disk.
    ensure_dir(cfg.output_dir)
    path = os.path.join(cfg.output_dir, f"{spec.exp_id}.json")
    write_json(path, spec.model_dump())

    print("\nNOVELTY VERDICT:", novelty.status if novelty else "(none)")
    print("Wrote experiment spec to:", path)
    print("\nCanonical query:\n", spec.query_canonical)

    # Store memory record for future runs, even for successful / passed
    # experiments. This allows the agent to stay consistent over time
    # and avoid regenerating very similar specifications.
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
