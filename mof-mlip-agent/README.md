# MOF MLIP Agent (LangChain + OpenAI)

## 1) Setup
Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Set your OpenAI API key
```bash
export OPENAI_API_KEY="YOUR_KEY"
```


## 3) (Optional) Configure local PDF RAG

By default, the agent can use your own PDF papers/notes as a local RAG source
to help with novelty checking (in addition to any external searches).

- **Default folder**: `./local_pdfs`
- You can override this with:

```bash
export LOCAL_PDF_DIR="/path/to/my/pdfs"
```

Steps:

1. Create the folder if it does not exist:

```bash
mkdir -p local_pdfs
```

2. Drop your MOF / MLIP related PDFs into that folder.

During the `novelty` step, the agent will search these PDFs (using PyMuPDF)
and pass the most relevant snippets to the LLM, reducing external token usage.

## 4) Run
You can either call the module directly:

```bash
python -m app.run "Evaluate whether UiO-66 is stable under geometry relaxation using a ML interatomic potential. Use fmax 0.05 eV/Å and max 500 steps. Output a JSON spec."
```

or use the convenience script `run.sh` (which also sets some defaults
for RAG and model configuration):

```bash
bash run.sh "Your scientific research query in quotes"
```

## 5) Check Output

- ./outputs/<exp_id>.json
  
## 6) Project structure

High-level layout of the MOF MLIP Agent package:

- `app/`  
  Core Python package for the agent.

  - `run.py`  
    Command-line entrypoint. Parses the user query string, loads configuration, sets up the LLM and persistent memory, builds the LangGraph, invokes it, handles novelty verdicts, and writes the final JSON experiment spec to `outputs/`.

  - `config.py`  
    Defines `AppConfig` (Pydantic model) and `load_config()`, which reads environment variables and returns a typed configuration object (model name, temperature, arXiv limits, output directory, memory paths, etc.).

  - `schemas.py`  
    Pydantic models that define all structured data exchanged between chains and the graph: `QueryIntent`, `CanonicalQuery`, `PaperRef`, `NoveltyVerdict`, and `ExperimentSpec`.

  - `graph.py`  
    Builds the LangGraph state machine (`build_graph`) that wires together the reasoning steps (`intent`, `canonicalize`, `retrieve`, `novelty`, `spec`). Each node is implemented as a Python function that reads/writes fields on the shared `AgentState`.

  - `__init__.py`  
    Convenience re-exports for common helpers (`make_exp_id`, `ensure_dir`, `write_json`) so they can be imported directly from `app`.

  - `chains/`  
    LLM chains for each reasoning stage.

    - `intent.py` – `build_intent_chain()` and helpers to parse the user's free-form query into a structured `QueryIntent`.  
    - `canonicalize.py` – `build_canonicalize_chain()` and `intent_to_jsonable()` to rewrite the query and intent into a precise `CanonicalQuery`.  
    - `novelty.py` – `build_novelty_chain()` to decide whether a proposed experiment is novel vs prior literature and local RAG context.  
    - `specgen.py` – `build_spec_chain()` and `novelty_to_jsonable()` to turn the canonical query + novelty info into a structured `ExperimentSpec`.

    - `__init__.py` – Aggregates the chain builder functions into a single import surface.

  - `tools/`  
    External and internal tools used by the graph.

    - `arxiv_tool.py` – Thin wrapper around LangChain's `ArxivLoader` plus a helper to compact arXiv documents into a single text block.  
    - `local_rag.py` – Local PDF RAG implementation using PyMuPDF. Scans `local_pdfs/` (or `LOCAL_PDF_DIR`), chunks and scores text, and returns the top-matching passages and simple metadata.  
    - `__init__.py` – Re-exports tool functions for easy importing.

  - `utils/`  
    Generic utilities.

    - `ids.py` – Helper for generating experiment IDs (`make_exp_id`).  
    - `io.py` – Simple file I/O helpers like `ensure_dir` and `write_json`.

  - `memory/`  
    Lightweight persistent memory store for past runs.

    - `store.py` – `MemoryStore` implementation (append JSONL records, retrieve top-k by keyword overlap, and format them into a compact context string).  
    - `__init__.py` – Exposes `MemoryStore` at `app.memory`.

- `local_pdfs/`  
  Folder where you can drop PDF papers/notes for local RAG (created automatically on first use if missing).

- `outputs/`  
  Default directory for generated JSON experiment specs.

