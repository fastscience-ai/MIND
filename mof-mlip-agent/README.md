# MoF(Metal-organic Framework) MLIP(Machine Learning interatomic Potential) Agent (LangChain + OpenAI)

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

2. Drop your MoF(Metal-organic Framework) / MLIP(Machine Learning interatomic Potential) related PDFs into that folder.

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

### Run faster (fewer API calls, less context)

Set `FAST_MODE=1` to reduce latency:

- Skips arXiv retrieval (no external calls).
- Uses minimal memory context (1 past run).
- Skips the novelty-check LLM step and goes straight to spec generation (saves one round-trip).

```bash
export FAST_MODE=1
bash run.sh "Your scientific research query in quotes"
```

Or add `export FAST_MODE=1` to `run.sh` if you usually want fast runs.

## 5) Check Output

- ./outputs/<exp_id>.json
  
## 6) Project structure

High-level layout of the MoF(Metal-organic Framework) MLIP(Machine Learning interatomic Potential) Agent package:

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

## 7) System workflow

When you execute:

```bash
python -m app.run "Compute a geometry relaxation for UiO-66 using an ML interatomic potential. Use fmax 0.05 eV/Å and max 500 steps. Return a JSON experiment spec for SevenNet."
```

the following happens in order.

1. **Startup**  
   Config is loaded, `OPENAI_API_KEY` is checked, and a unique experiment ID (e.g. `mof-20260212-1234`) is generated. Past runs are read from the memory store and the top‑k similar entries are formatted into a short **memory context** string.

2. **Intent** (1st LLM call)  
   Your raw query and the memory context are sent to the LLM. It returns a structured **QueryIntent**: MoF(Metal-organic Framework) name (e.g. UiO-66), goal, task type (e.g. relaxation), missing inputs, and whether the request is MLIP(Machine Learning interatomic Potential)-feasible.

3. **Canonicalize** (2nd LLM call)  
   The original query plus the intent (as JSON) and memory context are sent to the LLM. It returns a **CanonicalQuery**: a single, precise sentence suitable for search and spec generation (e.g. “Perform a geometry relaxation of UiO-66 with fmax 0.05 eV/Å, max 500 steps, output a JSON spec for SevenNet.”).

4. **Retrieve** (no LLM)  
   - **arXiv:** The canonical query is sent to the arXiv API; up to `arxiv_max_docs` results are fetched and compacted into a **literature text** block. If the API fails (e.g. 500/429), this block is left empty and the run continues.  
   - **Local PDFs:** The canonical query is used to search PDFs in `local_pdfs/` (PyMuPDF). Top‑matching chunks become **local context**.  
   Both are trimmed to a maximum length to keep prompts manageable.

5. **Novelty** (3rd LLM call)  
   The canonical question, memory context, literature text (arXiv), and local context (PDFs) are sent to the LLM. It returns a **NoveltyVerdict**: pass / reject / uncertain, with a short rationale and optional references.  
   - If **reject**: the run prints the verdict, appends a memory record, and exits without writing a spec.  
   - If **pass** or **uncertain**: the run continues to the next step.

6. **Spec** (4th LLM call)  
   The original and canonical queries, memory context, novelty verdict (as JSON), and experiment ID are sent to the LLM. It returns a structured **ExperimentSpec** (structure, calculator, task, postprocess, notes, etc.).

7. **Finish**  
   The spec is written to `outputs/<exp_id>.json`, and a memory record (query, verdict, task type, etc.) is appended so future runs can reuse this history.

**Summary (normal mode):** 4 LLM calls (intent → canonicalize → novelty → spec), plus one arXiv fetch and one local PDF search. Novelty is decided by the LLM using the retrieved literature and local context.

---

### What is different with fast mode?

When `FAST_MODE=1` (e.g. `export FAST_MODE=1` before the same command):

| Step            | Normal mode                          | Fast mode |
|----------------|--------------------------------------|-----------|
| Memory context | Top‑k past runs (default k=5)        | Only 1 past run (k=1) |
| Retrieve       | arXiv + local PDF search              | **No arXiv** (literature text empty); local PDF search still runs. Shorter caps on literature/local text (5000 / 3000 chars). |
| Novelty        | **LLM call** with lit + local_ctx    | **Skipped.** A synthetic “pass” verdict is injected; no 3rd LLM call. |
| Spec           | 4th LLM call                         | 3rd (and final) LLM call. |

So with fast mode you get **3 LLM calls** (intent → canonicalize → spec), no arXiv call, less memory in the prompt, and smaller context limits. The run is faster and uses fewer tokens; novelty is not evaluated by the model (everything is treated as pass).

