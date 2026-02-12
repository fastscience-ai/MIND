"""
Configuration model and loader for the MoF(Metal-organic Framework) MLIP(Machine Learning interatomic Potential) Agent.

All runtime configuration is funneled through the AppConfig Pydantic model,
which makes it easy to reason about and document environment variables.
"""

from pydantic import BaseModel, Field
import os


class AppConfig(BaseModel):
    """
    Strongly-typed application configuration.

    Most fields can be controlled by environment variables; see load_config().
    """

    # We currently fix the model to gpt-4.1-mini to avoid
    # accidentally using long-context variants that hit TPM limits.
    openai_model: str = Field(default="gpt-4.1-mini")
    temperature: float = Field(default=0.0)

    # arXiv search
    arxiv_max_docs: int = Field(default=6)

    # Where to write JSON outputs
    output_dir: str = Field(default="outputs")

    # Persistent memory
    memory_file: str = Field(default="memory/memory_store.jsonl")
    memory_max_items: int = Field(default=50)          # how many to keep on disk (soft)
    memory_retrieve_k: int = Field(default=5)          # how many to inject into prompts

    # Fast mode: skip arXiv, use less memory context, optional skip of novelty step
    fast_mode: bool = Field(default=False)


def load_config() -> AppConfig:
    """
    Read configuration from the environment and return an AppConfig instance.

    Recognised environment variables:
      - OPENAI_TEMPERATURE   : float, default "0.0"
      - ARXIV_MAX_DOCS       : int,   default "6"
      - OUTPUT_DIR           : str,   default "outputs"
      - MEMORY_FILE          : str,   default "memory/memory_store.jsonl"
      - MEMORY_MAX_ITEMS     : int,   default "50"
      - MEMORY_RETRIEVE_K    : int,   default "5"
      - FAST_MODE            : "1" or "0", default "0" (faster: less context, skip arXiv, skip novelty)

    Note: the model name is intentionally pinned to gpt-4.1-mini in code.
    """

    # Force the model to gpt-4.1-mini regardless of environment,
    # so OPENAI_MODEL overrides don't accidentally select a heavy model.
    model = "gpt-4.1-mini"
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    fast_mode = os.getenv("FAST_MODE", "0").strip().lower() in ("1", "true", "yes")
    # In fast mode: no arXiv, minimal memory context
    arxiv_max_docs = 0 if fast_mode else int(os.getenv("ARXIV_MAX_DOCS", "6"))
    output_dir = os.getenv("OUTPUT_DIR", "outputs")

    memory_file = os.getenv("MEMORY_FILE", "memory/memory_store.jsonl")
    memory_max_items = int(os.getenv("MEMORY_MAX_ITEMS", "50"))
    memory_retrieve_k = 1 if fast_mode else int(os.getenv("MEMORY_RETRIEVE_K", "5"))

    return AppConfig(
        openai_model=model,
        temperature=temperature,
        arxiv_max_docs=arxiv_max_docs,
        output_dir=output_dir,
        memory_file=memory_file,
        memory_max_items=memory_max_items,
        memory_retrieve_k=memory_retrieve_k,
        fast_mode=fast_mode,
    )
