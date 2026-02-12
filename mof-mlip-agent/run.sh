#!/usr/bin/env bash
# Set your key in the environment before running: export OPENAI_API_KEY="your-key"
# Do not commit API keys to the repository.
export ARXIV_MAX_DOCS=2
export MEMORY_RETRIEVE_K=1
export FAST_MODE=1
python -m app.run "$@"
