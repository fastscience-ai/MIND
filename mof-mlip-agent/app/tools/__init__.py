"""
Public import surface for tool functions.

This lets other modules write:

    from app.tools import fetch_arxiv_docs, local_rag_search

without needing to know about the underlying module layout.
"""

from .arxiv_tool import fetch_arxiv_docs, docs_to_compact_text
from .local_rag import local_rag_search

__all__ = [
    "fetch_arxiv_docs",
    "docs_to_compact_text",
    "local_rag_search",
]
