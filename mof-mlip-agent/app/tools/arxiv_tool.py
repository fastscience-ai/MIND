"""
Thin wrappers around LangChain's arXiv loader.

We keep all arXiv-specific logic here so the main graph and chains only
depend on simple Python callables for document retrieval and formatting.
"""

from typing import List
from langchain_community.document_loaders import ArxivLoader


def fetch_arxiv_docs(query: str, max_docs: int = 6):
    """
    Load arXiv documents for a free-text query.

    Parameters
    ----------
    query:
        Search string to pass to arXiv.
    max_docs:
        Maximum number of documents to retrieve.

    Returns
    -------
    List[Document]
        LangChain `Document` objects with metadata and page_content.
    """
    loader = ArxivLoader(query=query, load_max_docs=max_docs)
    docs = loader.load()
    return docs


def docs_to_compact_text(docs) -> str:
    """
    Compact arXiv documents into a single string for LLM novelty judgement.

    Each document is rendered as:
        TITLE: <title>
        ID: <entry_id>
        SUMMARY:
        <abstract text>
        ---
    """
    chunks: List[str] = []
    for d in docs:
        title = d.metadata.get("Title", "") or d.metadata.get("title", "")
        entry_id = d.metadata.get("Entry ID", "") or d.metadata.get("entry_id", "")
        summary = (d.page_content or "").strip()
        chunks.append(
            f"TITLE: {title}\nID: {entry_id}\nSUMMARY:\n{summary}\n---"
        )
    return "\n".join(chunks).strip()
