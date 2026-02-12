from typing import Tuple, List, Dict
from pathlib import Path
import os
import re

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None


# Default folder where you can drop local PDFs for RAG.
# You can override this with the LOCAL_PDF_DIR env var.
_DEFAULT_PDF_DIR = Path(__file__).resolve().parents[2] / "local_pdfs"
LOCAL_PDF_DIR = Path(os.getenv("LOCAL_PDF_DIR", str(_DEFAULT_PDF_DIR)))

# Simple chunking / scoring settings
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_TOP_CHUNKS = 5


def _ensure_pdf_dir() -> None:
    """
    Make sure the local PDF directory exists so users can just
    drop files into it without manual setup.
    """
    try:
        LOCAL_PDF_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot create it, we just silently fall back to "no context".
        pass


def _extract_chunks_from_pdf(pdf_path: Path) -> List[Dict]:
    """Extract text chunks from a single PDF file."""
    chunks: List[Dict] = []

    if fitz is None:
        return chunks

    doc = fitz.open(pdf_path)
    try:
        for page_idx, page in enumerate(doc):
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                continue

            start = 0
            n = len(text)
            while start < n:
                end = min(start + CHUNK_SIZE, n)
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        {
                            "source": pdf_path.name,
                            "page": page_idx,
                            "text": chunk_text,
                        }
                    )
                if end >= n:
                    break
                start = max(0, end - CHUNK_OVERLAP)
    finally:
        doc.close()

    return chunks


def _score_chunk(query: str, text: str) -> float:
    """
    Very simple lexical overlap score between query and chunk.
    This avoids needing any embedding model.
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return 0.0

    text_tokens = re.findall(r"\w+", text.lower())
    if not text_tokens:
        return 0.0

    matches = sum(1 for t in query_tokens if t in text_tokens)
    return matches / float(len(query_tokens))


def local_rag_search(query: str) -> Tuple[str, list]:
    """
    Local PDF / notes RAG.

    Looks for PDFs in LOCAL_PDF_DIR (default: <repo_root>/local_pdfs),
    extracts text with PyMuPDF, picks the top-matching passages, and
    returns:
      - context_text: str (concatenated top passages)
      - refs: list of dicts with metadata (source, page, score)
    """
    _ensure_pdf_dir()

    if fitz is None or not LOCAL_PDF_DIR.exists():
        return "", []

    all_chunks: List[Dict] = []
    for pdf_path in LOCAL_PDF_DIR.glob("*.pdf"):
        try:
            all_chunks.extend(_extract_chunks_from_pdf(pdf_path))
        except Exception:
            # Skip unreadable PDFs
            continue

    if not all_chunks:
        return "", []

    for ch in all_chunks:
        ch["score"] = _score_chunk(query, ch["text"])

    # Filter out zero-score chunks
    scored = [ch for ch in all_chunks if ch.get("score", 0.0) > 0.0]
    if not scored:
        return "", []

    scored.sort(key=lambda c: c["score"], reverse=True)
    top = scored[:MAX_TOP_CHUNKS]

    context_parts: List[str] = []
    refs: List[Dict] = []

    for ch in top:
        src = ch["source"]
        page = int(ch["page"]) + 1  # 1-based for humans
        snippet = ch["text"]
        context_parts.append(f"[{src} p.{page}] {snippet}")
        refs.append(
            {
                "source": src,
                "page": page,
                "score": float(ch["score"]),
            }
        )

    context_text = "\n\n".join(context_parts)
    return context_text, refs
