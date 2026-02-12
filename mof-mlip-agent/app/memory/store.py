"""
Lightweight JSONL-based persistent memory for past agent runs.

The store supports three main operations:
  - append(record)         : add a new run summary as a JSON object.
  - retrieve(query, k)     : fetch top-k "similar" records by word overlap.
  - format_context(records): turn records into a compact prompt string.
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _tokenize(text: str) -> List[str]:
    """Basic alphanumeric tokenisation, lowercased and filtered by length."""
    return [t.lower() for t in _WORD_RE.findall(text or "") if len(t) >= 2]


class MemoryStore:
    """
    Persistent memory for past runs.

    Data is stored in a JSONL file at `path`, where each line is a single
    JSON record describing one run. Retrieval is based on simple keyword
    overlap, which is cheap and has no external dependencies.
    """

    def __init__(self, path: str, max_items: int = 50):
        """
        Parameters
        ----------
        path:
            File path of the JSONL memory store.
        max_items:
            Soft cap on how many records to keep on disk.
        """
        self.path = path
        self.max_items = max_items
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Ensure file exists
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8"):
                pass

    def append(self, record: Dict[str, Any]) -> None:
        """
        Append a new memory record as a JSON line and trim if needed.
        """
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Soft trim if huge (optional; simple approach)
        self._trim_if_needed()

    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all records currently stored in the JSONL file.
        """
        items: List[Dict[str, Any]] = []
        if not os.path.exists(self.path):
            return items
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve up to k memory records that are "similar" to the query.

        Similarity is based on overlapping word tokens between the query
        and selected fields of each record.
        """
        items = self.load_all()
        if not items:
            return []

        q_tokens = set(_tokenize(query))
        if not q_tokens:
            # If the query is empty or has no tokens, fall back to the
            # most recent k runs.
            return items[-k:]

        scored: List[Tuple[int, Dict[str, Any]]] = []
        for it in items:
            text = " ".join([
                str(it.get("query_original", "")),
                str(it.get("query_canonical", "")),
                str(it.get("mof_name", "")),
                str(it.get("task_type", "")),
                str(it.get("verdict_status", "")),
            ])
            it_tokens = set(_tokenize(text))
            score = len(q_tokens.intersection(it_tokens))
            if score > 0:
                scored.append((score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [it for _, it in scored[:k]]
        if len(top) < k:
            # Backfill with most recent runs, avoiding duplicates (by exp_id).
            recent = items[-k:]
            seen = set(x.get("exp_id") for x in top)
            for r in reversed(recent):
                if r.get("exp_id") not in seen:
                    top.append(r)
                    seen.add(r.get("exp_id"))
                if len(top) >= k:
                    break
        return top[:k]

    def format_context(self, records: List[Dict[str, Any]]) -> str:
        """
        Turn retrieved memory records into a compact context string for prompts.
        """
        if not records:
            return "(no prior memory)"
        lines: List[str] = []
        for r in records:
            lines.append(
                "PAST_RUN:"
                f" exp_id={r.get('exp_id','')};"
                f" mof={r.get('mof_name','')};"
                f" task={r.get('task_type','')};"
                f" verdict={r.get('verdict_status','')}\n"
                f"  original={r.get('query_original','')}\n"
                f"  canonical={r.get('query_canonical','')}\n"
            )
        return "\n".join(lines).strip()

    def _trim_if_needed(self) -> None:
        """
        Keep only the last `max_items` records in the JSONL file.
        """
        if self.max_items <= 0:
            return
        items = self.load_all()
        if len(items) <= self.max_items:
            return
        items = items[-self.max_items :]
        with open(self.path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
