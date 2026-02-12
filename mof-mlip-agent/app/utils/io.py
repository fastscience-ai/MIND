"""
Small IO helpers used across the agent.

We keep these as thin wrappers so that all file-system interactions go
through a tiny, well-tested surface.
"""

import json
import os
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Dict[str, Any]) -> None:
    """
    Serialize a Python dict as pretty-printed JSON to disk.

    Parameters
    ----------
    path:
        Target file path.
    data:
        JSON-serializable object (typically a dict).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
