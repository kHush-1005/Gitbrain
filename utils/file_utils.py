"""
utils/file_utils.py
─────────────────────────────────────────────────────────────────────────────
Utility functions for saving and loading chunk data to/from disk.
(Carried forward from Week 2 — no changes needed for Week 3)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug(f"Saved JSON: {path}")


def save_jsonl(chunks: list, path: str) -> int:
    ensure_dir(str(Path(path).parent))
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            written += 1
    logger.info(f"Saved {written} chunks to: {path}")
    return written


def load_jsonl(path: str) -> list:
    """
    Read a JSONL file and return list of dicts.
    Uses pathlib for Windows/Linux path compatibility.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}\n"
            "Run scripts/export_chunks.py first to generate it.\n"
            f"Expected location: {file_path.resolve()}"
        )

    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num} in {path}: {e}")

    logger.info(f"Loaded {len(chunks)} chunks from: {path}")
    return chunks


def sanitize_repo_name(owner: str, repo: str) -> str:
    safe_owner = owner.lower().replace("-", "_").replace(".", "_")
    safe_repo  = repo.lower().replace("-",  "_").replace(".", "_")
    return f"{safe_owner}__{safe_repo}"


def get_chunk_file_path(owner: str, repo: str, base_dir: str = "data/chunks") -> str:
    name = sanitize_repo_name(owner, repo)
    return str(Path(base_dir) / f"{name}_chunks.jsonl")
