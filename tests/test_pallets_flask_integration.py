"""
tests/test_pallets_flask_integration.py
─────────────────────────────────────────────────────────────────────────────
Smoke test for the existing pallets__flask repository export.

This test verifies that:
  - the expected JSONL chunk file exists
  - the file can be loaded as valid JSONL
  - the chunks contain required metadata fields
  - the repo name derived from the filename is correct

This is useful to confirm the project can work with the existing
pallets__flask dataset without needing a full live GitHub ingestion run.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.file_utils import load_jsonl
from scripts.build_vector_index import derive_repo_name_from_path

CHUNKS_PATH = ROOT / "data" / "chunks" / "pallets__flask_chunks.jsonl"


def test_pallets_flask_chunk_file_exists():
    assert CHUNKS_PATH.exists(), (
        f"Expected pallets__flask chunk file at {CHUNKS_PATH}, but it was missing."
    )


def test_pallets_flask_chunk_file_loads():
    chunks = load_jsonl(str(CHUNKS_PATH))
    assert len(chunks) > 0, "The pallets__flask chunk file should contain at least one chunk."

    sample = chunks[:10]
    for chunk in sample:
        assert "file_path" in chunk, "Each chunk must include file_path metadata."
        assert "text" in chunk, "Each chunk must include chunk text."
        assert "language" in chunk, "Each chunk must include a language label."
        assert "chunk_type" in chunk, "Each chunk must include a chunk_type label."


def test_pallets_flask_repo_name_is_derived_correctly():
    assert derive_repo_name_from_path(str(CHUNKS_PATH)) == "pallets__flask"
