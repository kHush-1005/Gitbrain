"""
scripts/validate_chunks.py  [NEW — Week 2]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Quality-check a JSONL chunk file before using it in Week 3 embeddings.

    This script reads the JSONL file produced by export_chunks.py and checks:
        ✓ Required fields are present on every chunk
        ✓ Line numbers are valid (start_line > 0, end_line >= start_line)
        ✓ Chunk text is not empty
        ✓ No chunks are oversized (> 512 token estimate)
        ✓ Distribution of chunk types is reasonable
        ✓ No duplicate chunk IDs (file_path + chunk_name + start_line)

    If all checks pass, it prints a green "READY FOR WEEK 3" message.
    If issues are found, it describes them and suggests fixes.

USAGE:
    python scripts/validate_chunks.py data/chunks/tiangolo__fastapi_chunks.jsonl
    python scripts/validate_chunks.py data/chunks/psf__requests_chunks.jsonl
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import json
from collections import Counter

# ── Ensure project root is on path ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.file_utils import load_jsonl

# ─── Validation thresholds ────────────────────────────────────────────────────
MAX_TOKEN_ESTIMATE = 512     # chunks above this are "oversized"
MAX_CHAR_COUNT     = 2048    # hard char limit (512 tokens × 4 chars/token)
MIN_CHAR_COUNT     = 10      # chunks below this are suspiciously small
REQUIRED_FIELDS    = {
    "text", "file_path", "language", "chunk_type",
    "chunk_name", "start_line", "end_line", "char_count", "token_estimate"
}


# ─────────────────────────────────────────────────────────────────────────────
def validate_chunks(jsonl_path: str) -> bool:
    """
    Run all quality checks on a JSONL chunk file.

    Args:
        jsonl_path: Path to the .jsonl file to validate.

    Returns:
        True if all critical checks pass, False if critical issues found.
    """
    print()
    print("=" * 70)
    print("  GitBrain — Week 2 Chunk Validator")
    print("=" * 70)
    print(f"  File: {jsonl_path}\n")

    # ── Load the file ──────────────────────────────────────────────────────────
    try:
        chunks = load_jsonl(jsonl_path)
    except FileNotFoundError as e:
        print(f"  ✗ ERROR: {e}")
        print("  Run export_chunks.py first to generate the file.")
        return False

    if not chunks:
        print("  ✗ ERROR: JSONL file is empty — no chunks found.")
        return False

    total = len(chunks)
    print(f"  Total chunks loaded: {total}\n")

    # ── Track issues ──────────────────────────────────────────────────────────
    critical_issues = []    # these block Week 3
    warnings        = []    # these are notable but not blocking

    # ── CHECK 1: Required fields ───────────────────────────────────────────────
    print("  ── CHECK 1: Required Fields ─────────────────────────────────────")
    missing_field_counts = Counter()
    chunks_with_missing  = 0

    for chunk in chunks:
        missing = REQUIRED_FIELDS - set(chunk.keys())
        if missing:
            chunks_with_missing += 1
            for field in missing:
                missing_field_counts[field] += 1

    if chunks_with_missing == 0:
        print(f"  ✓ All {total} chunks have all required fields")
    else:
        msg = f"{chunks_with_missing} chunks are missing required fields"
        critical_issues.append(msg)
        print(f"  ✗ {msg}")
        for field, count in missing_field_counts.most_common():
            print(f"      Missing '{field}': {count} chunks")

    # ── CHECK 2: Empty text ────────────────────────────────────────────────────
    print("\n  ── CHECK 2: Empty Text ──────────────────────────────────────────")
    empty_text = [c for c in chunks if not c.get("text", "").strip()]
    if not empty_text:
        print(f"  ✓ All chunks have non-empty text")
    else:
        msg = f"{len(empty_text)} chunks have empty text"
        critical_issues.append(msg)
        print(f"  ✗ {msg}")
        for c in empty_text[:3]:
            print(f"      {c.get('file_path','?')} → {c.get('chunk_name','?')}")

    # ── CHECK 3: Line number validity ─────────────────────────────────────────
    print("\n  ── CHECK 3: Line Number Validity ────────────────────────────────")
    bad_lines = []
    for c in chunks:
        sl = c.get("start_line", 0)
        el = c.get("end_line",   0)
        if sl <= 0 or el <= 0 or el < sl:
            bad_lines.append(c)

    if not bad_lines:
        print(f"  ✓ All line numbers are valid (start_line ≤ end_line, both > 0)")
    else:
        msg = f"{len(bad_lines)} chunks have invalid line numbers"
        critical_issues.append(msg)
        print(f"  ✗ {msg}")
        for c in bad_lines[:3]:
            print(
                f"      {c.get('file_path','?')} → {c.get('chunk_name','?')} "
                f"(start={c.get('start_line')}, end={c.get('end_line')})"
            )

    # ── CHECK 4: Chunk size ────────────────────────────────────────────────────
    print("\n  ── CHECK 4: Chunk Size (Token Estimates) ────────────────────────")
    oversized   = [c for c in chunks if c.get("token_estimate", 0) > MAX_TOKEN_ESTIMATE]
    undersized  = [c for c in chunks if c.get("char_count",     0) < MIN_CHAR_COUNT]
    token_vals  = [c.get("token_estimate", 0) for c in chunks]
    char_vals   = [c.get("char_count",     0) for c in chunks]

    avg_tokens  = sum(token_vals) / len(token_vals)
    avg_chars   = sum(char_vals)  / len(char_vals)
    max_tokens  = max(token_vals)
    max_chars   = max(char_vals)

    print(f"    Average size   : {avg_chars:.0f} chars  (~{avg_tokens:.0f} tokens)")
    print(f"    Largest chunk  : {max_chars} chars  (~{max_tokens} tokens)")
    print(f"    Target limit   : {MAX_CHAR_COUNT} chars  ({MAX_TOKEN_ESTIMATE} tokens)")

    if not oversized:
        print(f"  ✓ All chunks are within the {MAX_TOKEN_ESTIMATE}-token limit")
    else:
        msg = (f"{len(oversized)} chunks exceed {MAX_TOKEN_ESTIMATE} token estimate "
               f"({len(oversized)/total*100:.1f}% of total)")
        warnings.append(msg)
        print(f"  ⚠  {msg}")
        print(f"     Largest {min(5, len(oversized))} oversized chunks:")
        for c in sorted(oversized, key=lambda x: x.get("token_estimate", 0), reverse=True)[:5]:
            print(
                f"      [{c.get('token_estimate',0):>5} tok] "
                f"{c.get('file_path','?')} → {c.get('chunk_name','?')}"
            )

    if undersized:
        msg = f"{len(undersized)} chunks are very small (< {MIN_CHAR_COUNT} chars)"
        warnings.append(msg)
        print(f"  ⚠  {msg} (these may be noise)")

    # ── CHECK 5: Duplicate detection ──────────────────────────────────────────
    print("\n  ── CHECK 5: Duplicate Chunk IDs ─────────────────────────────────")
    seen_ids  = Counter()
    for c in chunks:
        uid = f"{c.get('file_path','')}::{c.get('chunk_name','')}::{c.get('start_line','')}"
        seen_ids[uid] += 1

    duplicates = {k: v for k, v in seen_ids.items() if v > 1}
    if not duplicates:
        print(f"  ✓ No duplicate chunk IDs found")
    else:
        msg = f"{len(duplicates)} duplicate chunk IDs detected"
        warnings.append(msg)
        print(f"  ⚠  {msg}")
        for uid, count in list(duplicates.items())[:3]:
            print(f"      ({count}×) {uid[:70]}")

    # ── CHECK 6: Chunk type distribution ──────────────────────────────────────
    print("\n  ── CHECK 6: Chunk Type Distribution ─────────────────────────────")
    type_counts = Counter(c.get("chunk_type", "unknown") for c in chunks)
    for ctype, count in type_counts.most_common():
        pct = count / total * 100
        bar = "█" * min(int(pct / 2), 35)
        print(f"    {ctype:<22} {count:>5}  ({pct:5.1f}%)  {bar}")

    # ── CHECK 7: Language distribution ────────────────────────────────────────
    print("\n  ── CHECK 7: Language Distribution ───────────────────────────────")
    lang_counts = Counter(c.get("language", "unknown") for c in chunks)
    for lang, count in lang_counts.most_common(10):
        pct = count / total * 100
        bar = "█" * min(int(pct / 2), 35)
        print(f"    {lang:<22} {count:>5}  ({pct:5.1f}%)  {bar}")

    # ── CHECK 8: File path coverage ───────────────────────────────────────────
    print("\n  ── CHECK 8: File Coverage ───────────────────────────────────────")
    unique_files = len(set(c.get("file_path", "") for c in chunks))
    print(f"    Unique files represented : {unique_files}")
    avg_per_file = total / unique_files if unique_files else 0
    print(f"    Average chunks per file  : {avg_per_file:.1f}")

    # ── PRINT TOP 5 LARGEST CHUNKS ────────────────────────────────────────────
    print("\n  ── Top 5 Largest Chunks ─────────────────────────────────────────")
    top5 = sorted(chunks, key=lambda c: c.get("char_count", 0), reverse=True)[:5]
    for c in top5:
        print(
            f"    [{c.get('char_count',0):>6} chars | ~{c.get('token_estimate',0):>4} tok] "
            f"{c.get('file_path','?'):<40} → {c.get('chunk_name','?')}"
        )

    # ── FINAL VERDICT ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if critical_issues:
        print("  ✗ CRITICAL ISSUES FOUND — Fix before proceeding to Week 3:")
        for issue in critical_issues:
            print(f"    • {issue}")
        print()
        if warnings:
            print("  ⚠  Warnings (non-blocking):")
            for w in warnings:
                print(f"    • {w}")
        print("=" * 70)
        print()
        return False
    else:
        print("  ✓ ALL CRITICAL CHECKS PASSED")
        if warnings:
            print()
            print("  ⚠  Warnings (non-blocking, review before Week 3):")
            for w in warnings:
                print(f"    • {w}")
        print()
        print("  ✓ CHUNKS ARE READY FOR WEEK 3 EMBEDDING")
        print()
        print("  Next steps (Week 3):")
        print("    pip install chromadb sentence-transformers")
        print("    Build core/embedder.py    → embed chunks with all-MiniLM-L6-v2")
        print("    Build core/vector_store.py → store vectors in ChromaDB")
        print("=" * 70)
        print()
        return True


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print()
        print("Usage:   python scripts/validate_chunks.py <path_to_chunks.jsonl>")
        print()
        print("Example: python scripts/validate_chunks.py data/chunks/tiangolo__fastapi_chunks.jsonl")
        print()
        sys.exit(1)

    jsonl_path = sys.argv[1]
    success    = validate_chunks(jsonl_path)
    sys.exit(0 if success else 1)
