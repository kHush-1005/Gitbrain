"""
core/code_chunker.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Split source code files into small, semantically meaningful "chunks."

WHY CHUNKING MATTERS:
    A 500-line Python file cannot be given to an LLM as one block —
    it's too big, and most of it is irrelevant to any single question.

    Instead, we split it into individual functions and classes.
    When a user asks "how does login work?", we retrieve only the
    `login()` function chunk — not the entire file.

CHUNKING STRATEGIES USED:

    1. AST-BASED (Python only):
       Uses Python's built-in `ast` module to parse the file into a syntax
       tree and extract exact function and class definitions with their
       line numbers. This is the most accurate method.

    2. FALLBACK — SLIDING WINDOW:
       For non-Python files (JS, Go, Markdown, YAML, etc.), we split
       the file into overlapping windows of fixed line count.
       Each window becomes one chunk.
       This is simpler and language-agnostic.

OUTPUT FORMAT (each chunk is a dict):
    {
        "text":        "def login(user, password):\n    ...",  # the actual code
        "file_path":   "src/auth/login.py",
        "language":    "python",
        "chunk_type":  "function",          # "function", "class", "method", "window"
        "chunk_name":  "login",             # function/class name, or "window_N"
        "start_line":  42,                  # 1-based line number
        "end_line":    58,
    }
─────────────────────────────────────────────────────────────────────────────
"""

import ast
import logging

logger = logging.getLogger(__name__)

# ─── Sliding window configuration ─────────────────────────────────────────────
WINDOW_SIZE    = 60   # lines per window chunk
WINDOW_OVERLAP = 10   # lines of overlap between consecutive windows
MIN_CHUNK_LINES = 3   # ignore chunks smaller than this (likely just comments)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def chunk_file(content: str, file_path: str, language: str) -> list[dict]:
    """
    Split a source file into chunks using the best available strategy.

    For Python files: uses AST-based extraction (functions + classes).
    For everything else: uses sliding window fallback.

    Args:
        content:   Raw file content as a string.
        file_path: Path to the file relative to repo root.
        language:  Detected language string (e.g. "python", "javascript").

    Returns:
        A list of chunk dicts (see module docstring for format).
        Returns an empty list if the file is empty or cannot be parsed.
    """
    if not content or not content.strip():
        return []

    if language == "python":
        chunks = chunk_python_file(content, file_path)
        # If AST parsing failed or returned nothing, fall back to sliding window
        if not chunks:
            logger.debug(
                f"AST chunking returned 0 chunks for {file_path}, "
                "falling back to sliding window."
            )
            chunks = fallback_chunk_text(content, file_path, language)
    else:
        chunks = fallback_chunk_text(content, file_path, language)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: AST-BASED PYTHON CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
def chunk_python_file(content: str, file_path: str) -> list[dict]:
    """
    Parse a Python file using the ast module and extract:
        - Top-level functions  (chunk_type = "function")
        - Top-level classes    (chunk_type = "class")
        - Methods inside classes (chunk_type = "method")

    Line numbers are 1-based (matching what editors show).

    Args:
        content:   Raw Python source code as a string.
        file_path: Path to the file (used in chunk metadata).

    Returns:
        A list of chunk dicts, or [] if the file cannot be parsed.
    """
    # Split content into lines so we can extract exact line ranges
    lines = content.splitlines()

    # Try to parse the file with ast
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logger.debug(f"AST parse failed for {file_path}: {e}")
        return []
    except Exception as e:
        logger.debug(f"Unexpected AST error for {file_path}: {e}")
        return []

    chunks = []

    # Walk all top-level nodes in the module
    for node in ast.iter_child_nodes(tree):

        # ── Top-level function (def my_function(...): ) ────────────────────────
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _extract_function_chunk(node, lines, file_path, language="python")
            if chunk:
                chunks.append(chunk)

        # ── Top-level class (class MyClass: ) ─────────────────────────────────
        elif isinstance(node, ast.ClassDef):
            # First, add the whole class as a single chunk (for class-level questions)
            class_chunk = _extract_class_chunk(node, lines, file_path)
            if class_chunk:
                chunks.append(class_chunk)

            # Then, add each method as its own chunk (for method-level questions)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = _extract_function_chunk(
                        child, lines, file_path,
                        language="python",
                        chunk_type="method",
                        parent_class=node.name,
                    )
                    if method_chunk:
                        chunks.append(method_chunk)

    return chunks


def _extract_function_chunk(
    node:         ast.FunctionDef,
    lines:        list[str],
    file_path:    str,
    language:     str = "python",
    chunk_type:   str = "function",
    parent_class: str = "",
) -> dict | None:
    """
    Build a chunk dict from an AST FunctionDef (or AsyncFunctionDef) node.

    Args:
        node:         The AST node for this function.
        lines:        All lines of the file as a list (0-indexed internally).
        file_path:    File path for metadata.
        language:     Always "python" here.
        chunk_type:   "function" or "method".
        parent_class: Class name if this is a method, else "".

    Returns:
        A chunk dict, or None if the chunk is too small to be useful.
    """
    start_line = node.lineno          # ast line numbers are 1-based
    end_line   = node.end_lineno      # also 1-based

    if (end_line - start_line) < MIN_CHUNK_LINES - 1:
        return None  # skip trivially small functions

    # Extract the actual source lines (convert to 0-based index)
    chunk_lines = lines[start_line - 1 : end_line]
    text = "\n".join(chunk_lines)

    # Build a display name: "MyClass.my_method" or just "my_function"
    if parent_class:
        display_name = f"{parent_class}.{node.name}"
    else:
        display_name = node.name

    return {
        "text":        text,
        "file_path":   file_path,
        "language":    language,
        "chunk_type":  chunk_type,
        "chunk_name":  display_name,
        "start_line":  start_line,
        "end_line":    end_line,
    }


def _extract_class_chunk(
    node:      ast.ClassDef,
    lines:     list[str],
    file_path: str,
) -> dict | None:
    """
    Build a chunk dict for a whole class definition.

    The class chunk contains the class header plus its docstring
    (if any) but excludes method bodies (those are separate chunks).

    For simplicity in Week 1, we include the ENTIRE class body.
    In later weeks, very large classes can be trimmed to just the
    header + docstring to reduce chunk size.

    Args:
        node:      The AST node for this class.
        lines:     All lines of the file as a list.
        file_path: File path for metadata.

    Returns:
        A chunk dict, or None if too small.
    """
    start_line = node.lineno
    end_line   = node.end_lineno

    if (end_line - start_line) < MIN_CHUNK_LINES - 1:
        return None

    chunk_lines = lines[start_line - 1 : end_line]
    text = "\n".join(chunk_lines)

    return {
        "text":        text,
        "file_path":   file_path,
        "language":    "python",
        "chunk_type":  "class",
        "chunk_name":  node.name,
        "start_line":  start_line,
        "end_line":    end_line,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: SLIDING WINDOW FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def fallback_chunk_text(content: str, file_path: str, language: str) -> list[dict]:
    """
    Split any text file into overlapping windows of fixed line count.

    This is used for:
        - JavaScript, TypeScript, Go, Rust, Java, etc.
        - Markdown, YAML, JSON, plain text
        - Any file that AST chunking cannot handle

    Window configuration (defined at top of this file):
        WINDOW_SIZE    = 60 lines per chunk
        WINDOW_OVERLAP = 10 lines of overlap between chunks

    The overlap ensures that code at the boundary between two windows
    appears in both chunks, so context is never completely cut off.

    Args:
        content:   Raw file content as a string.
        file_path: Path to the file.
        language:  Detected language string.

    Returns:
        A list of window chunk dicts.
    """
    lines = content.splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return []

    # If the file is small enough to fit in one window, just return it as-is
    if total_lines <= WINDOW_SIZE:
        text = "\n".join(lines).strip()
        if len(text) < 10:  # skip empty or near-empty files
            return []
        return [{
            "text":        text,
            "file_path":   file_path,
            "language":    language,
            "chunk_type":  "window",
            "chunk_name":  "window_1",
            "start_line":  1,
            "end_line":    total_lines,
        }]

    chunks = []
    window_number = 1
    step = WINDOW_SIZE - WINDOW_OVERLAP  # how far to advance each step

    start = 0  # 0-based index into lines list
    while start < total_lines:
        end = min(start + WINDOW_SIZE, total_lines)

        # Extract the lines for this window
        window_lines = lines[start:end]
        text = "\n".join(window_lines).strip()

        # Skip windows that are too small (e.g. trailing whitespace at end of file)
        if len(text.splitlines()) >= MIN_CHUNK_LINES:
            chunks.append({
                "text":        text,
                "file_path":   file_path,
                "language":    language,
                "chunk_type":  "window",
                "chunk_name":  f"window_{window_number}",
                "start_line":  start + 1,      # convert to 1-based
                "end_line":    end,             # already 1-based (end is exclusive in slice)
            })
            window_number += 1

        # Move the window forward
        start += step

        # If we've reached the last window, stop
        if end == total_lines:
            break

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Chunk multiple files at once
# ─────────────────────────────────────────────────────────────────────────────
def chunk_all_files(files: list[dict]) -> list[dict]:
    """
    Chunk a list of file dicts (output of fetch_repository_files).

    Args:
        files: List of dicts with keys: path, content, language, size.

    Returns:
        A flat list of all chunks from all files.
    """
    all_chunks = []
    total_files = len(files)

    for i, file_info in enumerate(files, start=1):
        path     = file_info["path"]
        content  = file_info["content"]
        language = file_info["language"]

        file_chunks = chunk_file(content, path, language)
        all_chunks.extend(file_chunks)

        if i % 20 == 0 or i == total_files:
            print(
                f"  [Chunker] [{i:>4}/{total_files}] {path[:60]} "
                f"→ {len(file_chunks)} chunks"
            )

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE_PYTHON = '''
def add(a, b):
    """Add two numbers and return the result."""
    return a + b


def subtract(a, b):
    """Subtract b from a."""
    return a - b


class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def multiply(self, x, y):
        """Multiply x and y."""
        result = x * y
        self.history.append(("multiply", x, y, result))
        return result

    def divide(self, x, y):
        """Divide x by y. Raises ZeroDivisionError if y is 0."""
        if y == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return x / y
'''

    print("Testing code_chunker.py with sample Python code\n" + "─" * 50)
    chunks = chunk_file(SAMPLE_PYTHON, "calculator.py", "python")
    for chunk in chunks:
        print(f"  Chunk    : {chunk['chunk_name']}")
        print(f"  Type     : {chunk['chunk_type']}")
        print(f"  Lines    : {chunk['start_line']} – {chunk['end_line']}")
        print(f"  Preview  : {chunk['text'][:60].strip()!r}")
        print()

    print(f"Total chunks from sample Python file: {len(chunks)}")
    print()

    SAMPLE_JS = "\n".join([f"// Line {i}" for i in range(1, 100)])
    print("Testing fallback_chunk_text with 99-line text file")
    js_chunks = chunk_file(SAMPLE_JS, "sample.js", "javascript")
    print(f"Total window chunks: {len(js_chunks)}")
    for c in js_chunks:
        print(f"  {c['chunk_name']}: lines {c['start_line']}–{c['end_line']}")
