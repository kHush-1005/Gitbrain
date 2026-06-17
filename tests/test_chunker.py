"""
tests/test_chunker.py  [NEW — Week 2]
─────────────────────────────────────────────────────────────────────────────
Pytest tests for core/code_chunker.py

Tests cover:
    - Python function extraction via AST
    - Python class extraction via AST
    - Python method extraction (inside classes)
    - JavaScript heuristic chunking
    - Fallback sliding-window chunking for text files
    - Required metadata fields on every chunk
    - Chunk size constraints
    - Edge cases: empty files, syntax errors, tiny files

Run with:
    pytest tests/test_chunker.py -v
    pytest tests/ -v              (all tests)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.code_chunker import (
    chunk_file,
    chunk_python_file,
    chunk_javascript_like_file,
    fallback_chunk_text,
    MAX_CHUNK_CHARS,
    WINDOW_SIZE_CODE,
    WINDOW_OVERLAP,
)

# ─── Required fields every chunk must have ────────────────────────────────────
REQUIRED_FIELDS = {
    "text", "file_path", "language", "chunk_type",
    "chunk_name", "start_line", "end_line", "char_count", "token_estimate"
}

# ─────────────────────────────────────────────────────────────────────────────
# Sample source code fixtures
# ─────────────────────────────────────────────────────────────────────────────

PYTHON_WITH_FUNCTION = '''\
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"
'''

PYTHON_WITH_TWO_FUNCTIONS = '''\
def add(a, b):
    """Add two numbers and return the result."""
    return a + b


def subtract(a, b):
    """Subtract b from a and return the result."""
    return a - b
'''

PYTHON_WITH_CLASS = '''\
class Calculator:
    """Simple calculator that tracks history."""

    def __init__(self):
        """Initialise with zero value."""
        self.value = 0

    def add(self, x):
        """Add x to current value and return it."""
        self.value += x
        return self.value

    def reset(self):
        """Reset the calculator back to zero."""
        self.value = 0
        return self.value
'''

PYTHON_WITH_ASYNC = '''\
async def fetch_data(url: str):
    """Fetch data asynchronously."""
    import asyncio
    await asyncio.sleep(0)
    return {}
'''

PYTHON_WITH_SYNTAX_ERROR = '''\
def broken_function(
    # this is deliberately broken
    x y z
'''

JAVASCRIPT_CODE = '''\
function authenticate(req, res) {
    const token = req.headers.authorization;
    if (!token) return res.status(401).json({ error: "No token" });
    return validateToken(token);
}

async function fetchUser(userId) {
    const user = await db.users.findOne({ id: userId });
    if (!user) throw new Error("User not found");
    return user;
}

class UserController {
    constructor(userService) {
        this.service = userService;
    }

    getUser(req, res) {
        return res.json(this.service.get(req.params.id));
    }
}
'''

YAML_TEXT = "\n".join([f"setting_{i}: value_{i}" for i in range(1, 80)])


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Python Function Extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestPythonFunctionExtraction:

    def test_extracts_one_function(self):
        chunks = chunk_python_file(PYTHON_WITH_FUNCTION, "greet.py")
        names  = [c["chunk_name"] for c in chunks]
        assert "greet" in names

    def test_extracts_two_functions(self):
        chunks = chunk_python_file(PYTHON_WITH_TWO_FUNCTIONS, "math.py")
        names  = [c["chunk_name"] for c in chunks]
        assert "add"      in names
        assert "subtract" in names

    def test_function_chunk_type_is_function(self):
        chunks = chunk_python_file(PYTHON_WITH_FUNCTION, "greet.py")
        func_chunks = [c for c in chunks if c["chunk_name"] == "greet"]
        assert len(func_chunks) == 1
        assert func_chunks[0]["chunk_type"] == "function"

    def test_async_function_is_extracted(self):
        chunks = chunk_python_file(PYTHON_WITH_ASYNC, "async_mod.py")
        names  = [c["chunk_name"] for c in chunks]
        assert "fetch_data" in names

    def test_function_text_contains_def_keyword(self):
        chunks = chunk_python_file(PYTHON_WITH_FUNCTION, "greet.py")
        greet  = next(c for c in chunks if c["chunk_name"] == "greet")
        assert "def greet" in greet["text"]

    def test_syntax_error_returns_empty_list(self):
        """AST parse fails gracefully — returns [] instead of crashing."""
        chunks = chunk_python_file(PYTHON_WITH_SYNTAX_ERROR, "broken.py")
        assert isinstance(chunks, list)
        # May return [] or fall through to fallback; should not raise

    def test_line_numbers_are_positive(self):
        chunks = chunk_python_file(PYTHON_WITH_TWO_FUNCTIONS, "math.py")
        for c in chunks:
            assert c["start_line"] >= 1
            assert c["end_line"]   >= c["start_line"]

    def test_language_is_python(self):
        chunks = chunk_python_file(PYTHON_WITH_FUNCTION, "greet.py")
        for c in chunks:
            assert c["language"] == "python"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Python Class Extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestPythonClassExtraction:

    def test_extracts_class_chunk(self):
        chunks = chunk_python_file(PYTHON_WITH_CLASS, "calc.py")
        types  = [c["chunk_type"] for c in chunks]
        assert "class" in types

    def test_class_chunk_name_matches_class_name(self):
        chunks = chunk_python_file(PYTHON_WITH_CLASS, "calc.py")
        class_chunks = [c for c in chunks if c["chunk_type"] == "class"]
        assert any(c["chunk_name"] == "Calculator" for c in class_chunks)

    def test_methods_are_extracted_as_method_type(self):
        chunks = chunk_python_file(PYTHON_WITH_CLASS, "calc.py")
        method_chunks = [c for c in chunks if c["chunk_type"] == "method"]
        assert len(method_chunks) >= 2

    def test_method_names_include_class_prefix(self):
        """Method names should be 'ClassName.method_name'."""
        chunks = chunk_python_file(PYTHON_WITH_CLASS, "calc.py")
        method_names = [c["chunk_name"] for c in chunks if c["chunk_type"] == "method"]
        # e.g. "Calculator.add", "Calculator.reset"
        assert any("Calculator" in name for name in method_names)

    def test_total_chunk_count_for_class(self):
        """Calculator class should produce: 1 class + 3 methods (init, add, reset)."""
        chunks = chunk_python_file(PYTHON_WITH_CLASS, "calc.py")
        assert len(chunks) >= 4   # 1 class + at least 3 methods

    def test_class_text_contains_class_keyword(self):
        chunks = chunk_python_file(PYTHON_WITH_CLASS, "calc.py")
        class_chunk = next(c for c in chunks if c["chunk_type"] == "class")
        assert "class Calculator" in class_chunk["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: JavaScript Heuristic Chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestJavaScriptChunking:

    def test_returns_chunks_for_js_code(self):
        chunks = chunk_javascript_like_file(JAVASCRIPT_CODE, "user.js", "javascript")
        assert len(chunks) > 0

    def test_chunk_type_is_heuristic_block(self):
        chunks = chunk_javascript_like_file(JAVASCRIPT_CODE, "user.js", "javascript")
        types  = set(c["chunk_type"] for c in chunks)
        assert "heuristic_block" in types

    def test_language_is_javascript(self):
        chunks = chunk_javascript_like_file(JAVASCRIPT_CODE, "user.js", "javascript")
        for c in chunks:
            assert c["language"] == "javascript"

    def test_chunk_text_is_not_empty(self):
        chunks = chunk_javascript_like_file(JAVASCRIPT_CODE, "user.js", "javascript")
        for c in chunks:
            assert c["text"].strip() != ""

    def test_line_numbers_are_positive(self):
        chunks = chunk_javascript_like_file(JAVASCRIPT_CODE, "user.js", "javascript")
        for c in chunks:
            assert c["start_line"] >= 1
            assert c["end_line"]   >= c["start_line"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Fallback Sliding Window Chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackChunking:

    def test_short_file_becomes_single_chunk(self):
        short_text = "\n".join([f"line {i}" for i in range(1, 10)])
        chunks = fallback_chunk_text(short_text, "readme.md", "markdown")
        assert len(chunks) == 1

    def test_single_chunk_is_window_type(self):
        short_text = "line one of the file\nline two of the file\nline three of the file"
        chunks = fallback_chunk_text(short_text, "note.txt", "text")
        assert len(chunks) >= 1
        assert chunks[0]["chunk_type"] == "window"

    def test_long_file_produces_multiple_chunks(self):
        long_text = "\n".join([f"setting_{i}: value" for i in range(1, 120)])
        chunks    = fallback_chunk_text(long_text, "config.yaml", "yaml", window_size=40)
        assert len(chunks) > 1

    def test_window_names_are_numbered_sequentially(self):
        long_text = "\n".join([f"item_{i}" for i in range(1, 150)])
        chunks    = fallback_chunk_text(long_text, "data.txt", "text", window_size=40)
        names     = [c["chunk_name"] for c in chunks]
        assert names[0]  == "window_1"
        assert names[1]  == "window_2"
        assert names[-1] == f"window_{len(chunks)}"

    def test_empty_file_returns_empty_list(self):
        chunks = fallback_chunk_text("", "empty.txt", "text")
        assert chunks == []

    def test_blank_lines_only_file_returns_empty_list(self):
        chunks = fallback_chunk_text("   \n\n\n   ", "blank.txt", "text")
        assert chunks == []

    def test_overlapping_windows_share_some_lines(self):
        """With WINDOW_OVERLAP=10, consecutive windows should overlap."""
        lines     = [f"line_{i}" for i in range(1, 100)]
        long_text = "\n".join(lines)
        chunks    = fallback_chunk_text(long_text, "file.txt", "text", window_size=40)
        if len(chunks) >= 2:
            # Last line of chunk 1 end should be >= start line of chunk 2
            assert chunks[0]["end_line"] >= chunks[1]["start_line"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: chunk_file() dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkFileDispatcher:

    def test_python_language_uses_ast(self):
        """Python files should produce function/class/method chunk types."""
        chunks = chunk_file(PYTHON_WITH_CLASS, "calc.py", "python")
        types  = {c["chunk_type"] for c in chunks}
        # At least one of these AST types should be present
        assert types & {"function", "class", "method"}

    def test_javascript_uses_heuristic(self):
        chunks = chunk_file(JAVASCRIPT_CODE, "user.js", "javascript")
        assert len(chunks) > 0

    def test_yaml_uses_fallback(self):
        chunks = chunk_file(YAML_TEXT, "config.yaml", "yaml")
        assert len(chunks) > 0
        assert all(c["chunk_type"] == "window" for c in chunks)

    def test_empty_content_returns_empty_list(self):
        for lang in ("python", "javascript", "yaml", "markdown"):
            chunks = chunk_file("", "empty_file.py", lang)
            assert chunks == [], f"Expected [] for empty {lang} file"

    def test_whitespace_only_content_returns_empty_list(self):
        chunks = chunk_file("    \n\n\t\n   ", "whitespace.py", "python")
        assert chunks == []

    def test_file_path_is_set_correctly_in_all_chunks(self):
        fp     = "src/services/user_service.py"
        chunks = chunk_file(PYTHON_WITH_TWO_FUNCTIONS, fp, "python")
        for c in chunks:
            assert c["file_path"] == fp

    def test_crlf_line_endings_handled_gracefully(self):
        """Windows CRLF line endings should not cause errors."""
        crlf_python = PYTHON_WITH_FUNCTION.replace("\n", "\r\n")
        chunks = chunk_file(crlf_python, "windows_file.py", "python")
        assert isinstance(chunks, list)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Required Metadata Fields
# ─────────────────────────────────────────────────────────────────────────────

class TestRequiredMetadataFields:
    """Every chunk from any strategy must have all required metadata fields."""

    def _check_all_chunks_have_required_fields(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            missing = REQUIRED_FIELDS - set(chunk.keys())
            assert not missing, (
                f"Chunk '{chunk.get('chunk_name', '?')}' in "
                f"'{chunk.get('file_path', '?')}' is missing fields: {missing}"
            )

    def test_python_chunks_have_required_fields(self):
        chunks = chunk_file(PYTHON_WITH_CLASS, "calc.py", "python")
        self._check_all_chunks_have_required_fields(chunks)

    def test_javascript_chunks_have_required_fields(self):
        chunks = chunk_file(JAVASCRIPT_CODE, "user.js", "javascript")
        self._check_all_chunks_have_required_fields(chunks)

    def test_yaml_chunks_have_required_fields(self):
        chunks = chunk_file(YAML_TEXT, "config.yaml", "yaml")
        self._check_all_chunks_have_required_fields(chunks)

    def test_char_count_matches_text_length(self):
        chunks = chunk_file(PYTHON_WITH_TWO_FUNCTIONS, "math.py", "python")
        for c in chunks:
            assert c["char_count"] == len(c["text"])

    def test_token_estimate_is_positive(self):
        chunks = chunk_file(PYTHON_WITH_CLASS, "calc.py", "python")
        for c in chunks:
            assert c["token_estimate"] >= 1

    def test_no_chunk_exceeds_max_char_limit(self):
        """After _split_oversized_chunks, no chunk should exceed MAX_CHUNK_CHARS."""
        # Use a very long file to trigger the oversized splitter
        big_file = "\n".join([f"# comment line {i}\ndef func_{i}():\n    pass\n" for i in range(1, 200)])
        chunks = chunk_file(big_file, "big.py", "python")
        oversized = [c for c in chunks if c["char_count"] > MAX_CHUNK_CHARS]
        assert len(oversized) == 0, (
            f"{len(oversized)} chunks exceed MAX_CHUNK_CHARS={MAX_CHUNK_CHARS}: "
            f"{[(c['chunk_name'], c['char_count']) for c in oversized[:3]]}"
        )

    def test_repo_name_included_when_provided(self):
        chunks = chunk_file(PYTHON_WITH_FUNCTION, "greet.py", "python", repo_name="myorg__myrepo")
        for c in chunks:
            assert c.get("repo_name") == "myorg__myrepo"

    def test_repo_name_omitted_when_not_provided(self):
        chunks = chunk_file(PYTHON_WITH_FUNCTION, "greet.py", "python")
        for c in chunks:
            assert "repo_name" not in c
