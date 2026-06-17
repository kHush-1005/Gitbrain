"""
core/code_chunker.py  [UPDATED — Week 2]
─────────────────────────────────────────────────────────────────────────────
CHANGES FROM WEEK 1:
    ✓ Added chunk_javascript_like_file() for JS, TS, Java, Go, Rust, C, C++
    ✓ Added MAX_CHUNK_TOKENS limit — chunks exceeding it are split further
    ✓ Added configurable WINDOW_SIZE and WINDOW_OVERLAP constants
    ✓ Richer metadata: char_count, token_estimate, repo_name field support
    ✓ Better fallback for YAML / JSON / Markdown (tighter window = 40 lines)
    ✓ Graceful handling of files with Windows-style line endings (CRLF)
    ✓ chunk_all_files() preserves repo_name in metadata when provided

CHUNKING STRATEGIES:
    1. AST-BASED     — Python only (ast module, exact line numbers)
    2. HEURISTIC     — JS / TS / Java / Go / Rust / C / C++ (regex boundaries)
    3. SLIDING WINDOW— Markdown / YAML / JSON / plain text / everything else

OUTPUT FORMAT (every chunk dict contains):
    {
        "text":            "def login(user, pw):\n    ...",
        "file_path":       "src/auth/login.py",
        "language":        "python",
        "chunk_type":      "function",     # function | class | method |
                                           # heuristic_block | window
        "chunk_name":      "login",
        "start_line":      42,
        "end_line":        58,
        "char_count":      312,
        "token_estimate":  78,             # rough: char_count / 4
        "repo_name":       "owner__repo",  # optional, set by chunk_all_files()
    }
─────────────────────────────────────────────────────────────────────────────
"""

import ast
import re
import logging

logger = logging.getLogger(__name__)

# ─── Chunk size configuration ─────────────────────────────────────────────────
MAX_CHUNK_TOKENS   = 512    # Chunks larger than this get split further
                            # 512 tokens ≈ 2,048 characters (rough estimate)
MAX_CHUNK_CHARS    = MAX_CHUNK_TOKENS * 4

# Sliding window configuration (used for non-code / fallback)
WINDOW_SIZE_CODE   = 60    # lines per window for code files
WINDOW_SIZE_TEXT   = 40    # lines per window for Markdown/YAML/JSON
WINDOW_OVERLAP     = 10    # overlap lines between consecutive windows
MIN_CHUNK_LINES    = 3     # skip chunks smaller than this

# ─── Language groups for heuristic chunker ────────────────────────────────────
HEURISTIC_LANGUAGES = {
    "javascript", "typescript", "java", "go", "rust", "c", "cpp",
    "csharp", "ruby", "php", "kotlin", "swift"
}

TEXT_LANGUAGES = {
    "markdown", "rst", "text", "yaml", "json", "toml", "ini",
    "bash", "makefile", "dockerfile", "sql"
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def chunk_file(
    content:   str,
    file_path: str,
    language:  str,
    repo_name: str = "",
) -> list[dict]:
    """
    Split a source file into chunks using the best strategy for its language.

    Strategy selection:
        Python                   → AST-based (most accurate, exact line numbers)
        JS/TS/Java/Go/Rust/C/C++ → Heuristic regex-based function/class detection
        Markdown/YAML/JSON/text  → Small sliding window (40 lines)
        Everything else          → Standard sliding window (60 lines)

    Args:
        content:   Raw file content as a string.
        file_path: File path for metadata (e.g. "src/auth/login.py").
        language:  Detected language string from detect_language().
        repo_name: Optional repository name for metadata.

    Returns:
        List of chunk dicts. Empty list if file is empty.
    """
    if not content or not content.strip():
        return []

    # Normalize line endings (Windows CRLF → Unix LF)
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    if language == "python":
        chunks = chunk_python_file(content, file_path, repo_name)
        if not chunks:
            logger.debug(f"AST returned 0 chunks for {file_path}, using fallback.")
            chunks = fallback_chunk_text(content, file_path, language, repo_name, WINDOW_SIZE_CODE)

    elif language in HEURISTIC_LANGUAGES:
        chunks = chunk_javascript_like_file(content, file_path, language, repo_name)
        if not chunks:
            logger.debug(f"Heuristic returned 0 chunks for {file_path}, using fallback.")
            chunks = fallback_chunk_text(content, file_path, language, repo_name, WINDOW_SIZE_CODE)

    elif language in TEXT_LANGUAGES:
        # Use a tighter window for non-code content
        chunks = fallback_chunk_text(content, file_path, language, repo_name, WINDOW_SIZE_TEXT)

    else:
        chunks = fallback_chunk_text(content, file_path, language, repo_name, WINDOW_SIZE_CODE)

    # Post-process: split any chunks that are still too large
    chunks = _split_oversized_chunks(chunks)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: AST-BASED PYTHON CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
def chunk_python_file(
    content:   str,
    file_path: str,
    repo_name: str = "",
) -> list[dict]:
    """
    Parse a Python file using the ast module and extract functions and classes.

    What gets extracted:
        - Top-level functions         → chunk_type = "function"
        - Top-level classes           → chunk_type = "class"
        - Methods inside classes      → chunk_type = "method"

    All line numbers are 1-based (matching what your editor shows).

    Args:
        content:   Python source code string.
        file_path: File path for metadata.
        repo_name: Optional repository name for metadata.

    Returns:
        List of chunk dicts, or [] if the file cannot be parsed by ast.
    """
    lines = content.splitlines()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logger.debug(f"AST SyntaxError in {file_path} (line {e.lineno}): {e.msg}")
        return []
    except Exception as e:
        logger.debug(f"AST unexpected error in {file_path}: {e}")
        return []

    chunks = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _make_python_function_chunk(
                node, lines, file_path, "function", "", repo_name
            )
            if chunk:
                chunks.append(chunk)

        elif isinstance(node, ast.ClassDef):
            # Add the whole class as one chunk
            class_chunk = _make_python_class_chunk(node, lines, file_path, repo_name)
            if class_chunk:
                chunks.append(class_chunk)

            # Add each method as a separate chunk
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = _make_python_function_chunk(
                        child, lines, file_path, "method", node.name, repo_name
                    )
                    if method_chunk:
                        chunks.append(method_chunk)

    return chunks


def _make_python_function_chunk(
    node:         ast.FunctionDef,
    lines:        list[str],
    file_path:    str,
    chunk_type:   str,
    parent_class: str,
    repo_name:    str,
) -> dict | None:
    """Build a chunk dict from an AST function/method node."""
    start = node.lineno      # 1-based
    end   = node.end_lineno  # 1-based

    if (end - start) < MIN_CHUNK_LINES - 1:
        return None

    text = "\n".join(lines[start - 1 : end])
    name = f"{parent_class}.{node.name}" if parent_class else node.name

    return _make_chunk(text, file_path, "python", chunk_type, name, start, end, repo_name)


def _make_python_class_chunk(
    node:      ast.ClassDef,
    lines:     list[str],
    file_path: str,
    repo_name: str,
) -> dict | None:
    """Build a chunk dict from an AST class node."""
    start = node.lineno
    end   = node.end_lineno

    if (end - start) < MIN_CHUNK_LINES - 1:
        return None

    text = "\n".join(lines[start - 1 : end])
    return _make_chunk(text, file_path, "python", "class", node.name, start, end, repo_name)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: HEURISTIC CHUNKING FOR JS/TS/JAVA/GO/RUST/C/C++
# ─────────────────────────────────────────────────────────────────────────────
def chunk_javascript_like_file(
    content:   str,
    file_path: str,
    language:  str,
    repo_name: str = "",
) -> list[dict]:
    """
    Split JS/TS/Java/Go/Rust/C/C++ files by detecting function and class
    boundaries using regular expressions.

    This is less precise than AST (it can't detect nested functions
    or handle edge cases in all language variants), but it correctly
    identifies the most common patterns and is far better than splitting
    by line count alone.

    Patterns detected:
        // JavaScript / TypeScript
        function myFunc(...)  { ... }
        const myFunc = (...) => { ... }
        async function myFunc() { ... }
        class MyClass { ... }
        myMethod() { ... }          (inside a class)

        // Java / C# / Go / Rust / C / C++
        public void myMethod(...)   { ... }
        func myFunc(...)            { ... }    (Go)
        fn my_func(...)             { ... }    (Rust)
        void myFunc(...)            { ... }    (C/C++)

    Strategy:
        1. Find all lines that look like function/class boundaries
        2. Split the file at those boundaries
        3. Each resulting block becomes one chunk

    Args:
        content:   Raw file source code.
        file_path: For metadata.
        language:  For metadata.
        repo_name: For metadata.

    Returns:
        List of chunk dicts.
    """
    lines = content.splitlines()
    if not lines:
        return []

    # Regex patterns that signal the START of a new function or class block
    BOUNDARY_PATTERNS = [
        # JavaScript/TypeScript: function declarations
        r"^\s*(export\s+)?(default\s+)?(async\s+)?function\s+\w+",
        # JS/TS: arrow function assigned to const/let/var
        r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s*)?\(",
        # JS/TS: class declaration
        r"^\s*(export\s+)?(default\s+)?class\s+\w+",
        # JS/TS: method inside class (no preceding keyword)
        r"^\s*(async\s+)?\w+\s*\([^)]*\)\s*\{",
        # Java/C#: method with access modifier
        r"^\s*(public|private|protected|static|final|abstract|override)\s+.*\w+\s*\(",
        # Java: class declaration
        r"^\s*(public|private|abstract|final)?\s*class\s+\w+",
        # Go: function declaration
        r"^\s*func\s+\w+",
        # Rust: function declaration
        r"^\s*(pub\s+)?(async\s+)?fn\s+\w+",
        # C/C++: common function patterns
        r"^\s*(static\s+|inline\s+|extern\s+)?\w+[\s\*]+\w+\s*\(",
    ]

    compiled = [re.compile(p) for p in BOUNDARY_PATTERNS]

    def is_boundary(line: str) -> bool:
        return any(pattern.match(line) for pattern in compiled)

    # Find boundary line indices (0-based)
    boundary_indices = [0]  # always start with line 0
    for i, line in enumerate(lines):
        if i > 0 and is_boundary(line):
            boundary_indices.append(i)

    # If no boundaries found, return empty (fallback will handle it)
    if len(boundary_indices) <= 1:
        return []

    chunks = []
    block_number = 1

    for idx, start_0 in enumerate(boundary_indices):
        # End of this block is the start of the next, or end of file
        end_0 = boundary_indices[idx + 1] if idx + 1 < len(boundary_indices) else len(lines)

        block_lines = lines[start_0:end_0]
        text = "\n".join(block_lines).strip()

        if len(text.splitlines()) < MIN_CHUNK_LINES:
            continue

        # Try to extract a name from the first line of the block
        first_line = block_lines[0].strip()
        chunk_name = _extract_heuristic_name(first_line, block_number)

        chunk = _make_chunk(
            text       = text,
            file_path  = file_path,
            language   = language,
            chunk_type = "heuristic_block",
            chunk_name = chunk_name,
            start_line = start_0 + 1,   # convert to 1-based
            end_line   = end_0,         # end_0 is exclusive, so line end_0 in 1-based
            repo_name  = repo_name,
        )
        chunks.append(chunk)
        block_number += 1

    return chunks


def _extract_heuristic_name(first_line: str, fallback_number: int) -> str:
    """
    Try to extract a meaningful name from the first line of a code block.

    Attempts to find:  function foo(  →  "foo"
                       class Bar      →  "Bar"
                       func baz(      →  "baz"
                       fn qux(        →  "qux"
                       public void do →  "do"

    Falls back to "block_{N}" if no name can be found.
    """
    # Try keywords: function, class, func, fn, def (shouldn't appear here but safe)
    match = re.search(
        r"\b(?:function|class|func|fn|def|void|public|private|protected|static)\s+(\w+)",
        first_line
    )
    if match:
        return match.group(1)

    # Try: identifier followed immediately by (
    match = re.search(r"(\w+)\s*\(", first_line)
    if match:
        return match.group(1)

    return f"block_{fallback_number}"


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3: SLIDING WINDOW FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def fallback_chunk_text(
    content:     str,
    file_path:   str,
    language:    str,
    repo_name:   str = "",
    window_size: int = WINDOW_SIZE_CODE,
) -> list[dict]:
    """
    Split any text file into overlapping windows of fixed line count.

    Used for:
        - Markdown, YAML, JSON, plain text
        - Any language that AST and heuristic chunking cannot handle
        - As a fallback when other strategies return 0 chunks

    Args:
        content:     Raw file content.
        file_path:   For metadata.
        language:    For metadata.
        repo_name:   For metadata.
        window_size: Lines per window (default varies by language type).

    Returns:
        List of window chunk dicts.
    """
    lines = content.splitlines()
    total = len(lines)

    if total == 0:
        return []

    # Small file: return as single chunk
    if total <= window_size:
        text = "\n".join(lines).strip()
        if len(text.splitlines()) < MIN_CHUNK_LINES:
            return []
        return [_make_chunk(text, file_path, language, "window", "window_1", 1, total, repo_name)]

    chunks = []
    window_number = 1
    step  = max(1, window_size - WINDOW_OVERLAP)
    start = 0

    while start < total:
        end        = min(start + window_size, total)
        text       = "\n".join(lines[start:end]).strip()

        if len(text.splitlines()) >= MIN_CHUNK_LINES:
            chunks.append(_make_chunk(
                text       = text,
                file_path  = file_path,
                language   = language,
                chunk_type = "window",
                chunk_name = f"window_{window_number}",
                start_line = start + 1,
                end_line   = end,
                repo_name  = repo_name,
            ))
            window_number += 1

        start += step
        if end == total:
            break

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build a standardized chunk dict
# ─────────────────────────────────────────────────────────────────────────────
def _make_chunk(
    text:       str,
    file_path:  str,
    language:   str,
    chunk_type: str,
    chunk_name: str,
    start_line: int,
    end_line:   int,
    repo_name:  str = "",
) -> dict:
    """
    Build a standardized chunk dictionary with all required metadata.

    All chunks produced by this module use this function, ensuring a
    consistent structure that Week 3's embedding pipeline can rely on.
    """
    char_count      = len(text)
    token_estimate  = max(1, char_count // 4)   # rough: 1 token ≈ 4 characters

    chunk = {
        "text":           text,
        "file_path":      file_path,
        "language":       language,
        "chunk_type":     chunk_type,
        "chunk_name":     chunk_name,
        "start_line":     start_line,
        "end_line":       end_line,
        "char_count":     char_count,
        "token_estimate": token_estimate,
    }
    if repo_name:
        chunk["repo_name"] = repo_name

    return chunk


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSOR: Split oversized chunks
# ─────────────────────────────────────────────────────────────────────────────
def _split_oversized_chunks(chunks: list[dict]) -> list[dict]:
    """
    Split any chunk whose text exceeds MAX_CHUNK_CHARS into smaller windows.

    This is a safety net for large class bodies or heuristic blocks that
    couldn't be broken down further by the primary strategy.

    Oversized chunks are split using fallback_chunk_text() with window_size=40.
    The original chunk is replaced by the resulting sub-chunks.
    """
    result = []
    for chunk in chunks:
        if chunk["char_count"] <= MAX_CHUNK_CHARS:
            result.append(chunk)
            continue

        # Split this oversized chunk into smaller windows
        logger.debug(
            f"Splitting oversized chunk '{chunk['chunk_name']}' "
            f"({chunk['char_count']} chars) in {chunk['file_path']}"
        )
        sub_chunks = fallback_chunk_text(
            content    = chunk["text"],
            file_path  = chunk["file_path"],
            language   = chunk["language"],
            repo_name  = chunk.get("repo_name", ""),
            window_size = 40,
        )
        # Fix sub-chunk line numbers relative to the original chunk's start
        offset = chunk["start_line"] - 1
        for sc in sub_chunks:
            sc["start_line"] += offset
            sc["end_line"]   += offset
            # Mark as a split piece
            sc["chunk_name"] = f"{chunk['chunk_name']}__{sc['chunk_name']}"
            sc["chunk_type"] = f"{chunk['chunk_type']}_split"

        result.extend(sub_chunks if sub_chunks else [chunk])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Chunk all files in a list
# ─────────────────────────────────────────────────────────────────────────────
def chunk_all_files(files: list[dict], repo_name: str = "") -> list[dict]:
    """
    Chunk every file in a list of file dicts.

    Args:
        files:     List of {path, content, language, size} dicts
                   from github_ingester.fetch_repository_files().
        repo_name: Optional repo name to embed in chunk metadata.

    Returns:
        Flat list of all chunks from all files.
    """
    all_chunks = []
    total = len(files)

    for i, file_info in enumerate(files, start=1):
        file_chunks = chunk_file(
            content   = file_info["content"],
            file_path = file_info["path"],
            language  = file_info["language"],
            repo_name = repo_name,
        )
        all_chunks.extend(file_chunks)

        if i % 20 == 0 or i == total:
            print(
                f"  [Chunker] [{i:>4}/{total}] {file_info['path'][:60]:<60} "
                f"→ {len(file_chunks)} chunks"
            )

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing code_chunker.py (Week 2)\n" + "─" * 50)

    # ── Test 1: Python AST chunking ───────────────────────────────────────────
    PYTHON_CODE = '''
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


async def fetch_data(url: str):
    """Fetch data from a URL asynchronously."""
    pass


class UserService:
    """Handles user-related operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        """Retrieve a user by ID."""
        return self.db.query(user_id)

    def create_user(self, name: str, email: str):
        """Create a new user."""
        return self.db.insert({"name": name, "email": email})
'''
    py_chunks = chunk_file(PYTHON_CODE, "services/user.py", "python", "myorg__myrepo")
    print(f"Python AST chunks: {len(py_chunks)}")
    for c in py_chunks:
        print(f"  [{c['chunk_type']:12}] {c['chunk_name']:<30} lines {c['start_line']:>3}–{c['end_line']}")

    # ── Test 2: JavaScript heuristic chunking ──────────────────────────────────
    JS_CODE = '''
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
        const user = this.service.get(req.params.id);
        res.json(user);
    }
}
'''
    js_chunks = chunk_file(JS_CODE, "controllers/user.js", "javascript", "myorg__myrepo")
    print(f"\nJavaScript heuristic chunks: {len(js_chunks)}")
    for c in js_chunks:
        print(f"  [{c['chunk_type']:16}] {c['chunk_name']:<25} lines {c['start_line']:>3}–{c['end_line']}")

    # ── Test 3: Fallback text chunking ─────────────────────────────────────────
    yaml_text = "\n".join([f"key_{i}: value_{i}" for i in range(1, 100)])
    yaml_chunks = chunk_file(yaml_text, "config/settings.yaml", "yaml")
    print(f"\nYAML fallback chunks: {len(yaml_chunks)}")
    for c in yaml_chunks:
        print(f"  [{c['chunk_type']:8}] {c['chunk_name']} lines {c['start_line']}–{c['end_line']}")

    # ── Test 4: Metadata completeness ─────────────────────────────────────────
    required = {"text", "file_path", "language", "chunk_type",
                "chunk_name", "start_line", "end_line", "char_count", "token_estimate"}
    all_test = py_chunks + js_chunks + yaml_chunks
    for chunk in all_test:
        missing = required - set(chunk.keys())
        assert not missing, f"Chunk missing fields: {missing}"
    print(f"\n✓ All {len(all_test)} chunks have required metadata fields")

    # ── Test 5: No oversized chunks ────────────────────────────────────────────
    oversized = [c for c in all_test if c["char_count"] > MAX_CHUNK_CHARS]
    print(f"✓ Oversized chunks after splitting: {len(oversized)} (should be 0)")
