"""
core/github_ingester.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Talk to the GitHub REST API to fetch all code files from a repository.

WHAT IT DOES:
    1. Builds the correct Authorization headers (with or without token).
    2. Fetches the complete file tree of a repository.
    3. Filters out binary/unwanted files (images, zips, lock files, etc.).
    4. Fetches the raw text content of each code file.
    5. Returns a list of file dictionaries for the chunker to process.

OUTPUT FORMAT (list of dicts):
    [
        {
            "path":     "src/auth/login.py",
            "content":  "def login(user, password): ...",
            "language": "python"
        },
        ...
    ]

GITHUB API ENDPOINTS USED:
    GET /repos/{owner}/{repo}
        → Discover the default branch name (main / master)

    GET /repos/{owner}/{repo}/branches/{branch}
        → Get the latest commit SHA for that branch

    GET /repos/{owner}/{repo}/git/trees/{sha}?recursive=1
        → Get the full file tree (all file paths)

    GET /repos/{owner}/{repo}/contents/{path}
        → Get the content of one file (returned as base64)
─────────────────────────────────────────────────────────────────────────────
"""

import base64
import os
import time
import logging
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Logger setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── GitHub API base URL ──────────────────────────────────────────────────────
GITHUB_API = "https://api.github.com"

# ─── File extensions to skip (binary / non-code files) ───────────────────────
# These will never contain useful text for a code assistant.
SKIP_EXTENSIONS = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp", ".tiff",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z", ".xz",
    # Compiled / binary
    ".exe", ".dll", ".so", ".dylib", ".bin", ".out", ".class", ".jar", ".war",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Media
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    # Build artifacts
    ".pyc", ".pyo", ".pyd", ".o", ".a",
    # Font files
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Lock files (not useful for code understanding)
    ".lock",
    # Minified JS/CSS (unreadable)
    ".min.js", ".min.css",
    # Source map files
    ".map",
}

# ─── Directory names to skip ──────────────────────────────────────────────────
# These directories typically contain dependencies or generated files, not
# the project's own source code.
SKIP_DIRECTORIES = {
    "node_modules",   # JavaScript dependencies
    ".git",           # Git internal files
    "__pycache__",    # Python bytecode cache
    "dist",           # Build output
    "build",          # Build output
    "vendor",         # Go / PHP dependencies
    ".venv",          # Python virtual environment
    "venv",           # Python virtual environment
    "env",            # Python virtual environment
    ".idea",          # JetBrains IDE files
    ".vscode",        # VS Code settings
    "coverage",       # Test coverage reports
    ".nyc_output",    # NYC coverage output
    "eggs",           # Python egg distributions
    ".eggs",
}

# ─── Maximum file size to fetch (bytes) ───────────────────────────────────────
# GitHub truncates files > 1MB anyway. We skip large files proactively.
MAX_FILE_SIZE_BYTES = 500_000  # 500 KB


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build request headers
# ─────────────────────────────────────────────────────────────────────────────
def get_headers(token: str = "") -> dict:
    """
    Build HTTP headers for GitHub API requests.

    Args:
        token: GitHub Personal Access Token. Optional but highly recommended.
               Without a token: 60 requests/hour limit.
               With a token:   5,000 requests/hour limit.

    Returns:
        A dict of headers to pass to requests.get().
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Should we skip this file?
# ─────────────────────────────────────────────────────────────────────────────
def should_skip_file(path: str, size_bytes: int = 0) -> bool:
    """
    Decide whether a file should be skipped during ingestion.

    Reasons to skip:
        - File is in a skip directory (node_modules, .git, etc.)
        - File has a binary/non-code extension (.png, .zip, etc.)
        - File is too large (> 500 KB)

    Args:
        path:       Relative file path from the repo root, e.g. "src/auth.py"
        size_bytes: File size in bytes (from the GitHub tree API).

    Returns:
        True  → skip this file
        False → process this file
    """
    # Split the path into its directory components
    parts = path.split("/")

    # Check if any directory segment is in the skip list
    # e.g. "node_modules/lodash/index.js" → parts[0] = "node_modules" → skip
    for part in parts[:-1]:  # exclude the filename itself
        if part in SKIP_DIRECTORIES:
            return True

    # Check file extension
    # Get extension from the filename (last part of the path)
    filename = parts[-1].lower()
    for ext in SKIP_EXTENSIONS:
        if filename.endswith(ext):
            return True

    # Skip files that have no extension and are not common script files
    # (e.g. Makefile, Dockerfile are fine; binary blobs are not)
    # We'll allow no-extension files through and rely on content detection.

    # Check file size
    if size_bytes > MAX_FILE_SIZE_BYTES:
        logger.debug(f"Skipping large file ({size_bytes} bytes): {path}")
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Detect programming language from file extension
# ─────────────────────────────────────────────────────────────────────────────
def detect_language(path: str) -> str:
    """
    Detect the programming language of a file based on its extension.

    This is used as metadata on each chunk, and also determines which
    chunking strategy to use (AST-based for Python, heuristic for others).

    Args:
        path: File path, e.g. "src/auth/login.py"

    Returns:
        A lowercase language string, e.g. "python", "javascript", "text"
    """
    # Extract the extension from the filename
    filename = path.rsplit("/", 1)[-1].lower()  # just the filename
    if "." not in filename:
        return "text"

    ext = filename.rsplit(".", 1)[-1]  # everything after the last dot

    LANGUAGE_MAP = {
        # Python
        "py":    "python",
        "pyw":   "python",
        # JavaScript / TypeScript
        "js":    "javascript",
        "jsx":   "javascript",
        "ts":    "typescript",
        "tsx":   "typescript",
        "mjs":   "javascript",
        "cjs":   "javascript",
        # Java
        "java":  "java",
        # Go
        "go":    "go",
        # Rust
        "rs":    "rust",
        # C / C++
        "c":     "c",
        "h":     "c",
        "cpp":   "cpp",
        "cc":    "cpp",
        "cxx":   "cpp",
        "hpp":   "cpp",
        # C#
        "cs":    "csharp",
        # Ruby
        "rb":    "ruby",
        # PHP
        "php":   "php",
        # Shell
        "sh":    "bash",
        "bash":  "bash",
        "zsh":   "bash",
        # Data / Config
        "json":  "json",
        "yaml":  "yaml",
        "yml":   "yaml",
        "toml":  "toml",
        "ini":   "ini",
        "cfg":   "ini",
        "env":   "text",
        # Markup
        "md":    "markdown",
        "rst":   "rst",
        "txt":   "text",
        "html":  "html",
        "htm":   "html",
        "xml":   "xml",
        "css":   "css",
        "scss":  "css",
        "sass":  "css",
        # SQL
        "sql":   "sql",
        # Dockerfile
        "dockerfile": "dockerfile",
    }

    # Special case: files named exactly "Dockerfile"
    if filename == "dockerfile":
        return "dockerfile"
    if filename == "makefile":
        return "makefile"

    return LANGUAGE_MAP.get(ext, "text")


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION 1: Fetch the complete file tree
# ─────────────────────────────────────────────────────────────────────────────
def fetch_file_tree(owner: str, repo: str, token: str = "") -> list[dict]:
    """
    Fetch the complete list of files in a GitHub repository.

    Uses the Git Trees API with recursive=1 to get every file in one request,
    rather than traversing directories one by one.

    Args:
        owner: Repository owner username, e.g. "tiangolo"
        repo:  Repository name, e.g. "fastapi"
        token: GitHub Personal Access Token (optional but recommended).

    Returns:
        A list of file info dicts:
        [
            {"path": "src/main.py", "sha": "abc123", "size": 1024},
            ...
        ]
        Only blob (file) entries are included — directories are excluded.

    Raises:
        RuntimeError: If the GitHub API returns an error response.
    """
    headers = get_headers(token)

    # ── Step 1: Get repository info to find the default branch ────────────────
    logger.info(f"Fetching repo info for: {owner}/{repo}")
    repo_url = f"{GITHUB_API}/repos/{owner}/{repo}"
    response = requests.get(repo_url, headers=headers, timeout=15)

    if response.status_code == 404:
        raise RuntimeError(
            f"Repository not found: {owner}/{repo}\n"
            "Check: is the URL correct? Is the repo private? Do you need a token?"
        )
    if response.status_code == 403:
        remaining = response.headers.get("X-RateLimit-Remaining", "?")
        raise RuntimeError(
            f"GitHub API access forbidden (HTTP 403).\n"
            f"Rate limit remaining: {remaining}\n"
            "Add a GitHub token to your .env file to increase the limit to 5,000/hour."
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Unexpected GitHub API error {response.status_code}: {response.text[:200]}"
        )

    repo_data = response.json()
    default_branch = repo_data.get("default_branch", "main")
    logger.info(f"Default branch: {default_branch}")

    # ── Step 2: Get the latest commit SHA for the default branch ──────────────
    branch_url = f"{GITHUB_API}/repos/{owner}/{repo}/branches/{default_branch}"
    branch_response = requests.get(branch_url, headers=headers, timeout=15)
    branch_response.raise_for_status()
    branch_data = branch_response.json()
    sha = branch_data["commit"]["sha"]
    logger.info(f"Latest commit SHA: {sha[:8]}...")

    # ── Step 3: Get the complete recursive file tree ───────────────────────────
    tree_url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
    logger.info(f"Fetching file tree (recursive)...")
    tree_response = requests.get(tree_url, headers=headers, timeout=30)
    tree_response.raise_for_status()
    tree_data = tree_response.json()

    if tree_data.get("truncated"):
        logger.warning(
            "WARNING: The file tree was truncated by GitHub (repo has > 100,000 files). "
            "Some files may be missing from the index."
        )

    # ── Step 4: Filter to only file blobs, applying skip rules ────────────────
    all_items = tree_data.get("tree", [])
    files = []
    skipped = 0

    for item in all_items:
        if item.get("type") != "blob":
            # Skip tree (directory) entries
            continue

        path = item.get("path", "")
        size = item.get("size", 0)

        if should_skip_file(path, size):
            skipped += 1
            continue

        files.append({
            "path": path,
            "sha":  item.get("sha", ""),
            "size": size,
        })

    logger.info(
        f"File tree fetched: {len(files)} code files found, "
        f"{skipped} files skipped (binary/unwanted)."
    )
    return files


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Rate limit checker
# ─────────────────────────────────────────────────────────────────────────────
def _check_and_handle_rate_limit(response: requests.Response) -> None:
    """
    Inspect GitHub API rate limit headers and pause if we're running low.

    GitHub returns these headers on every response:
        X-RateLimit-Limit     → total requests allowed per hour
        X-RateLimit-Remaining → requests left in current window
        X-RateLimit-Reset     → Unix timestamp when the window resets

    This function pauses execution if remaining requests < 10,
    waiting until the rate limit window resets.
    """
    remaining = int(response.headers.get("X-RateLimit-Remaining", 100))
    reset_ts   = int(response.headers.get("X-RateLimit-Reset",     time.time() + 60))

    if remaining < 10:
        wait_seconds = max(0, reset_ts - int(time.time())) + 5  # +5s buffer
        logger.warning(
            f"GitHub API rate limit low! Only {remaining} requests remaining. "
            f"Pausing for {wait_seconds:.0f} seconds until reset..."
        )
        time.sleep(wait_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION 2: Fetch the content of a single file
# ─────────────────────────────────────────────────────────────────────────────
def fetch_file_content(owner: str, repo: str, path: str, token: str = "") -> str:
    """
    Fetch the raw text content of a single file from a GitHub repository.

    GitHub returns file content encoded as Base64.
    This function decodes it and returns the raw UTF-8 text.

    Args:
        owner: Repository owner username.
        repo:  Repository name.
        path:  File path relative to repo root, e.g. "src/auth/login.py"
        token: GitHub Personal Access Token (optional).

    Returns:
        The decoded file content as a string.
        Returns an empty string if the file cannot be fetched or decoded.
    """
    headers = get_headers(token)
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        _check_and_handle_rate_limit(response)

        if response.status_code == 200:
            data = response.json()
            encoding = data.get("encoding", "")

            if encoding == "base64":
                # Decode the base64 content
                # GitHub adds newlines inside the base64 string — remove them first
                raw_base64 = data["content"].replace("\n", "")
                decoded_bytes = base64.b64decode(raw_base64)

                # Decode bytes to string, replacing undecodable bytes
                return decoded_bytes.decode("utf-8", errors="replace")

            elif encoding == "none":
                # File is too large for the contents API (> 1MB)
                logger.debug(f"File too large for contents API, skipping: {path}")
                return ""

            else:
                logger.debug(f"Unknown encoding '{encoding}' for file: {path}")
                return ""

        elif response.status_code == 404:
            logger.debug(f"File not found (may have been deleted): {path}")
            return ""

        else:
            logger.warning(
                f"Could not fetch {path} — HTTP {response.status_code}"
            )
            return ""

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching file: {path}")
        return ""
    except Exception as e:
        logger.warning(f"Error fetching file {path}: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION 3: Orchestrate full repository file fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_repository_files(
    owner: str,
    repo: str,
    token: str = "",
    max_files: int = 500,
) -> list[dict]:
    """
    Fetch all code files from a GitHub repository.

    This is the main function called by the test script and (in Week 4) by
    the FastAPI /ingest endpoint.

    It:
        1. Fetches the file tree (list of all file paths).
        2. Fetches the raw content of each file.
        3. Attaches detected language to each file.
        4. Returns the result as a list of file dicts.

    Args:
        owner:     Repository owner username.
        repo:      Repository name.
        token:     GitHub Personal Access Token (optional but recommended).
        max_files: Safety cap — stop after this many files. Default: 500.
                   Increase for very large repositories.

    Returns:
        A list of file content dicts:
        [
            {
                "path":     "src/auth/login.py",
                "content":  "def login(user, password):\n    ...",
                "language": "python",
                "size":     1234
            },
            ...
        ]
    """
    # ── Step 1: Get the file list ──────────────────────────────────────────────
    print(f"\n[GitBrain] Fetching file tree for: {owner}/{repo}")
    file_tree = fetch_file_tree(owner, repo, token)

    # Apply safety cap
    if len(file_tree) > max_files:
        print(
            f"[GitBrain] Large repository detected ({len(file_tree)} files). "
            f"Processing first {max_files} files only.\n"
            f"           Increase max_files parameter to process more."
        )
        file_tree = file_tree[:max_files]

    total = len(file_tree)
    print(f"[GitBrain] {total} files to fetch. Starting download...\n")

    # ── Step 2: Fetch each file's content ─────────────────────────────────────
    result = []
    fetched  = 0
    failed   = 0
    empty    = 0

    for i, file_info in enumerate(file_tree, start=1):
        path = file_info["path"]

        # Progress indicator every 10 files
        if i % 10 == 0 or i == total:
            print(f"  [{i:>4}/{total}] Fetching: {path[:70]}")

        content = fetch_file_content(owner, repo, path, token)

        if not content.strip():
            empty += 1
            continue

        language = detect_language(path)

        result.append({
            "path":     path,
            "content":  content,
            "language": language,
            "size":     file_info.get("size", 0),
        })
        fetched += 1

    # ── Step 3: Print summary ──────────────────────────────────────────────────
    print(f"\n[GitBrain] ✓ Fetch complete!")
    print(f"           Files fetched with content : {fetched}")
    print(f"           Files empty / skipped      : {empty}")
    print(f"           Files failed               : {failed}")

    return result
