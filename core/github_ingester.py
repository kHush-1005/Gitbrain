"""
core/github_ingester.py  [UPDATED — Week 2]
─────────────────────────────────────────────────────────────────────────────
CHANGES FROM WEEK 1:
    ✓ Explicit 1 MB file-size guard before fetching content
    ✓ Structured custom exceptions (GitHubRateLimitError, RepoNotFoundError)
    ✓ Retry logic (up to 3 attempts) for transient network failures
    ✓ Cleaner progress reporting with per-file size display
    ✓ Returns ingestion summary dict alongside files list
    ✓ SKIP_EXTENSIONS and SKIP_DIRECTORIES exposed as module-level constants
      (so validate_chunks.py and tests can reference them)

GITHUB API ENDPOINTS USED:
    GET /repos/{owner}/{repo}
        → Discover the default branch name

    GET /repos/{owner}/{repo}/branches/{branch}
        → Get the latest commit SHA

    GET /repos/{owner}/{repo}/git/trees/{sha}?recursive=1
        → Full recursive file tree in one request

    GET /repos/{owner}/{repo}/contents/{path}
        → Raw file content (Base64 encoded by GitHub)
─────────────────────────────────────────────────────────────────────────────
"""

import base64
import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── GitHub API base ──────────────────────────────────────────────────────────
GITHUB_API = "https://api.github.com"

# ─── File-size limit (bytes) ──────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 1_000_000   # 1 MB — GitHub truncates above this anyway

# ─── Extensions to skip (binary / non-textual / build artifacts) ─────────────
SKIP_EXTENSIONS = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp", ".tiff",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z", ".xz",
    # Compiled / binary
    ".exe", ".dll", ".so", ".dylib", ".bin", ".out",
    ".class", ".jar", ".war", ".ear",
    ".pyc", ".pyo", ".pyd", ".o", ".a",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Media
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Lock files (generated, not useful for comprehension)
    ".lock",
    # Minified / source-map
    ".min.js", ".min.css", ".map",
})

# ─── Directory names to skip ──────────────────────────────────────────────────
SKIP_DIRECTORIES = frozenset({
    "node_modules", ".git", "__pycache__", "dist", "build",
    "vendor", ".venv", "venv", "env", ".idea", ".vscode",
    "coverage", ".nyc_output", "eggs", ".eggs",
    "site-packages", ".tox", ".pytest_cache", ".mypy_cache",
})


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────
class GitHubRateLimitError(Exception):
    """Raised when the GitHub API rate limit is exhausted."""


class RepoNotFoundError(Exception):
    """Raised when the target repository does not exist or is inaccessible."""


class GitHubAPIError(Exception):
    """Raised for unexpected GitHub API responses."""


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build request headers
# ─────────────────────────────────────────────────────────────────────────────
def get_headers(token: str = "") -> dict:
    """
    Build HTTP headers for GitHub API requests.

    Without token: 60 requests / hour.
    With token:    5,000 requests / hour.
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
def should_skip_file(path: str, size_bytes: int = 0) -> tuple[bool, str]:
    """
    Decide whether a file should be skipped during ingestion.

    Returns:
        (True, reason_string)  — if the file should be skipped
        (False, "")            — if the file should be processed

    Having a reason string makes logging and debugging much clearer.
    """
    parts = path.split("/")

    # Check directory segments (everything except the filename)
    for part in parts[:-1]:
        if part in SKIP_DIRECTORIES:
            return True, f"directory '{part}' is in skip list"

    # Check file extension
    filename = parts[-1].lower()
    for ext in SKIP_EXTENSIONS:
        if filename.endswith(ext):
            return True, f"extension '{ext}' is in skip list"

    # Check file size
    if size_bytes > MAX_FILE_SIZE_BYTES:
        return True, f"file too large ({size_bytes:,} bytes > {MAX_FILE_SIZE_BYTES:,} limit)"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Detect programming language from file extension
# ─────────────────────────────────────────────────────────────────────────────
def detect_language(path: str) -> str:
    """Map a file path to a language string based on its extension."""
    filename = path.rsplit("/", 1)[-1].lower()

    # Special-case filenames with no extension
    if filename == "dockerfile":
        return "dockerfile"
    if filename in ("makefile", "gnumakefile"):
        return "makefile"
    if filename in ("gemfile", "rakefile", "guardfile"):
        return "ruby"
    if filename in ("pipfile",):
        return "toml"

    if "." not in filename:
        return "text"

    ext = filename.rsplit(".", 1)[-1]

    LANGUAGE_MAP = {
        "py": "python", "pyw": "python",
        "js": "javascript", "jsx": "javascript", "mjs": "javascript", "cjs": "javascript",
        "ts": "typescript", "tsx": "typescript",
        "java": "java",
        "go": "go",
        "rs": "rust",
        "c": "c", "h": "c",
        "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp",
        "cs": "csharp",
        "rb": "ruby",
        "php": "php",
        "kt": "kotlin", "kts": "kotlin",
        "swift": "swift",
        "sh": "bash", "bash": "bash", "zsh": "bash",
        "json": "json",
        "yaml": "yaml", "yml": "yaml",
        "toml": "toml",
        "ini": "ini", "cfg": "ini",
        "md": "markdown", "mdx": "markdown",
        "rst": "rst",
        "txt": "text",
        "html": "html", "htm": "html",
        "xml": "xml",
        "css": "css", "scss": "css", "sass": "css",
        "sql": "sql",
        "graphql": "graphql", "gql": "graphql",
        "proto": "protobuf",
    }

    return LANGUAGE_MAP.get(ext, "text")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Rate-limit awareness
# ─────────────────────────────────────────────────────────────────────────────
def _handle_rate_limit(response: requests.Response) -> None:
    """
    Inspect rate-limit headers and pause if remaining requests are low.

    Pauses execution until the rate limit window resets when remaining < 10.
    """
    try:
        remaining = int(response.headers.get("X-RateLimit-Remaining", "100"))
        reset_ts  = int(response.headers.get("X-RateLimit-Reset",     str(int(time.time()) + 60)))
    except (ValueError, TypeError):
        return

    if remaining < 10:
        wait = max(5, reset_ts - int(time.time())) + 5   # +5 s buffer
        print(
            f"\n  ⚠ GitHub rate limit low ({remaining} remaining). "
            f"Pausing {wait:.0f} s until reset..."
        )
        logger.warning(f"Rate limit low ({remaining}). Waiting {wait}s.")
        time.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: GET with retry
# ─────────────────────────────────────────────────────────────────────────────
def _get_with_retry(url: str, headers: dict, max_retries: int = 3, timeout: int = 20) -> requests.Response:
    """
    Make a GET request with exponential-backoff retry for transient failures.

    Retries on:
        - Connection errors
        - Timeout errors
        - HTTP 5xx server errors

    Does NOT retry on:
        - HTTP 4xx client errors (bad token, repo not found, etc.)

    Args:
        url:         URL to fetch.
        headers:     Request headers.
        max_retries: Maximum number of attempts (default: 3).
        timeout:     Request timeout in seconds (default: 20).

    Returns:
        requests.Response object.

    Raises:
        requests.exceptions.RequestException if all retries fail.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            # Retry on 5xx errors but not on 4xx
            if response.status_code < 500:
                return response
            logger.warning(
                f"Server error {response.status_code} on attempt {attempt}/{max_retries}: {url}"
            )
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_error = e
            logger.warning(f"Network error on attempt {attempt}/{max_retries}: {e}")

        if attempt < max_retries:
            wait = 2 ** (attempt - 1)   # 1s, 2s, 4s
            logger.debug(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise requests.exceptions.RequestException(
        f"All {max_retries} retries failed for: {url}"
    ) from last_error


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Fetch the complete file tree
# ─────────────────────────────────────────────────────────────────────────────
def fetch_file_tree(owner: str, repo: str, token: str = "") -> list[dict]:
    """
    Fetch the complete list of file paths in a GitHub repository.

    Uses the Git Trees API (recursive=1) to get all files in a single request.

    Args:
        owner: Repository owner (e.g. "tiangolo").
        repo:  Repository name (e.g. "fastapi").
        token: GitHub Personal Access Token (optional, but strongly recommended).

    Returns:
        List of dicts: [{"path": "src/main.py", "sha": "abc", "size": 1024}, ...]

    Raises:
        RepoNotFoundError:     If the repository doesn't exist or is private.
        GitHubRateLimitError:  If the API rate limit is exhausted.
        GitHubAPIError:        For other unexpected API errors.
    """
    headers = get_headers(token)

    # ── Step 1: Get repo metadata (find default branch) ───────────────────────
    logger.info(f"Fetching repo info: {owner}/{repo}")
    resp = _get_with_retry(f"{GITHUB_API}/repos/{owner}/{repo}", headers)

    if resp.status_code == 404:
        raise RepoNotFoundError(
            f"Repository not found: github.com/{owner}/{repo}\n"
            "  Check: correct URL? Is the repo private? Do you need a token?"
        )
    if resp.status_code == 403:
        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
        if remaining == "0":
            raise GitHubRateLimitError(
                "GitHub API rate limit exhausted. "
                "Add a GITHUB_TOKEN to .env or wait ~60 minutes."
            )
        raise GitHubAPIError(
            f"Access forbidden (HTTP 403). "
            f"Token may lack required permissions. Remaining: {remaining}"
        )
    if resp.status_code == 401:
        raise GitHubAPIError(
            "Unauthorized (HTTP 401). Your GITHUB_TOKEN is invalid or expired."
        )
    if resp.status_code != 200:
        raise GitHubAPIError(
            f"Unexpected API response {resp.status_code} for {owner}/{repo}: "
            f"{resp.text[:200]}"
        )

    repo_data       = resp.json()
    default_branch  = repo_data.get("default_branch", "main")
    logger.info(f"Default branch: {default_branch}")

    # ── Step 2: Get latest commit SHA ──────────────────────────────────────────
    branch_resp = _get_with_retry(
        f"{GITHUB_API}/repos/{owner}/{repo}/branches/{default_branch}", headers
    )
    branch_resp.raise_for_status()
    sha = branch_resp.json()["commit"]["sha"]
    logger.info(f"Latest commit: {sha[:8]}...")

    # ── Step 3: Recursive file tree ────────────────────────────────────────────
    tree_resp = _get_with_retry(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{sha}?recursive=1",
        headers, timeout=30
    )
    tree_resp.raise_for_status()
    tree_data = tree_resp.json()

    if tree_data.get("truncated"):
        logger.warning(
            "GitHub returned a TRUNCATED file tree (repo > 100,000 files). "
            "Some files will be missing."
        )
        print("  ⚠  Warning: repository is very large — file tree was truncated by GitHub.")

    # ── Step 4: Filter blobs ───────────────────────────────────────────────────
    files, skipped = [], 0
    for item in tree_data.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        size = item.get("size", 0)
        skip, reason = should_skip_file(path, size)
        if skip:
            logger.debug(f"SKIP {path}: {reason}")
            skipped += 1
        else:
            files.append({"path": path, "sha": item.get("sha", ""), "size": size})

    print(
        f"  File tree fetched: {len(files)} code files, "
        f"{skipped} skipped (binary/large/unwanted)"
    )
    return files


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Fetch single file content
# ─────────────────────────────────────────────────────────────────────────────
def fetch_file_content(owner: str, repo: str, path: str, token: str = "") -> str:
    """
    Fetch and decode the raw text content of one repository file.

    GitHub returns content as Base64. This function decodes it.

    Returns empty string ("") if:
        - File is not found
        - File encoding is not base64 (e.g. large files with encoding="none")
        - Any network/decoding error occurs

    Args:
        owner: Repo owner.
        repo:  Repo name.
        path:  File path from repo root.
        token: Optional GitHub token.

    Returns:
        Decoded UTF-8 text content, or "".
    """
    headers = get_headers(token)
    url     = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

    try:
        resp = _get_with_retry(url, headers, max_retries=2, timeout=15)
        _handle_rate_limit(resp)

        if resp.status_code == 200:
            data     = resp.json()
            encoding = data.get("encoding", "")

            if encoding == "base64":
                raw = data["content"].replace("\n", "")
                return base64.b64decode(raw).decode("utf-8", errors="replace")

            if encoding == "none":
                # File is too large for the Contents API (> 1 MB)
                logger.debug(f"File too large for contents API: {path}")
                return ""

            logger.debug(f"Unknown encoding '{encoding}': {path}")
            return ""

        if resp.status_code == 404:
            logger.debug(f"File not found (possibly deleted): {path}")
            return ""

        logger.warning(f"HTTP {resp.status_code} fetching {path}")
        return ""

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching: {path}")
        return ""
    except Exception as e:
        logger.warning(f"Error fetching {path}: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Orchestrate full repository ingestion
# ─────────────────────────────────────────────────────────────────────────────
def fetch_repository_files(
    owner:     str,
    repo:      str,
    token:     str = "",
    max_files: int = 500,
) -> tuple[list[dict], dict]:
    """
    Fetch all code files from a GitHub repository.

    New in Week 2: also returns a summary dict with ingestion statistics.

    Args:
        owner:     Repository owner.
        repo:      Repository name.
        token:     GitHub Personal Access Token.
        max_files: Safety cap on number of files fetched.

    Returns:
        Tuple of:
            files   — list of {path, content, language, size} dicts
            summary — dict with ingestion statistics

    Raises:
        RepoNotFoundError, GitHubRateLimitError, GitHubAPIError
    """
    print(f"\n[GitBrain] Fetching file tree: {owner}/{repo}")
    file_tree = fetch_file_tree(owner, repo, token)

    if not file_tree:
        return [], {"total_files": 0, "fetched": 0, "empty": 0, "failed": 0}

    if len(file_tree) > max_files:
        print(
            f"  ⚠  Large repo: {len(file_tree)} files found, "
            f"capping at {max_files}. Increase max_files to process more."
        )
        file_tree = file_tree[:max_files]

    total   = len(file_tree)
    fetched = 0
    empty   = 0
    failed  = 0
    result  = []

    print(f"[GitBrain] Fetching content for {total} files...\n")
    for i, file_info in enumerate(file_tree, start=1):
        path = file_info["path"]
        size = file_info.get("size", 0)

        if i % 10 == 0 or i == total:
            print(f"  [{i:>4}/{total}]  {path[:65]:<65}  ({size:>7,} B)")

        content = fetch_file_content(owner, repo, path, token)

        if not content.strip():
            empty += 1
            continue

        result.append({
            "path":     path,
            "content":  content,
            "language": detect_language(path),
            "size":     size,
        })
        fetched += 1

    summary = {
        "owner":       owner,
        "repo":        repo,
        "total_files": total,
        "fetched":     fetched,
        "empty":       empty,
        "failed":      failed,
    }

    print(f"\n[GitBrain] ✓ Done! {fetched} files fetched, {empty} empty/skipped.")
    return result, summary
