"""
utils/repo_parser.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Parse a GitHub repository URL and extract the owner and repository name.

WHAT IT DOES:
    Given a URL like:  https://github.com/tiangolo/fastapi
    It returns:        owner="tiangolo", repo="fastapi"

WHY IT EXISTS:
    The GitHub REST API needs the owner and repo separately.
    This module keeps that parsing logic in one clean place so
    github_ingester.py stays focused on API calls.
─────────────────────────────────────────────────────────────────────────────
"""


def parse_github_url(url: str) -> tuple[str, str]:
    """
    Parse a GitHub repository URL and return (owner, repo).

    Handles these URL formats:
        https://github.com/owner/repo
        https://github.com/owner/repo/
        https://github.com/owner/repo.git
        github.com/owner/repo          (no https://)

    Args:
        url: The GitHub repository URL as a string.

    Returns:
        A tuple of (owner, repo_name) — both as strings.

    Raises:
        ValueError: If the URL does not look like a valid GitHub repo URL.

    Example:
        owner, repo = parse_github_url("https://github.com/tiangolo/fastapi")
        # owner = "tiangolo"
        # repo  = "fastapi"
    """
    # Remove trailing slash and .git suffix if present
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    # Accept URLs with or without the https:// prefix
    if url.startswith("https://"):
        url = url[len("https://"):]
    elif url.startswith("http://"):
        url = url[len("http://"):]

    # Now url looks like: github.com/owner/repo
    parts = url.split("/")

    # We expect at least: ["github.com", "owner", "repo"]
    if len(parts) < 3 or parts[0].lower() != "github.com":
        raise ValueError(
            f"Invalid GitHub URL: '{url}'\n"
            "Expected format: https://github.com/owner/repository"
        )

    owner = parts[1].strip()
    repo  = parts[2].strip()

    if not owner or not repo:
        raise ValueError(
            f"Could not extract owner or repo from URL: '{url}'"
        )

    return owner, repo


def make_collection_name(owner: str, repo: str) -> str:
    """
    Create a sanitized collection name for use with ChromaDB (Week 3).

    ChromaDB has strict naming rules:
        - alphanumeric characters and underscores only
        - no hyphens, slashes, or dots

    Args:
        owner: GitHub repository owner.
        repo:  GitHub repository name.

    Returns:
        A safe collection name string, e.g. "tiangolo__fastapi"

    Example:
        make_collection_name("my-org", "my-repo") → "my_org__my_repo"
    """
    safe_owner = owner.replace("-", "_").replace(".", "_")
    safe_repo  = repo.replace("-",  "_").replace(".", "_")
    return f"{safe_owner}__{safe_repo}"


# ─── Quick self-test (runs only when this file is executed directly) ──────────
if __name__ == "__main__":
    test_urls = [
        "https://github.com/tiangolo/fastapi",
        "https://github.com/tiangolo/fastapi/",
        "https://github.com/tiangolo/fastapi.git",
        "github.com/openai/openai-python",
    ]
    print("Testing repo_parser.py\n" + "─" * 40)
    for test_url in test_urls:
        try:
            owner, repo = parse_github_url(test_url)
            name = make_collection_name(owner, repo)
            print(f"URL    : {test_url}")
            print(f"Owner  : {owner}")
            print(f"Repo   : {repo}")
            print(f"CollName: {name}")
            print()
        except ValueError as e:
            print(f"ERROR  : {e}\n")
