"""
tests/test_repo_parser.py  [NEW — Week 2]
─────────────────────────────────────────────────────────────────────────────
Pytest tests for utils/repo_parser.py

Run with:
    pytest tests/test_repo_parser.py -v
    pytest tests/ -v                      (runs all tests)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import pytest

# ── Add project root to path so imports work ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.repo_parser import parse_github_url, make_collection_name


# ─────────────────────────────────────────────────────────────────────────────
# Tests for parse_github_url()
# ─────────────────────────────────────────────────────────────────────────────

class TestParseGithubUrl:
    """Tests for the parse_github_url() function."""

    def test_standard_https_url(self):
        """Standard HTTPS URL should be parsed correctly."""
        owner, repo = parse_github_url("https://github.com/tiangolo/fastapi")
        assert owner == "tiangolo"
        assert repo  == "fastapi"

    def test_trailing_slash_is_removed(self):
        """URL with a trailing slash should still parse correctly."""
        owner, repo = parse_github_url("https://github.com/tiangolo/fastapi/")
        assert owner == "tiangolo"
        assert repo  == "fastapi"

    def test_dot_git_suffix_is_removed(self):
        """URL ending in .git (clone URL format) should parse correctly."""
        owner, repo = parse_github_url("https://github.com/tiangolo/fastapi.git")
        assert owner == "tiangolo"
        assert repo  == "fastapi"

    def test_url_without_https_prefix(self):
        """URL without 'https://' prefix should still parse."""
        owner, repo = parse_github_url("github.com/psf/requests")
        assert owner == "psf"
        assert repo  == "requests"

    def test_http_url(self):
        """URL with http:// (not https://) should parse correctly."""
        owner, repo = parse_github_url("http://github.com/pallets/flask")
        assert owner == "pallets"
        assert repo  == "flask"

    def test_url_with_hyphens_in_repo_name(self):
        """Repository names with hyphens should be preserved."""
        owner, repo = parse_github_url("https://github.com/openai/openai-python")
        assert owner == "openai"
        assert repo  == "openai-python"

    def test_url_with_dots_in_repo_name(self):
        """Repository names with dots (e.g. version suffixes) should be preserved."""
        owner, repo = parse_github_url("https://github.com/someorg/my.repo.v2")
        assert owner == "someorg"
        assert repo  == "my.repo.v2"

    def test_url_with_org_that_has_hyphens(self):
        """Organisation names with hyphens should be preserved."""
        owner, repo = parse_github_url("https://github.com/my-cool-org/some-repo")
        assert owner == "my-cool-org"
        assert repo  == "some-repo"

    def test_returns_tuple_of_strings(self):
        """Return value should be a tuple of exactly two strings."""
        result = parse_github_url("https://github.com/tiangolo/fastapi")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for invalid URLs — should raise ValueError
# ─────────────────────────────────────────────────────────────────────────────

class TestParseGithubUrlInvalidInputs:
    """Tests for invalid URL inputs — all should raise ValueError."""

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_github_url("")

    def test_random_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_github_url("not-a-url-at-all")

    def test_non_github_domain_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_github_url("https://gitlab.com/someuser/somerepo")

    def test_url_with_no_repo_raises_value_error(self):
        """A URL that only has the owner but no repo name should fail."""
        with pytest.raises(ValueError):
            parse_github_url("https://github.com/tiangolo")

    def test_url_with_only_domain_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_github_url("https://github.com/")

    def test_bitbucket_url_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_github_url("https://bitbucket.org/user/repo")


# ─────────────────────────────────────────────────────────────────────────────
# Tests for make_collection_name()
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeCollectionName:
    """Tests for the make_collection_name() helper."""

    def test_simple_names_joined_with_double_underscore(self):
        name = make_collection_name("tiangolo", "fastapi")
        assert name == "tiangolo__fastapi"

    def test_hyphens_in_owner_replaced_with_underscore(self):
        name = make_collection_name("my-org", "myrepo")
        assert "-" not in name

    def test_hyphens_in_repo_replaced_with_underscore(self):
        name = make_collection_name("myorg", "my-repo")
        assert "-" not in name

    def test_dots_replaced_with_underscore(self):
        name = make_collection_name("someorg", "my.repo.v2")
        assert "." not in name

    def test_result_contains_double_underscore_separator(self):
        """The owner and repo should always be separated by '__'."""
        name = make_collection_name("owner", "repo")
        assert "__" in name

    def test_result_is_string(self):
        assert isinstance(make_collection_name("owner", "repo"), str)

    def test_no_slashes_in_result(self):
        """ChromaDB collection names cannot contain slashes."""
        name = make_collection_name("owner", "repo")
        assert "/" not in name
        assert "\\" not in name
