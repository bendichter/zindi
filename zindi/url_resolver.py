"""URL resolution for DANDI and other services.

Handles:
- DANDI API URL → presigned S3 URL resolution (with optional API key auth)
- Resolved URL caching (10-minute TTL)
- Pluggable additional URL resolvers

Ported from lindi.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import requests

_resolved_url_cache: dict[str, dict] = {}  # url -> {"timestamp": float, "url": str}
_additional_url_resolvers: list[Callable[[str], str]] = []


def add_url_resolver(resolver: Callable[[str], str]) -> None:
    """Register an additional URL resolver.

    The resolver is a function that takes a URL and returns a (possibly
    modified) URL. Resolvers are called in order before DANDI resolution.
    """
    _additional_url_resolvers.append(resolver)


def resolve_url(url: str) -> str:
    """Resolve a URL, handling DANDI redirects and caching."""
    for resolver in _additional_url_resolvers:
        url = resolver(url)

    if url in _resolved_url_cache:
        elapsed = time.time() - _resolved_url_cache[url]["timestamp"]
        if elapsed < 60 * 10:  # 10-minute TTL
            return _resolved_url_cache[url]["url"]

    if _is_dandi_url(url):
        resolved = _resolve_dandi_url(url)
    else:
        resolved = url

    _resolved_url_cache[url] = {"timestamp": time.time(), "url": resolved}
    return resolved


def _is_dandi_url(url: str) -> bool:
    return url.startswith("https://api.dandiarchive.org/api/") or url.startswith(
        "https://api.sandbox.dandiarchive.org/"
    )


def _resolve_dandi_url(url: str) -> str:
    """Resolve a DANDI API URL to a presigned S3 URL."""
    headers: dict[str, str] = {}
    if url.startswith("https://api.dandiarchive.org/api/"):
        api_key = os.environ.get("DANDI_API_KEY")
        if api_key:
            headers["Authorization"] = f"token {api_key}"
    elif url.startswith("https://api.sandbox.dandiarchive.org/"):
        api_key = os.environ.get("DANDI_SANDBOX_API_KEY")
        if api_key:
            headers["Authorization"] = f"token {api_key}"
    resp = requests.head(url, allow_redirects=True, headers=headers)
    return str(resp.url)
