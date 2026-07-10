"""Tool layer: web search + date extraction. The seam for swapping search providers.

Dates matter: the model cannot weight recency if recency never reaches it. Tavily
returns published_date reliably on the news topic and spottily elsewhere, so we
take what we're given and mark the rest explicitly undated.
"""

import re
import time
from email.utils import parsedate_to_datetime

import requests

import config

TAVILY_URL = "https://api.tavily.com/search"

_cache: dict[str, tuple[float, list[dict]]] = {}


class SearchError(RuntimeError):
    pass


_ISO_RE = re.compile(r"(?:19|20)\d{2}-\d{2}-\d{2}")


def _normalise_date(raw) -> str | None:
    """Tavily returns RFC 2822: 'Mon, 06 Jul 2026 06:34:45 GMT'. Some sources
    return ISO. Normalise both to YYYY-MM-DD; anything else is treated as undated."""
    if not raw:
        return None
    s = str(raw).strip()

    m = _ISO_RE.search(s)
    if m:
        return m.group(0)

    try:
        return parsedate_to_datetime(s).date().isoformat()
    except (TypeError, ValueError):
        return None


def _cache_get(key: str) -> list[dict] | None:
    hit = _cache.get(key)
    if not hit:
        return None
    ts, val = hit
    if time.time() - ts > config.SEARCH_CACHE_TTL:
        _cache.pop(key, None)
        return None
    return val


def web_search(
    query: str,
    topic: str = "general",
    max_results: int | None = None,
    include_domains: list[str] | None = None,
) -> list[dict]:
    """Search. Returns [{title, url, content, date, score}]. `date` may be None."""
    key = f"{topic}:{query}:{','.join(include_domains or [])}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    payload = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results or config.SEARCH_MAX_RESULTS,
        "include_answer": False,
        "include_raw_content": False,
        "topic": topic,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    try:
        resp = requests.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            json=payload,
            timeout=config.SEARCH_TIMEOUT,
        )
    except requests.RequestException as e:
        raise SearchError(f"Search request failed: {e}") from e

    if resp.status_code != 200:
        raise SearchError(f"Tavily returned HTTP {resp.status_code}: {resp.text[:200]}")

    results = []
    for r in resp.json().get("results", []):
        content = (r.get("content") or "").strip()
        title = r.get("title", "").strip()
        if not content:
            # Video pages and social posts routinely extract to an empty body. Dropping
            # them here silently discarded the highest-signal sources -- a Citi video,
            # an X post -- before anything downstream could judge them. Fall back to the
            # title, which for these pages IS the content.
            if not title:
                continue
            content = title
        results.append(
            {
                "title": title or "Untitled",
                "url": r.get("url", ""),
                "content": content,
                "date": _normalise_date(r.get("published_date")),
                "score": float(r.get("score") or 0.0),
            }
        )

    _cache[key] = (time.time(), results)
    return results
