"""Pluggable web-search handler backing the ``search_web`` veracity tool.

The tool lets the CUA fetch the CURRENT real-world truth (sports fixtures,
exchange rates, weather, prices, ...) and compare it against what the app
displays, so a stale/mock value is caught rather than scored as merely
present.

Design goals:
- Provider-agnostic. Configured via ``cua.search`` in the eval yaml
  (``provider`` / ``api_key`` / ``endpoint`` / ``max_results``) and surfaced
  on ``RunnerConfig`` as ``search_*`` fields.
- Never a hard failure. If no key/endpoint is configured, or the provider
  errors, the handler returns ``success=True`` with a short "unavailable"
  note so the CUA/judge degrades to ``insufficient_evidence`` — it must NEVER
  cause a false ``contradicted``.
- Stdlib-only HTTP (``urllib``) run in a thread so the async loop is not
  blocked and no new dependency is introduced.

Built-in providers (all authenticate via ``Authorization: Bearer <api_key>``,
which works for both the native vendor APIs and the yibuapi.com unified
gateway the project already uses):
- ``tavily``  : POST {endpoint} {query, max_results} -> {answer?, results:[{title,url,content}]}
                default endpoint https://yibuapi.com/tavily/search
- ``serper``  : POST {endpoint} {q, num} -> {organic:[{title,link,snippet}]}
                (also sends X-API-KEY for native serper.dev)
- ``custom``  : POST {endpoint} {query} ; best-effort extraction of common
                result shapes.
"""
from __future__ import annotations
import asyncio
import json
import logging
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINTS = {
    # Default to the yibuapi unified gateway the project already uses, so the
    # same Bearer key that drives the LLM also drives veracity search.
    "tavily": "https://yibuapi.com/tavily/search",
    "serper": "https://yibuapi.com/serper/search",
}


def _ok(text: str) -> dict[str, Any]:
    return {"success": True, "error": None, "text": text, "image_b64": None, "clamped": False}


def _fail(msg: str) -> dict[str, Any]:
    return {"success": False, "error": msg, "text": msg, "image_b64": None, "clamped": False}


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _extract_results(provider: str, obj: dict[str, Any], max_results: int) -> list[str]:
    """Best-effort normalize a provider response into ``["<title> — <snippet> (<url>)", ...]``."""
    rows: list[dict[str, Any]] = []
    if provider == "serper":
        rows = obj.get("organic") or []
    else:
        # tavily and most "custom" shapes use ``results``.
        rows = obj.get("results") or obj.get("data") or obj.get("organic") or []
    # tavily also returns a synthesized ``answer`` — surface it first if present.
    out: list[str] = []
    answer = obj.get("answer")
    if isinstance(answer, str) and answer.strip():
        out.append(f"[answer] {answer.strip()}")
    for r in rows[: max(1, max_results)]:
        if not isinstance(r, dict):
            continue
        title = str(r.get("title") or r.get("name") or "").strip()
        snippet = str(r.get("content") or r.get("snippet") or r.get("description") or "").strip()
        link = str(r.get("url") or r.get("link") or "").strip()
        line = " — ".join([p for p in (title, snippet) if p])
        if link:
            line = f"{line} ({link})" if line else link
        if line:
            out.append(line)
    return out


async def run_search(cfg: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Handler for the ``search_web`` tool.

    ``cfg`` is the runner's ``RunnerConfig`` (reads ``search_provider`` /
    ``search_api_key`` / ``search_endpoint`` / ``search_max_results``).
    """
    query = str(args.get("query") or "").strip()
    if not query:
        return _fail("search_web requires a non-empty 'query'")

    provider = (getattr(cfg, "search_provider", "") or "tavily").strip().lower()
    api_key = (getattr(cfg, "search_api_key", "") or "").strip()
    endpoint = (getattr(cfg, "search_endpoint", "") or "").strip() or _DEFAULT_ENDPOINTS.get(provider, "")
    max_results = int(getattr(cfg, "search_max_results", 5) or 5)

    if not api_key or not endpoint:
        # Graceful degradation: never let a missing key false-fail a claim.
        return _ok(
            "search unavailable: no search API key/endpoint is configured "
            "(cua.search). Cannot verify against real-world truth — decide "
            "from on-page evidence only and do NOT mark the claim contradicted "
            "solely because you could not search."
        )

    # Every supported endpoint (native vendor APIs + yibuapi gateway) accepts
    # Bearer auth; serper.dev additionally wants X-API-KEY, so we send both.
    bearer = {"Authorization": f"Bearer {api_key}"}
    if provider == "tavily":
        payload = {"query": query, "max_results": max_results}
        headers: dict[str, str] = dict(bearer)
    elif provider == "serper":
        payload = {"q": query, "num": max_results}
        headers = {**bearer, "X-API-KEY": api_key}
    else:  # custom
        payload = {"query": query, "max_results": max_results}
        headers = dict(bearer)

    try:
        obj = await asyncio.to_thread(_http_post_json, endpoint, payload, headers, 20.0)
    except Exception as e:  # noqa: BLE001
        logger.warning("search_web provider=%s failed: %s", provider, e)
        return _ok(
            f"search unavailable (provider {provider} error: {type(e).__name__}); "
            "decide from on-page evidence only and do NOT mark the claim "
            "contradicted solely because search failed."
        )

    results = _extract_results(provider, obj, max_results)
    if not results:
        return _ok(f"search returned no usable results for: {query!r}")
    body = "\n".join(f"- {line}" for line in results)
    return _ok(f"Web search results for {query!r} (real-world ground truth):\n{body}")
