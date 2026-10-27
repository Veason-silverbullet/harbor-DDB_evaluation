"""
LLM/VLM judge for claims that need model-based verdict.

Uses OpenAI-compatible API to send evidence (text + optional screenshot)
and receive a structured verdict.
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Any
from types_.errors import QuotaExhaustedError, is_quota_error
import usage_tracker

logger = logging.getLogger(__name__)

VALID_VERDICTS = frozenset({
    "supported", "contradicted",
    "insufficient_evidence", "not_applicable",
})

_VERDICT_REGEX = (
    r'"verdict"\s*:\s*"(supported|contradicted|'
    r'insufficient_evidence|not_applicable|partial|partially_supported|not_supported)"'
)


def _normalize_verdict(v: str) -> str:
    if v in ("partial", "partially_supported"):
        return "contradicted"
    if v == "not_supported":
        return "insufficient_evidence"
    return v


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_llm_response(content: str) -> dict[str, Any]:
    """Parse LLM output into verdict dict with fallback strategies."""
    # Try direct JSON parse
    try:
        result = json.loads(content)
        if isinstance(result, dict) and result.get("verdict"):
            v = _normalize_verdict(str(result["verdict"]).strip())
            if v in VALID_VERDICTS:
                out = {
                    "verdict": v,
                    "confidence": float(result.get("confidence", 0.0)),
                    "reason": str(result.get("reason", "")),
                }
                if "evidence_sufficient" in result:
                    out["evidence_sufficient"] = bool(result["evidence_sufficient"])
                return out
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try extracting from markdown code block
    code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
    if code_match:
        try:
            result = json.loads(code_match.group(1).strip())
            if isinstance(result, dict) and result.get("verdict"):
                v = _normalize_verdict(str(result["verdict"]).strip())
                if v in VALID_VERDICTS:
                    out = {
                        "verdict": v,
                        "confidence": float(result.get("confidence", 0.0)),
                        "reason": str(result.get("reason", "")),
                    }
                    if "evidence_sufficient" in result:
                        out["evidence_sufficient"] = bool(result["evidence_sufficient"])
                    return out
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Try extracting first JSON object
    blob = _extract_json_object(content)
    if blob:
        try:
            result = json.loads(blob)
            if isinstance(result, dict) and result.get("verdict"):
                v = _normalize_verdict(str(result["verdict"]).strip())
                if v in VALID_VERDICTS:
                    out = {
                        "verdict": v,
                        "confidence": float(result.get("confidence", 0.0)),
                        "reason": str(result.get("reason", "")),
                    }
                    if "evidence_sufficient" in result:
                        out["evidence_sufficient"] = bool(result["evidence_sufficient"])
                    return out
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Regex fallback
    verdict_match = re.search(_VERDICT_REGEX, content)
    reason_match = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', content)

    if verdict_match:
        v = _normalize_verdict(verdict_match.group(1))
        return {
            "verdict": v if v in VALID_VERDICTS else "insufficient_evidence",
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "reason": reason_match.group(1) if reason_match else "Extracted from LLM output",
        }

    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.0,
        "reason": f"[JSON parse failed] Raw: {content[:200]}",
    }


def _format_runtime_signals(
    runtime_signals: dict[str, Any] | None,
) -> dict[str, str]:
    """Render runtime_signals into 5 short text blocks for the prompt.

    Always returns 5 keys (console_errors / failed_requests /
    external_hosts / hit_test_results / tts_events) so ``str.format``
    never KeyErrors, even on older runs that don't carry runtime_signals
    yet.
    """
    rs = runtime_signals or {}

    def _empty(label: str) -> str:
        return f"(none recorded — {label})"

    def _fmt_console(items: list[dict[str, Any]]) -> str:
        if not items:
            return _empty("no console.error / pageerror during this claim")
        lines: list[str] = []
        for it in items[:20]:
            lines.append(
                f"  - [{it.get('type', '?')}] "
                f"{(it.get('text') or '')[:240]}"
                + (f"  @{it.get('url')}" if it.get("url") else "")
            )
        if len(items) > 20:
            lines.append(f"  ... +{len(items) - 20} more")
        return "\n".join(lines)

    def _fmt_failed(items: list[dict[str, Any]]) -> str:
        if not items:
            return _empty("no requestfailed / 4xx / 5xx during this claim")
        lines: list[str] = []
        for it in items[:20]:
            status = it.get("status")
            status_s = str(status) if status is not None else (it.get("error") or "failed")
            lines.append(
                f"  - {it.get('method', '')} {it.get('url', '')}"
                f"  -> {it.get('kind', '?')}={status_s}"
                f" (host={it.get('host', '')})"
            )
        if len(items) > 20:
            lines.append(f"  ... +{len(items) - 20} more")
        return "\n".join(lines)

    def _fmt_hosts(items: list[dict[str, Any]]) -> str:
        if not items:
            return _empty("no external host hit recorded")
        lines: list[str] = []
        for it in sorted(items, key=lambda x: -int(x.get("count", 0)))[:20]:
            lines.append(
                f"  - {it.get('host', '')} "
                f"(count={it.get('count', 0)}, "
                f"is_external={bool(it.get('is_external'))}, "
                f"sample={it.get('sample_url', '')})"
            )
        return "\n".join(lines)

    def _fmt_hits(items: list[dict[str, Any]]) -> str:
        if not items:
            return _empty("no hit-test performed (CUA never invoked elementFromPoint)")
        lines: list[str] = []
        for it in items[:15]:
            actual = it.get("actual") or {}
            lines.append(
                f"  - label={it.get('label') or '(unlabeled)'}"
                f" pos=({it.get('x')},{it.get('y')})"
                f" hit={bool(it.get('hit'))}"
                f" actual_tag={actual.get('tag') if isinstance(actual, dict) else None}"
                f" pointer_events={actual.get('pointer_events') if isinstance(actual, dict) else None}"
                f" text={(actual.get('text') if isinstance(actual, dict) else '') or ''}"
            )
        return "\n".join(lines)

    def _fmt_tts(items: list[dict[str, Any]]) -> str:
        if not items:
            return _empty(
                "no speechSynthesis.speak() call recorded during this claim"
            )
        lines: list[str] = []
        for it in items[:15]:
            lines.append(
                f"  - speak(text=\"{(it.get('text') or '')[:160]}\","
                f" lang=\"{it.get('lang') or ''}\")"
                f" started={it.get('started')}"
                f" ended={it.get('ended')}"
                + (f" error={it.get('error')}" if it.get("error") else "")
            )
        if len(items) > 15:
            lines.append(f"  ... +{len(items) - 15} more")
        return "\n".join(lines)

    return {
        "console_errors": _fmt_console(rs.get("console_errors") or []),
        "failed_requests": _fmt_failed(rs.get("failed_requests") or []),
        "external_hosts": _fmt_hosts(rs.get("external_hosts_called") or []),
        "hit_test_results": _fmt_hits(rs.get("hit_test_results") or []),
        "tts_events": _fmt_tts(rs.get("tts_events") or []),
    }


def _format_action_summary(
    action_summary: dict[str, Any] | None,
) -> dict[str, str]:
    """Render an ``action_summary`` block (from CUA runners) into two text
    snippets the verify_claim prompt template can splice in directly.

    Returns ``{"action_distribution", "action_diagnostics"}``. Both keys
    are always present so ``str.format_map`` never KeyErrors on older
    cua_results that lacked the field. When the summary is missing or
    malformed we fall back to a short ``(unavailable)`` marker so the
    judge can still reason on the rest of the evidence.
    """
    summary = action_summary or {}
    by_tool = summary.get("by_tool") if isinstance(summary.get("by_tool"), dict) else {}

    if not by_tool:
        distribution = "(no actions recorded by CUA)"
    else:
        # Sort by count desc so the most-used tools surface first; cap to
        # 12 entries so we don't blow up the prompt when CUA spammed
        # exotic actions.
        sorted_items = sorted(
            by_tool.items(),
            key=lambda kv: (-int(kv[1] or 0), str(kv[0])),
        )[:12]
        distribution = ", ".join(
            f"{name}={count}" for name, count in sorted_items
        )

    parts = [
        f"total_steps={int(summary.get('total_steps') or 0)}",
        f"failed={int(summary.get('failed_count') or 0)}",
        f"clamped={int(summary.get('clamped_count') or 0)}",
        f"synthetic={int(summary.get('synthetic_count') or 0)}",
        f"unknown={int(summary.get('unknown_count') or 0)}",
    ]
    diagnostics = " ".join(parts)

    return {
        "action_distribution": distribution,
        "action_diagnostics": diagnostics,
    }


class _SafeFormatDict(dict):
    """``str.format_map`` helper that tolerates unknown / future keys.

    If the prompt template on disk grows a new ``{some_key}`` placeholder
    while the running Python process still has an older mapping in memory,
    plain ``template.format(...)`` raises ``KeyError`` mid-run and bricks
    every remaining claim. Using ``template.format_map(_SafeFormatDict(...))``
    instead substitutes ``"(missing evidence: <key>)"`` for any unknown
    placeholder so the worst case is a slightly degraded prompt rather
    than a crash.
    """

    def __missing__(self, key: str) -> str:  # noqa: D401
        return f"(missing evidence: {key})"


def _frame_label(idx: int, total: int) -> str:
    """Human-readable, time-ordered label for a judge screenshot.

    Used so a vision LLM can tell that ``Screenshot 1/3`` is the
    *start* of the CUA interaction window and ``Screenshot 3/3`` is
    the *final* state — required for ``visual_semantics`` claims
    about directional change (countdown decreases, progress bar
    advances, etc.).
    """
    if total <= 1:
        return "[Screenshot 1/1]"
    if idx == 0:
        return f"[Screenshot {idx + 1}/{total} — start frame (before CUA interactions)]"
    if idx == total - 1:
        return f"[Screenshot {idx + 1}/{total} — final frame (after CUA interactions)]"
    return f"[Screenshot {idx + 1}/{total} — mid frame]"


def _build_judge_messages(
    checkpoint: str,
    verification: str,
    claim_type: str,
    criticality: str,
    page_info: dict[str, Any],
    cua_reasoning: str,
    action_trace: list[dict],
    runtime_signals: dict[str, Any] | None = None,
    screenshot_b64: str | None = None,
    screenshot_b64_list: list[str] | None = None,
    action_summary: dict[str, Any] | None = None,
    semantic_type: str | None = None,
) -> list[dict[str, Any]]:
    """Build messages for the LLM judge.

    ``screenshot_b64_list`` (time-ordered) is the preferred way to
    pass evidence. ``screenshot_b64`` (single image) is kept for
    backward compatibility with older callers — it is folded into a
    one-element list if ``screenshot_b64_list`` is empty/None.
    """
    prompt_path = Path(__file__).parent / "prompts" / "verify_claim.txt"
    template = prompt_path.read_text(encoding="utf-8")

    action_count = len(action_trace)
    action_success = sum(1 for a in action_trace if a.get("success", False))
    rs_blocks = _format_runtime_signals(runtime_signals)
    summary_blocks = _format_action_summary(action_summary)

    fmt_map = _SafeFormatDict(
        checkpoint=checkpoint,
        verification=verification,
        claim_type=claim_type,
        semantic_type=semantic_type or claim_type,
        criticality=criticality,
        page_url=page_info.get("page_url", ""),
        page_title=page_info.get("page_title", ""),
        visible_text=(page_info.get("visible_text", "") or "")[:4000],
        action_count=action_count,
        action_success_count=action_success,
        action_distribution=summary_blocks["action_distribution"],
        action_diagnostics=summary_blocks["action_diagnostics"],
        cua_reasoning=cua_reasoning or "(no CUA reasoning available)",
        console_errors=rs_blocks["console_errors"],
        failed_requests=rs_blocks["failed_requests"],
        external_hosts=rs_blocks["external_hosts"],
        hit_test_results=rs_blocks["hit_test_results"],
        tts_events=rs_blocks["tts_events"],
    )

    try:
        text_prompt = template.format_map(fmt_map)
    except (IndexError, ValueError) as e:
        logger.warning(
            "verify_claim template format failed (%s); falling back to "
            "the raw template text.",
            e,
        )
        text_prompt = template

    images: list[str] = [s for s in (screenshot_b64_list or []) if s]
    if not images and screenshot_b64:
        images = [screenshot_b64]

    if not images:
        return [{"role": "user", "content": text_prompt}]

    total = len(images)
    content_parts: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    if total > 1:
        content_parts.append({
            "type": "text",
            "text": (
                f"You will receive {total} screenshots captured during this "
                "claim's CUA session, in chronological order. They are labeled "
                "below; compare across frames to judge any directional / "
                "time-series change the claim references."
            ),
        })
    for i, b64 in enumerate(images):
        content_parts.append({"type": "text", "text": _frame_label(i, total)})
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })
    return [{"role": "user", "content": content_parts}]


# Total judge attempts (1 initial + retries) for transient failures before
# we give up and fall back to ``insufficient_evidence``. Quota errors
# short-circuit and are never retried here.
_JUDGE_MAX_ATTEMPTS = 2


def _is_temperature_unsupported_error(exc: Exception) -> bool:
    """Detect gateways/models that reject the ``temperature`` parameter.

    Some backends (e.g. ``claude-opus-4-8-aws`` via the az gateway) return a
    400 ```temperature` is deprecated for this model`` instead of silently
    ignoring the field. We retry without ``temperature`` rather than zeroing
    out the whole claim.
    """
    msg = str(exc).lower()
    return "temperature" in msg and (
        "deprecated" in msg
        or "unsupported" in msg
        or "not supported" in msg
        or "does not support" in msg
    )


async def _create_with_temperature_fallback(client, **kwargs):
    """Call chat.completions.create, retrying without ``temperature`` if the
    model/gateway rejects that parameter."""
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001 - re-raised below unless it's the temp case
        if _is_temperature_unsupported_error(e) and "temperature" in kwargs:
            logger.warning(
                "model %s rejected `temperature`; retrying without it",
                kwargs.get("model"),
            )
            kwargs.pop("temperature", None)
            return await client.chat.completions.create(**kwargs)
        raise


# ── Anthropic Messages fallback ────────────────────────────────────
# Some gateways (e.g. yibuapi's `claude-opus-4-8` channel) forward the
# OpenAI-format `/v1/chat/completions` body to Anthropic WITHOUT converting
# multimodal blocks, so Anthropic's schema validation rejects our
# `{"type": "image_url"}` parts with a 400:
#   Input tag 'image_url' found using 'type' does not match any of the
#   expected tags: ... 'image' ...
# Text-only judge calls happen to survive because both protocols accept
# `{"type": "text"}`. When we see this signature we retry the SAME evidence
# through the Anthropic-native Messages API (the exact transport the
# anthropic CUA adapter already uses successfully on these gateways) and
# remember the (base_url, model) pair so later claims skip the doomed
# OpenAI-format attempt entirely.
_ANTHROPIC_FALLBACK_ROUTES: set[tuple[str, str]] = set()


def _prefer_anthropic_route(base_url: str, model: str) -> bool:
    """Known-bad routes we skip the OpenAI-format attempt for entirely.

    yibu gateways verifiably do NOT translate OpenAI-style ``image_url``
    parts for their Claude channels (each run used to burn ~2 min on the
    doomed first attempt before the reactive fallback kicked in), so any
    Claude model behind a yibu base_url goes straight to the Anthropic
    Messages API.
    """
    return (
        "yibu" in (base_url or "").lower()
        and (model or "").lower().startswith("claude")
    )


def _is_anthropic_image_schema_error(exc: Exception) -> bool:
    """Detect the gateway passthrough bug described above."""
    msg = str(exc)
    return "image_url" in msg and (
        "does not match any of the expected tags" in msg
        or "Input tag" in msg
    )


def _openai_to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI chat-completions messages to Anthropic Messages format.

    Only the shapes emitted by ``_build_judge_messages`` are handled:
    plain-string content, ``{"type": "text"}`` parts, and
    ``{"type": "image_url"}`` parts carrying a base64 PNG data URL.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for p in content or []:
            if p.get("type") == "image_url":
                url = (p.get("image_url") or {}).get("url", "")
                b64 = url.split("base64,", 1)[-1]
                parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                })
            else:
                parts.append({"type": "text", "text": p.get("text", "")})
        out.append({"role": m["role"], "content": parts})
    return out


async def _call_anthropic_judge(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the judge through the Anthropic-native Messages API.

    Returns ``(text, thinking_trace)`` — the concatenated text of the
    response plus the serialized thinking/redacted_thinking blocks (text +
    signature, same shape as the CUA adapters' ``thinking_trace``; empty
    when ``thinking`` is off or the gateway strips the blocks). Mirrors the
    base_url normalization in ``cua/anthropic_fc_adapter.py`` (the SDK
    appends /v1 itself, while our yaml configs keep the /v1 suffix for
    OpenAI reuse).
    """
    import anthropic
    # Lazy import: keeps this module free of cua/* (and playwright)
    # dependencies unless thinking serialization is actually needed.
    from cua.native_adapter import _thinking_blocks_to_trace

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        kwargs["base_url"] = url

    client = anthropic.AsyncAnthropic(**kwargs)
    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": _openai_to_anthropic_messages(messages),
    }
    if thinking:
        # Same shape as cua/anthropic_fc_adapter (validated end-to-end by
        # probe_anthropic_fc.py): adaptive budget, summarized display.
        # Anthropic rejects `temperature` alongside thinking, so it is
        # deliberately NOT sent on this branch.
        create_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        create_kwargs["output_config"] = {"effort": "high"}
    else:
        create_kwargs["temperature"] = temperature
    try:
        response = await client.messages.create(**create_kwargs)
    except Exception as e:  # noqa: BLE001 - same fallback as the OpenAI path
        if not _is_temperature_unsupported_error(e) or "temperature" not in create_kwargs:
            raise
        logger.warning(
            "model %s rejected `temperature` on the Anthropic path; "
            "retrying without it", model,
        )
        create_kwargs.pop("temperature", None)
        response = await client.messages.create(**create_kwargs)
    usage_tracker.record_response("llm_judge", response, model=model)
    text = "".join(
        getattr(block, "text", "") for block in (response.content or [])
        if getattr(block, "type", "") == "text"
    ).strip()
    thinking_blocks = [
        b for b in (response.content or [])
        if getattr(b, "type", "") in ("thinking", "redacted_thinking")
    ]
    thinking_trace = (
        _thinking_blocks_to_trace(1, thinking_blocks) if thinking_blocks else []
    )
    return text, thinking_trace


async def judge_claim(
    checkpoint: str,
    verification: str,
    claim_type: str,
    criticality: str,
    page_info: dict[str, Any],
    cua_reasoning: str = "",
    action_trace: list[dict] | None = None,
    runtime_signals: dict[str, Any] | None = None,
    screenshot_b64: str | None = None,
    screenshot_b64_list: list[str] | None = None,
    action_summary: dict[str, Any] | None = None,
    semantic_type: str | None = None,
    *,
    api_key: str = "",
    base_url: str = "",
    model: str = "claude-opus-4-6",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    thinking: bool = False,
) -> dict[str, Any]:
    """
    Call LLM judge to evaluate a claim based on collected evidence.

    ``screenshot_b64_list`` (chronological) takes priority over the
    legacy single-image ``screenshot_b64``. Returns a dict with
    verdict, confidence, reason, evidence_sufficient — plus
    ``judge_thinking_trace`` (thinking text + signature blocks) when
    ``thinking`` is on and the Anthropic transport was used.
    """
    if not api_key:
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.0,
            "reason": "[LLM API key not configured]",
            "evidence_sufficient": False,
        }

    messages = _build_judge_messages(
        checkpoint=checkpoint,
        verification=verification,
        claim_type=claim_type,
        criticality=criticality,
        page_info=page_info,
        cua_reasoning=cua_reasoning,
        action_trace=action_trace or [],
        runtime_signals=runtime_signals,
        screenshot_b64=screenshot_b64,
        screenshot_b64_list=screenshot_b64_list,
        action_summary=action_summary,
        semantic_type=semantic_type,
    )

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.0,
            "reason": "[openai package not installed]",
            "evidence_sufficient": False,
        }

    client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)

    # Transient-failure retry. A single judge call hiccup (timeout, 5xx
    # gateway burp, malformed-but-retryable response) shouldn't depend on a
    # single network round-trip. We retry up to ``_JUDGE_MAX_ATTEMPTS`` times;
    # if every attempt fails we fall back to ``insufficient_evidence`` (which
    # is still scored as 0 in the denominator — a failed judgement is "I
    # couldn't tell", not a free pass).
    use_anthropic = (
        _prefer_anthropic_route(base_url, model)
        or (base_url, model) in _ANTHROPIC_FALLBACK_ROUTES
    )
    last_exc: Exception | None = None
    for attempt in range(_JUDGE_MAX_ATTEMPTS):
        try:
            judge_thinking_trace: list[dict[str, Any]] = []
            if use_anthropic:
                content, judge_thinking_trace = await _call_anthropic_judge(
                    messages,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking=thinking,
                )
            else:
                response = await _create_with_temperature_fallback(
                    client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage_tracker.record_response("llm_judge", response, model=model)
                content = response.choices[0].message.content.strip()
            result = parse_llm_response(content)
            if judge_thinking_trace:
                result["judge_thinking_trace"] = judge_thinking_trace
            return result
        except Exception as e:  # noqa: BLE001
            if is_quota_error(e):
                raise QuotaExhaustedError("llm_judge", str(e)) from e
            last_exc = e
            if not use_anthropic and _is_anthropic_image_schema_error(e):
                logger.warning(
                    "gateway rejected OpenAI-format image blocks for model %s "
                    "at %s; switching this route to the Anthropic Messages "
                    "API for the rest of the run",
                    model, base_url,
                )
                _ANTHROPIC_FALLBACK_ROUTES.add((base_url, model))
                use_anthropic = True
            logger.warning(
                "LLM judge call failed (attempt %d/%d): %s",
                attempt + 1, _JUDGE_MAX_ATTEMPTS, e,
            )

    logger.error("LLM judge call failed after %d attempts: %s",
                 _JUDGE_MAX_ATTEMPTS, last_exc)
    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.0,
        "reason": f"[LLM error] {type(last_exc).__name__}: {last_exc}",
        "evidence_sufficient": False,
    }
