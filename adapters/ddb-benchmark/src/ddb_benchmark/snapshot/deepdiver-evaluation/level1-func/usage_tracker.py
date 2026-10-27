"""
Global token-usage tracker for an evaluation run.

Every LLM/VLM call in the pipeline (CUA loop, LLM judge, meta-review) goes
through a handful of provider SDKs whose responses each carry a ``usage``
object. This module gives those call sites a single, process-wide,
thread/async-safe sink to record:

    * input tokens   (prompt / input)
    * output tokens  (completion / output)
    * request count  (一次成功的模型调用 = 一条)

Usage is bucketed by *category* (``cua`` / ``llm_judge`` / ``meta_review``)
and by *model* so the end-of-run report can break the bill down per stage
and per model while still exposing a grand total.

Why a module-level singleton instead of threading the tracker through every
constructor: the call sites are scattered across four CUA adapters plus two
verification helpers, several of which run on different threads (the sync
OpenAI/Anthropic clients are dispatched via ``run_in_executor``) and across
concurrent per-query orchestrators. A locked module global is the least
invasive way to make every one of them aggregate into the same place.

Recording is best-effort and never raises: a missing/odd ``usage`` shape
just increments the ``requests`` counter (and, separately, an
``unknown_usage`` counter) so a gateway that strips usage data degrades to
"we made N calls but couldn't price them" instead of crashing the eval.
"""
from __future__ import annotations
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class _Bucket:
    """Mutable accumulator for one (category) or (category, model) slice."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    # Calls whose response carried no usable usage payload. Counted in
    # ``requests`` too, but surfaced separately so a report can flag
    # "token totals are an undercount" honestly.
    unknown_usage: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
            "unknown_usage": self.unknown_usage,
        }


class UsageTracker:
    """Thread/async-safe accumulator of token usage across an eval run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_category: dict[str, _Bucket] = {}
        # Nested: {category: {model: _Bucket}}.
        self._by_category_model: dict[str, dict[str, _Bucket]] = {}

    def reset(self) -> None:
        with self._lock:
            self._by_category.clear()
            self._by_category_model.clear()

    def record(
        self,
        category: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str | None = None,
        had_usage: bool = True,
    ) -> None:
        """Record one model call.

        Args:
            category: stage bucket, e.g. ``"cua"`` / ``"llm_judge"`` /
                ``"meta_review"``.
            input_tokens / output_tokens: token counts (0 when unknown).
            model: optional model id for the per-model breakdown.
            had_usage: ``False`` when the response carried no usable usage
                payload; still counts as a request but bumps
                ``unknown_usage``.
        """
        cat = category or "unknown"
        mdl = model or "unknown"
        try:
            in_t = int(input_tokens or 0)
        except (TypeError, ValueError):
            in_t = 0
        try:
            out_t = int(output_tokens or 0)
        except (TypeError, ValueError):
            out_t = 0

        with self._lock:
            cb = self._by_category.setdefault(cat, _Bucket())
            cb.input_tokens += in_t
            cb.output_tokens += out_t
            cb.requests += 1
            if not had_usage:
                cb.unknown_usage += 1

            mb = self._by_category_model.setdefault(cat, {}).setdefault(mdl, _Bucket())
            mb.input_tokens += in_t
            mb.output_tokens += out_t
            mb.requests += 1
            if not had_usage:
                mb.unknown_usage += 1

    def record_response(
        self,
        category: str,
        response: Any,
        *,
        model: str | None = None,
    ) -> None:
        """Extract usage from a provider response object and record it.

        Tolerates OpenAI chat-completions (``prompt_tokens`` /
        ``completion_tokens``), OpenAI Responses API + Anthropic
        (``input_tokens`` / ``output_tokens``), plain dicts, and missing
        usage entirely. Never raises.
        """
        in_t, out_t, had_usage = extract_usage(response)
        self.record(
            category,
            input_tokens=in_t,
            output_tokens=out_t,
            model=model or _extract_model(response),
            had_usage=had_usage,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of everything recorded so far."""
        with self._lock:
            by_category = {
                cat: bucket.to_dict() for cat, bucket in self._by_category.items()
            }
            by_category_model = {
                cat: {mdl: b.to_dict() for mdl, b in models.items()}
                for cat, models in self._by_category_model.items()
            }
            total = _Bucket()
            for bucket in self._by_category.values():
                total.input_tokens += bucket.input_tokens
                total.output_tokens += bucket.output_tokens
                total.requests += bucket.requests
                total.unknown_usage += bucket.unknown_usage

        return {
            "total": total.to_dict(),
            "by_category": by_category,
            "by_category_model": by_category_model,
        }

    def format_summary(self) -> str:
        """Render a compact human-readable usage table for stdout/logs."""
        snap = self.snapshot()
        total = snap["total"]
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("Token usage / 用量统计")
        lines.append("=" * 60)

        header = f"{'category':<14}{'requests':>10}{'input':>14}{'output':>14}{'total':>14}"
        lines.append(header)
        lines.append("-" * len(header))
        for cat in sorted(snap["by_category"].keys()):
            b = snap["by_category"][cat]
            lines.append(
                f"{cat:<14}{b['requests']:>10}{b['input_tokens']:>14,}"
                f"{b['output_tokens']:>14,}{b['total_tokens']:>14,}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'TOTAL':<14}{total['requests']:>10}{total['input_tokens']:>14,}"
            f"{total['output_tokens']:>14,}{total['total_tokens']:>14,}"
        )
        if total["unknown_usage"]:
            lines.append(
                f"  note: {total['unknown_usage']} request(s) returned no usage "
                f"payload; token totals are a lower bound."
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def _get(obj: Any, name: str) -> Any:
    """Attribute-or-key getter for SDK pydantic models and raw dicts."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_model(response: Any) -> str | None:
    m = _get(response, "model")
    return str(m) if m else None


def extract_usage(response: Any) -> tuple[int, int, bool]:
    """Normalize a provider response into ``(input_tokens, output_tokens, had_usage)``.

    Handled shapes (in priority order):
      * OpenAI chat-completions: ``usage.prompt_tokens`` / ``usage.completion_tokens``
      * Anthropic + OpenAI Responses API: ``usage.input_tokens`` / ``usage.output_tokens``
      * dicts with either of the above key sets.

    Returns ``(0, 0, False)`` when no usable token counts are present.
    """
    usage = _get(response, "usage")
    if usage is None:
        return 0, 0, False

    prompt = _get(usage, "prompt_tokens")
    completion = _get(usage, "completion_tokens")
    input_t = _get(usage, "input_tokens")
    output_t = _get(usage, "output_tokens")

    in_val: int | None = None
    out_val: int | None = None

    if prompt is not None or completion is not None:
        in_val = _coerce_int(prompt)
        out_val = _coerce_int(completion)
    if in_val is None and input_t is not None:
        in_val = _coerce_int(input_t)
    if out_val is None and output_t is not None:
        out_val = _coerce_int(output_t)

    had_usage = in_val is not None or out_val is not None
    return (in_val or 0), (out_val or 0), had_usage


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ── module-level singleton ──────────────────────────────────────────
_TRACKER = UsageTracker()


def get_tracker() -> UsageTracker:
    """Return the process-wide usage tracker."""
    return _TRACKER


def record_response(category: str, response: Any, *, model: str | None = None) -> None:
    """Convenience wrapper recording into the global tracker. Never raises."""
    try:
        _TRACKER.record_response(category, response, model=model)
    except Exception:  # noqa: BLE001 — usage tracking must never break a run
        pass


def record(
    category: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str | None = None,
    had_usage: bool = True,
) -> None:
    """Convenience wrapper recording raw counts into the global tracker."""
    try:
        _TRACKER.record(
            category,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            had_usage=had_usage,
        )
    except Exception:  # noqa: BLE001
        pass
