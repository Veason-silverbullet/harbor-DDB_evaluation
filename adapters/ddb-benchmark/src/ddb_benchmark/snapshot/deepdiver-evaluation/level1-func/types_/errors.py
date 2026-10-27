"""
Custom exceptions for the evaluation pipeline.
"""
from __future__ import annotations
import re

_QUOTA_PATTERNS = [
    r"insufficient.?quota",
    r"rate.?limit",
    r"billing",
    r"exceeded.*quota",
    r"quota.*exceeded",
    r"credit",
    r"余额不足",
    r"额度",
    r"payment.?required",
    r"limit.*reached",
    r"too.?many.?requests",
    r"resource.?exhausted",
    r"out.?of.?credits",
    r"account.*deactivat",
    r"plan.*limit",
]

_QUOTA_RE = re.compile("|".join(_QUOTA_PATTERNS), re.IGNORECASE)

# 402 / 429 are unambiguous quota signals; 403 requires text confirmation
_QUOTA_HTTP_CODES_CERTAIN = {402, 429}
_QUOTA_HTTP_CODES_MAYBE = {403}


class QuotaExhaustedError(Exception):
    """Raised when an API call fails due to quota / rate-limit / billing issues."""

    def __init__(self, source: str, detail: str = "") -> None:
        self.source = source
        self.detail = detail
        super().__init__(f"[{source}] Quota exhausted: {detail}")


def _get_http_status(exc: BaseException) -> int | None:
    """Try to extract an HTTP status code from an exception."""
    for attr in ("status_code", "status"):
        val = getattr(exc, attr, None)
        if val is not None:
            return int(val)
    if hasattr(exc, "response"):
        resp = exc.response
        for attr in ("status_code", "status"):
            val = getattr(resp, attr, None)
            if val is not None:
                return int(val)
    return None


def is_quota_error(exc: BaseException) -> bool:
    """Heuristically decide whether *exc* signals a quota / billing problem."""
    msg = str(exc).lower()
    text_match = bool(_QUOTA_RE.search(msg))

    if text_match:
        return True

    status = _get_http_status(exc)
    if status in _QUOTA_HTTP_CODES_CERTAIN:
        return True
    # 403 is ambiguous (could be auth); only treat as quota if text also matches
    if status in _QUOTA_HTTP_CODES_MAYBE and text_match:
        return True

    return False
