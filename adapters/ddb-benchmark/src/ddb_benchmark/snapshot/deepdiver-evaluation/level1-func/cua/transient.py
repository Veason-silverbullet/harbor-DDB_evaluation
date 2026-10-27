"""Immediate (non-settling) screenshot capture for transient UI feedback.

Fast-disappearing UI — danmaku (弹幕), toasts, snackbars, flash confirmations,
optimistic vote/like bumps — is routinely MISSED by the normal capture path
because that path calls ``screenshot_bytes_stable`` which waits for the
viewport to visually settle: by the time the frame is stable, the transient
element has already flown off screen.

For claims that hinge on such feedback (``state_transition`` — the judge-side
claim_type covering 发弹幕 / 发评论 / 投票 / 点赞 etc.) we take ONE extra
screenshot IMMEDIATELY after the action batch, WITHOUT settling, and:
  * append it to ``screenshots`` so it enters the judge's start/mid/end
    sampling pool, and
  * hand its base64 back so the runner can also show it to the CUA in the
    next turn.

Shared by all five runners so the behaviour is identical across protocols.
"""
from __future__ import annotations
import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

# claim_types whose evidence is likely a transient/flash element. Kept narrow
# so we don't double the screenshot count for every claim. ``state_transition``
# is the judge-side type for functionality claims (send/post/vote/react).
TRANSIENT_CLAIM_TYPES: frozenset[str] = frozenset({"state_transition"})


def is_transient_claim(cfg: Any, claim_type: str) -> bool:
    return bool(getattr(cfg, "capture_transient_immediate", True)) and (
        claim_type in TRANSIENT_CLAIM_TYPES
    )


async def capture_transient_immediate(
    env: Any,
    cfg: Any,
    query_id: str,
    claim_id: str,
    step: int,
    screenshots: list[str],
    claim_type: str,
) -> str | None:
    """Take one immediate (non-settling) frame for transient-feedback claims.

    Returns the frame's base64 (so the runner can also feed it to the model),
    or ``None`` when not eligible / on failure. Never raises.
    """
    if not is_transient_claim(cfg, claim_type):
        return None
    try:
        raw = await env.screenshot_bytes()  # deliberately NO settle
    except Exception as e:  # noqa: BLE001
        logger.debug("transient immediate capture failed: %s", e)
        return None
    if getattr(cfg, "save_screenshots", True):
        try:
            path = (
                cfg.screenshot_dir
                / query_id
                / "screenshots"
                / f"{claim_id}_step{step}_imm.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            screenshots.append(str(path))
        except Exception as e:  # noqa: BLE001
            logger.debug("failed to persist transient frame: %s", e)
    return base64.b64encode(raw).decode("utf-8")
