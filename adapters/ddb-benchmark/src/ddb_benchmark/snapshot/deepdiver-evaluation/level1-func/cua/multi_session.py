"""Shared multi-session (viewer subagent) + veracity plumbing for all runners.

Extracted from ``CUARunner`` so EVERY protocol adapter can:

- spawn an isolated *viewer subagent* via the ``delegate_to_viewer`` tool for
  cross-identity claims (realtime sync / shared read-write / per-identity
  isolation), and
- look up real-world ground truth via the ``search_web`` tool for data-bearing
  claims (veracity check).

The viewer subagent is a fresh runner of the SAME class as the caller
(``type(self)``), so it speaks the same protocol/model/gateway. Its config
disables multi-session so it can never recurse. The isolated browser session
(fresh cookies/localStorage/anonymous identity) comes from
``PlaywrightEnv.spawn_isolated_session`` — protocol-agnostic.

Concrete runners are expected to expose ``self._cfg`` (RunnerConfig) and
``self._env`` (PlaywrightEnv). They call ``self._ms_reset(...)`` at the top of
``verify_claim`` to arm per-claim state, then dispatch ``delegate_to_viewer`` /
``search_web`` tool calls to ``self._delegate_to_viewer`` / ``self._run_search``.
"""
from __future__ import annotations
import dataclasses
import logging
from typing import Any
from types_.errors import QuotaExhaustedError
from .tools.search import run_search as _run_search_impl

logger = logging.getLogger(__name__)


# ── Prompt fragments ────────────────────────────────────────────

def multi_session_block(participants: list[str] | None, marker: str) -> str:
    """Multi-session collaboration protocol appended to the goal for
    ``multi_session`` claims. Teaches the CUA to bring in a second, truly
    separate identity via ``delegate_to_viewer`` and pins the unique marker
    both sides use. MUST NOT reveal the expected verdict.
    """
    who = ", ".join(participants or []) or "creator (you), viewer_a (the second person)"
    mark = marker or "<MARK>"
    return (
        "\n\n## Multi-session collaboration (this claim needs TWO identities)\n"
        f"Participants for this claim: {who}. You are the creator/first "
        "identity and stay in THIS session/room the whole time.\n"
        f"Unique marker for this claim: `{mark}` — use it verbatim in any "
        "content written across sessions so it cannot be confused with the "
        "app's own sample data or the other person's data.\n"
        "You must NOT try to be two people in one session, and you do NOT need "
        "to find or read any share/invite link — the harness already holds a "
        "pre-provisioned identity link for the second person and opens it "
        "automatically in an isolated session. To bring in the second person, "
        "call the `delegate_to_viewer` tool:\n"
        "  1. In `instruction`, tell the second person exactly ONE concrete "
        "action to perform (the tool auto-appends the marker), e.g. \"send a "
        f"chat comment containing {mark}\" or \"open your own settings page "
        f"and report whether you can see {mark}\".\n"
        "  2. The tool opens the second person's session on their own identity "
        "link, runs the action, returns their observation, and closes their "
        "session automatically.\n"
        "Decide the verdict from what YOUR OWN session shows:\n"
        "  - Realtime / shared (A observes what B wrote): after delegating, "
        "look at your own screen (a short `wait` or one `reload` is fine) and "
        f"check whether B's `{mark}` content appears here. supported only if "
        "it does.\n"
        "  - Per-identity isolation (private data must NOT leak): YOU write "
        f"the private `{mark}` data first, then delegate to B asking whether "
        "they can see it; supported only if B reports they CANNOT see or edit "
        "your marked data.\n"
    )


def veracity_block() -> str:
    """Ground-truth / veracity protocol appended to the goal for data-bearing
    claims. Teaches the CUA to verify the app's real-world data against the
    current truth via ``search_web``. MUST NOT reveal the expected verdict.
    """
    return (
        "\n\n## Real-world data veracity (this claim shows real-world data)\n"
        "This app is required to display REAL, live data from real-world APIs "
        "(sports fixtures/scores, exchange rates, weather, news, prices, "
        "schedules, etc.). Merely rendering some value is NOT enough — the "
        "value must actually match reality and be current, not stale, "
        "fabricated, or mock.\n"
        "Protocol:\n"
        "  1. Read the concrete real-world value(s) the page shows.\n"
        "  2. Call `search_web` with a query for the CURRENT ground truth of "
        "that value (search the fact, not the app).\n"
        "  3. Compare: if the page clearly disagrees with the real-world truth "
        "(e.g. shows a long-past fixture as \"current\", a wrong score, an "
        "obviously stale/placeholder value), the claim is `contradicted`.\n"
        "  4. If they match, that supports the claim. If `search_web` is "
        "unavailable or inconclusive, decide from on-page evidence and do NOT "
        "mark the claim `contradicted` solely because you could not verify.\n"
    )


# ── Viewer subagent fragments ───────────────────────────────────

_VIEWER_CHECKPOINT_TEMPLATE = "[Viewer subagent task] {instruction}"
_VIEWER_VERIFICATION_TEMPLATE = (
    "You are viewer B — a SECOND, independent person who just opened this app "
    "in a fresh session (you do NOT share the first person's login, cookies, "
    "or data). Carry out the task in the Claim above using ONLY normal UI "
    "actions.{marker_clause} When you are done — whether or not it worked — "
    "call `report_verdict` EXACTLY ONCE with verdict=\"supported\" if you "
    "completed the action (\"insufficient_evidence\" if you could not), and "
    "put a concise factual account of WHAT YOU DID and WHAT YOU OBSERVED "
    "(including whether the marker was visible to you) in `reasoning`. That "
    "text is sent back to the first person, so be specific and do not guess "
    "beyond what you saw."
)


class MultiSessionMixin:
    """Mixin providing viewer-subagent + veracity handlers to any runner.

    Requires the host class to define ``self._cfg`` and ``self._env``.
    """

    # ── per-claim state ─────────────────────────────────────────
    def _ms_reset(
        self,
        *,
        query_id: str = "",
        claim_id: str = "",
        app_url: str = "",
        marker: str = "",
        participants: list[str] | None = None,
        session_urls: dict[str, str] | None = None,
    ) -> None:
        self._active_query_id = query_id
        self._active_claim_id = claim_id
        self._active_app_url = app_url
        self._active_marker = marker or ""
        self._active_participants = list(participants or [])
        self._active_session_urls = dict(session_urls or {})
        self._viewer_count = 0

    def _ms_armed(self, multi_session: bool) -> bool:
        """Whether the viewer-subagent protocol is active for this claim."""
        return bool(multi_session) and bool(
            getattr(self._cfg, "enable_multi_session", False)
        )

    def _veracity_armed(self, veracity_check: bool) -> bool:
        """Whether the search_web veracity tool is active for this claim."""
        return bool(veracity_check) and bool(
            getattr(self._cfg, "enable_veracity_check", False)
        )

    # ── veracity tool ───────────────────────────────────────────
    async def _run_search(self, args: dict[str, Any]) -> dict[str, Any]:
        return await _run_search_impl(self._cfg, args)

    # ── viewer subagent ─────────────────────────────────────────
    def _make_viewer_config(self):
        """Derive the viewer subagent's config from the main config.

        The viewer is a genuine second runner loop, but with a small budget,
        an optionally cheaper model, and — critically — multi-session DISABLED
        so it can never receive ``delegate_to_viewer`` and recurse.
        """
        cfg = self._cfg
        return dataclasses.replace(
            cfg,
            model=cfg.viewer_model or cfg.model,
            max_iterations=max(1, cfg.viewer_max_iterations),
            max_iterations_by_claim_type={},
            enable_multi_session=False,
        )

    def _new_viewer_runner(self, session_env, viewer_cfg):
        """Instantiate a viewer runner of the SAME class/protocol as self.

        All five runner constructors accept ``(env=..., config=...)`` with
        their remaining args optional, so this works uniformly.
        """
        return type(self)(env=session_env, config=viewer_cfg)

    async def _delegate_to_viewer(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handler for the ``delegate_to_viewer`` tool.

        Spawns an isolated second session, runs a fresh runner (same protocol)
        as viewer B to perform ONE delegated action, closes the session, and
        returns viewer B's observation to the main CUA. Recursion is
        impossible: the viewer's config disables multi-session.
        """
        def _fail(msg: str) -> dict[str, Any]:
            return {
                "success": False, "error": msg,
                "text": msg, "image_b64": None, "clamped": False,
            }

        instruction = str(args.get("instruction") or "").strip()
        if not instruction:
            return _fail("delegate_to_viewer requires a non-empty 'instruction'")
        if self._viewer_count >= max(1, self._cfg.max_viewers_per_claim):
            return _fail(
                f"viewer budget exhausted (max_viewers_per_claim="
                f"{self._cfg.max_viewers_per_claim}); decide the verdict from "
                f"what your own session already shows."
            )

        viewer_idx = self._viewer_count + 1
        # Resolve the second person's identity via the pre-provisioned link,
        # NOT by reading a share link out of the app UI. The Nth viewer maps
        # to the Nth non-creator participant declared on the claim.
        viewer_roles = [
            str(p).strip()
            for p in self._active_participants
            if str(p).strip() and str(p).strip().lower() != "creator"
        ]
        role = (
            viewer_roles[self._viewer_count]
            if self._viewer_count < len(viewer_roles)
            else (viewer_roles[-1] if viewer_roles else "")
        )
        target_url = str(self._active_session_urls.get(role, "") or "").strip()
        if not target_url:
            return _fail(
                "未提供共享链接，无法进行多人协同评测。"
                f"（缺少参与者 '{role or 'viewer'}' 的 session_urls 链接）"
            )
        marker = self._active_marker

        if not hasattr(self._env, "spawn_isolated_session"):
            return _fail(
                "multi-session not supported by the current browser env; "
                "cannot spawn a viewer."
            )

        session = None
        try:
            session = await self._env.spawn_isolated_session(target_url)
        except Exception as e:  # noqa: BLE001
            logger.warning("spawn_isolated_session failed: %s", e)
            return _fail(f"failed to open a viewer session: {type(e).__name__}: {e}")

        try:
            viewer_runner = self._new_viewer_runner(session, self._make_viewer_config())
            marker_clause = (
                f" If you create any content, include the exact marker "
                f"'{marker}' verbatim so it can be recognised."
                if marker else ""
            )
            verification = _VIEWER_VERIFICATION_TEMPLATE.format(
                marker_clause=marker_clause
            )
            viewer_result = await viewer_runner.verify_claim(
                claim_id=f"{self._active_claim_id}_viewer{viewer_idx}",
                checkpoint=_VIEWER_CHECKPOINT_TEMPLATE.format(instruction=instruction),
                verification=verification,
                claim_type="state_transition",
                app_url=target_url,
                query_id=self._active_query_id,
                guideline="",
            )
            self._viewer_count = viewer_idx
            observation = (
                viewer_result.get("cua_reasoning")
                or viewer_result.get("reason")
                or ""
            ).strip()
            viewer_verdict = str(viewer_result.get("verdict") or "")
            summary = (
                f"viewer_{viewer_idx} finished ({viewer_verdict}). "
                f"observation: {observation or '(no observation reported)'}"
            )
            logger.info(
                "delegate_to_viewer: claim=%s viewer=%d verdict=%s",
                self._active_claim_id, viewer_idx, viewer_verdict,
            )
            return {
                "success": True,
                "error": None,
                "text": summary,
                "image_b64": None,
                "clamped": False,
                "observation": observation,
                "marker": marker,
            }
        except QuotaExhaustedError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("viewer subagent loop failed: %s", e)
            return _fail(f"viewer subagent failed: {type(e).__name__}: {e}")
        finally:
            if session is not None:
                try:
                    await self._env.close_session(session)
                except Exception as e:  # noqa: BLE001
                    logger.warning("failed to close viewer session: %s", e)
