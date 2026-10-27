"""
Claim verifier: routes each claim through CUA + programmatic/LLM judge pipeline.

For each claim:
1. CUA agent interacts with the app to collect evidence.
2. Page-load-failure detector inspects ``after_info`` + ``runtime_signals``
   to decide whether the page itself failed to boot. If yes, the
   orchestrator's resilience layer kicks in: ``env.reload()`` and retry the
   whole CUA+judge pipeline (G2). After ``claim_retry_on_page_failure``
   reloads without recovery we fall through to the J1 branch which forces
   ``contradicted`` on presence/visual_semantics/interaction_* claims and
   keeps ``insufficient_evidence`` for the rest, and skips the LLM judge to
   save quota.
3. Otherwise programmatic validators do a first-pass check.
4. If programmatic check is insufficient, LLM judge provides final verdict.
5. ``_merge_verdicts`` (J3) applies sanity caps before producing the
   final ``ClaimResult``.
"""
from __future__ import annotations
import logging
import secrets
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from cua.cua_agent import CUAAgent
from cua.playwright_env import PlaywrightEnv
from types_.claims import EvalClaim
from types_.results import ClaimResult
from config.settings import EvalConfig
from . import programmatic_validators as prog
from . import llm_judge

logger = logging.getLogger(__name__)

# ── Two-axis taxonomy (claim_type = verification method, 7 values) ──
# Programmatic-first methods: each has a dedicated programmatic validator
# in ``programmatic_validators`` (structural/navigation/state_transition/
# value_correctness/runtime_integrity) or a reload-protocol comparison
# (persistence). They may still defer to the LLM judge via ``needs_llm``.
PROGRAMMATIC_FIRST_TYPES = {
    "structural",
    "navigation",
    "state_transition",
    "value_correctness",
    "persistence",
    "runtime_integrity",
}

# Methods that always need LLM/VLM (no deterministic shortcut). ``perceptual``
# covers visual semantics / layout / color / animation (+ geo sub-check).
LLM_FIRST_TYPES = {"perceptual"}

# J1 cascade: methods we can confidently flip to ``contradicted`` when the
# page itself failed to load. These fundamentally assert that a UI element /
# view is observably present, so a dead page is a direct counter-example.
# The remaining methods (value_correctness, persistence, runtime_integrity)
# cannot be evaluated when the app didn't boot, so they stay
# ``insufficient_evidence``.
J1_CONTRADICTED_TYPES = {
    "structural",
    "perceptual",
    "navigation",
}

VERACITY_SEARCH_TOOL = "search_web"


def _gen_marker() -> str:
    """One-off unique marker for a multi-session claim (e.g. ``M-7A3F9C``).

    Short, easy for the CUA to type verbatim, and random enough not to
    collide with the app's own seed content or the other identity's data.
    Shared by the main goal and the viewer subagent for a single claim so
    both write/read the SAME token.
    """
    return "M-" + secrets.token_hex(3).upper()


class ClaimVerifier:
    """
    Verifies a single claim using CUA interaction + programmatic/LLM judging.
    """

    def __init__(
        self,
        cua_agent: CUAAgent,
        env: PlaywrightEnv,
        config: EvalConfig,
    ) -> None:
        self._cua = cua_agent
        self._env = env
        self._config = config

    async def verify(
        self,
        claim: EvalClaim,
        app_url: str,
        query_id: str = "",
        session_urls: dict[str, str] | None = None,
    ) -> ClaimResult:
        """Timed wrapper around :meth:`_verify_impl`.

        Attaches wall-clock ``timing`` to the result so serial-vs-parallel
        latency can be decomposed per claim from the saved JSON alone.
        """
        started = datetime.now()
        t0 = time.perf_counter()
        result = await self._verify_impl(
            claim, app_url, query_id=query_id, session_urls=session_urls,
        )
        result.timing = {
            "started_at": started.isoformat(timespec="milliseconds"),
            "ended_at": datetime.now().isoformat(timespec="milliseconds"),
            "duration_s": round(time.perf_counter() - t0, 2),
        }
        return result

    async def _verify_impl(
        self,
        claim: EvalClaim,
        app_url: str,
        query_id: str = "",
        session_urls: dict[str, str] | None = None,
    ) -> ClaimResult:
        """
        Full verification pipeline for one claim.

        Loop:
        1. Run CUA + judges once.
        2. Detect ``page_load_failed`` (blank visible_text + main-resource
           4xx/5xx + console pageerror). If ok, return.
        3. If failed and we still have ``claim_retry_on_page_failure`` budget,
           reload the page and retry.
        4. If the budget is exhausted, return a synthetic verdict via the
           J1 cascade and skip the LLM judge.
        """
        res_cfg = self._config.resilience
        max_retries = max(0, getattr(res_cfg, "claim_retry_on_page_failure", 3))

        attempts_log: list[dict[str, Any]] = []
        last_result_payload: dict[str, Any] | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                # Reload + warm-up before the next attempt. ``env.reload``
                # already swallows + logs its own errors so we trust the
                # boolean return rather than wrapping it again here.
                logger.warning(
                    "claim %s page_load_failed -> reload & retry %d/%d",
                    claim.id, attempt, max_retries,
                )
                ok = await self._env.reload(retries=1)
                if not ok:
                    logger.warning(
                        "claim %s: env.reload returned False on retry %d; "
                        "continuing with current page state",
                        claim.id, attempt,
                    )
                await self._env.wait(getattr(res_cfg, "retry_backoff_ms", 800))

            payload = await self._run_pipeline_once(
                claim, app_url, query_id, session_urls or {},
            )
            last_result_payload = payload

            page_load_failed, fail_reasons = self._detect_page_load_failure(
                payload["after_info"], payload["runtime_signals"],
                payload["cua_result"],
            )
            attempts_log.append({
                "attempt": attempt,
                "page_load_failed": page_load_failed,
                "reasons": fail_reasons,
            })

            if not page_load_failed:
                payload["cua_result"]["page_load_failure"] = {
                    "triggered": False,
                    "attempts": attempt + 1,
                    "history": attempts_log,
                    "final_reasons": [],
                }
                return self._build_result(
                    claim=claim,
                    query_id=query_id,
                    verdict_data=payload["verdict_data"],
                    cua_result=payload["cua_result"],
                    used_validators=payload["used_validators"],
                )

            if attempt >= max_retries:
                # G2 budget exhausted -> J1 cascade.
                forced = self._force_page_load_failure_verdict(
                    claim, fail_reasons, attempt + 1,
                )
                used_validators = list(payload["used_validators"])
                used_validators.append("page_load_failure_cascade")
                payload["cua_result"]["page_load_failure"] = {
                    "triggered": True,
                    "attempts": attempt + 1,
                    "history": attempts_log,
                    "final_reasons": fail_reasons,
                    "forced_verdict": forced["verdict"],
                }
                logger.error(
                    "claim %s: page_load_failure cascade triggered after %d "
                    "attempts (reasons=%s) -> forced verdict %s",
                    claim.id, attempt + 1, fail_reasons, forced["verdict"],
                )
                return self._build_result(
                    claim=claim,
                    query_id=query_id,
                    verdict_data=forced,
                    cua_result=payload["cua_result"],
                    used_validators=used_validators,
                )

        # Defensive: loop body always returns. We only get here if
        # max_retries is somehow negative.
        assert last_result_payload is not None
        return self._build_result(
            claim=claim,
            query_id=query_id,
            verdict_data=last_result_payload["verdict_data"],
            cua_result=last_result_payload["cua_result"],
            used_validators=last_result_payload["used_validators"],
        )

    async def _run_pipeline_once(
        self,
        claim: EvalClaim,
        app_url: str,
        query_id: str,
        session_urls: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run CUA + judges once and return all intermediate artifacts.

        Wrapped by :meth:`verify` which adds page-load-failure detection +
        reload-retry on top.
        """
        before_info = await self._env.get_page_info()

        try:
            self._env.reset_runtime_log()
        except AttributeError:
            pass

        # ── Multi-session arming ────────────────────────────────────
        # A claim runs the viewer-subagent protocol only when it is flagged
        # ``multi_session`` AND the feature is enabled in config. We mint a
        # fresh unique marker here and thread it to the CUA so the main goal
        # and any viewer subagent share the exact same token.
        claim_multi_session = bool(getattr(claim, "multi_session", False))
        participants = list(getattr(claim, "participants", []) or [])
        ms_active = claim_multi_session and bool(
            getattr(self._config.cua, "enable_multi_session", False)
        )
        marker = _gen_marker() if ms_active else ""
        if ms_active:
            logger.info(
                "claim %s: multi-session armed (participants=%s, marker=%s)",
                claim.id, participants, marker,
            )

        # ── Veracity arming (data-bearing claims only) ──────────────
        # The query dataset assumes every displayed data value comes from a
        # real-world API, so we check real-world truth WITHOUT needing any
        # per-query annotation: currently limited to Axis-2 semantic_type
        # ``data_integration``. Gated by ``enable_veracity_check`` so it is
        # off unless explicitly turned on in config.
        semantic_type = str(getattr(claim, "semantic_type", "") or "").lower()
        veracity_check = bool(
            getattr(self._config.cua, "enable_veracity_check", False)
        ) and (
            semantic_type in ("data_integration",)
        )
        if veracity_check:
            logger.info(
                "claim %s: veracity check armed (claim_type=%s semantic_type=%s)",
                claim.id, claim.claim_type, semantic_type,
            )

        cua_result = await self._cua.verify_claim(
            claim_id=claim.id,
            checkpoint=claim.checkpoint,
            verification=claim.verification,
            claim_type=claim.claim_type,
            app_url=app_url,
            query_id=query_id,
            guideline=getattr(claim, "verification_guideline", "") or "",
            multi_session=ms_active,
            participants=participants,
            marker=marker,
            session_urls=session_urls or {},
            veracity_check=veracity_check,
        )

        after_info = await self._env.get_page_info()

        # Final TTS drain before the snapshot: refreshes started/ended flags
        # on already-ingested utterances and catches speak() calls the
        # per-action drain missed (e.g. native adapter paths).
        try:
            await self._env.drain_tts_events()
        except AttributeError:
            pass

        runtime_signals: dict[str, Any] = {}
        try:
            runtime_signals = self._env.runtime_signals_snapshot()
        except AttributeError:
            runtime_signals = {}
        cua_result["runtime_signals"] = runtime_signals

        if veracity_check and not self._has_veracity_search(cua_result):
            logger.warning(
                "claim %s: veracity check was armed but no %s call was made; "
                "forcing insufficient_evidence",
                claim.id,
                VERACITY_SEARCH_TOOL,
            )
            forced = self._force_missing_veracity_search_verdict(claim)
            cua_result["veracity_check"] = {
                "armed": True,
                "search_performed": False,
                "forced_verdict": forced["verdict"],
            }
            return {
                "before_info": before_info,
                "after_info": after_info,
                "runtime_signals": runtime_signals,
                "cua_result": cua_result,
                "used_validators": ["cua", "veracity_search_gate"],
                "verdict_data": forced,
            }
        if veracity_check:
            cua_result["veracity_check"] = {
                "armed": True,
                "search_performed": True,
            }

        if (
            claim.claim_type == "perceptual"
            and prog.is_map_geo_claim(claim.checkpoint, claim.verification)
        ):
            try:
                geometry = await self._env.extract_map_geometry()
            except AttributeError:
                geometry = {}
            except Exception as e:  # noqa: BLE001
                logger.warning("extract_map_geometry failed: %s", e)
                geometry = {}
            cua_result["map_geometry"] = geometry

        # Persistence reload protocol: for persistence claims, capture the
        # current (post-CUA) state, reload the page, then capture again so
        # ``validate_persistence`` can compare before/after the reload.
        if claim.claim_type == "persistence":
            persistence_states: dict[str, Any] = {"performed": False}
            try:
                before_reload = {
                    "visible_text": after_info.get("visible_text", ""),
                    "page_url": after_info.get("page_url", ""),
                    "page_title": after_info.get("page_title", ""),
                }
                await self._env.reload()
                await self._env.wait(800)
                after_reload_info = await self._env.get_page_info()
                persistence_states = {
                    "performed": True,
                    "before_reload": before_reload,
                    "after_reload": {
                        "visible_text": after_reload_info.get("visible_text", ""),
                        "page_url": after_reload_info.get("page_url", ""),
                        "page_title": after_reload_info.get("page_title", ""),
                    },
                }
            except AttributeError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.warning("persistence reload protocol failed: %s", e)
            cua_result["persistence_states"] = persistence_states

        route = self._route(claim)
        used_validators = ["cua"]
        verdict_data: dict[str, Any] | None = None

        if route == "programmatic_first":
            prog_result = self._run_programmatic(
                claim, before_info, after_info, cua_result
            )
            used_validators.append(prog_result.get("validator", "programmatic"))

            # ── v2 reverse gate: prog says supported, CUA says contradicted ──
            # The dominant inflation pattern in S1-001 was the
            # ``validate_action_effect`` shortcut declaring ``supported``
            # because the page text changed (clicking a heart icon
            # navigated the user to the wishlist page) while CUA's own
            # ``report_verdict`` had clearly observed that the wishlist
            # was empty and declared ``contradicted@0.85``. Whenever
            # programmatic and CUA disagree like this, the programmatic
            # signal alone is not strong enough — force the LLM judge
            # to break the tie with screenshots in hand.
            cua_verdict = str(cua_result.get("verdict") or "")
            try:
                cua_conf = float(cua_result.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                cua_conf = 0.0
            prog_verdict = str(prog_result.get("verdict") or "")
            if (
                prog_verdict == "supported"
                and cua_verdict == "contradicted"
                and cua_conf >= 0.7
            ):
                logger.info(
                    "claim %s: prog=supported but CUA=contradicted@%.2f; "
                    "deferring to LLM judge (v2 reverse gate)",
                    claim.id, cua_conf,
                )
                prog_result["needs_llm"] = True
                prog_result.setdefault("sanity_notes", []).append(
                    f"prog_supported_but_cua_contradicted({cua_conf:.2f})"
                )

            if not prog_result.get("needs_llm", True):
                verdict_data = prog_result

        if verdict_data is None and (
            claim.claim_type == "perceptual"
            and "map_geometry" in cua_result
            and prog.is_map_geo_claim(claim.checkpoint, claim.verification)
        ):
            geo_result = prog.validate_geo_semantics(
                cua_result.get("map_geometry"),
                claim.checkpoint,
                claim.verification,
            )
            geo_result["validator"] = "geo_semantics_validator"
            used_validators.append("geo_semantics_validator")
            cua_result["geo_validation"] = geo_result
            if (
                not geo_result.get("needs_llm", True)
                and geo_result.get("evidence_sufficient", False)
                and float(geo_result.get("confidence", 0.0)) >= 0.8
            ):
                verdict_data = geo_result

        if verdict_data is None:
            cua_verdict = cua_result.get("verdict", "insufficient_evidence")
            cua_confidence = cua_result.get("confidence", 0.0)
            if cua_confidence >= 0.7 and cua_verdict in ("supported", "contradicted"):
                # Conservative gate (added 2026-05-21): a `contradicted`
                # verdict with ZERO real CUA interaction is highly
                # suspicious. The dominant failure mode we've seen is
                # CUA opening the page, seeing a well-formed empty-state
                # placeholder ("还没有专注记录", "no records yet"), and
                # mis-reading that as evidence the feature is missing
                # — when in fact the feature's container IS rendered
                # and just lacks user data. Force fallback to the LLM
                # judge so it can apply the empty-state rule in
                # verify_claim.txt instead of letting the shortcut
                # cement a false contradiction.
                #
                # `supported` shortcuts are left untouched: a presence
                # claim whose target UI is right there on the first
                # screenshot is the canonical fast path and works
                # fine.
                real_action_steps = sum(
                    1 for a in cua_result.get("action_trace", [])
                    if a.get("tool") not in (
                        "report_verdict",
                        "__screenshot__",
                        "__cursor__",
                        "__unknown__",
                    )
                )
                if cua_verdict == "contradicted" and real_action_steps == 0:
                    logger.info(
                        "claim %s: skip cua_verdict shortcut "
                        "(contradicted with no interaction); deferring "
                        "to LLM judge for empty-state check",
                        claim.id,
                    )
                else:
                    used_validators.append("cua_verdict")
                    verdict_data = cua_result

        if verdict_data is None:
            llm_result = await self._run_llm_judge(claim, after_info, cua_result)
            used_validators.append("llm_judge")
            verdict_data = self._merge_verdicts(cua_result, llm_result)

        return {
            "before_info": before_info,
            "after_info": after_info,
            "runtime_signals": runtime_signals,
            "cua_result": cua_result,
            "used_validators": used_validators,
            "verdict_data": verdict_data,
        }

    def _has_veracity_search(self, cua_result: dict[str, Any]) -> bool:
        """Return True only when the CUA actually called the ground-truth tool."""
        return any(
            a.get("tool") == VERACITY_SEARCH_TOOL
            for a in (cua_result.get("action_trace") or [])
        )

    def _force_missing_veracity_search_verdict(
        self,
        claim: EvalClaim,
    ) -> dict[str, Any]:
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.35,
            "reason": (
                "[veracity_search_gate] Real-world data veracity was required "
                f"for {claim.id}, but the CUA did not call "
                f"`{VERACITY_SEARCH_TOOL}`; on-page observations alone are "
                "not sufficient to support this claim."
            ),
            "evidence_sufficient": False,
            "validator": "veracity_search_gate",
        }

    def _detect_page_load_failure(
        self,
        after_info: dict[str, Any],
        runtime_signals: dict[str, Any],
        cua_result: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Heuristic detector: did the page actually load?

        Returns ``(failed, reasons)`` where ``reasons`` is a short list of
        machine-readable strings explaining the call. Multiple signals are
        OR'd: any single strong signal is enough to trigger a retry. We try
        to be conservative on false positives (don't reload pages that are
        merely sparse but functional), so the visible-text threshold is
        deliberately tiny.
        """
        res_cfg = self._config.resilience
        min_text = getattr(res_cfg, "page_load_failure_visible_text_min", 20)
        min_4xx = getattr(res_cfg, "page_load_failure_main_resource_status_min", 400)

        reasons: list[str] = []

        # Signal 1: visible_text is empty / near-empty
        raw_len = int(after_info.get("visible_text_raw_len") or 0)
        visible_text = after_info.get("visible_text") or ""
        if not visible_text.strip() and raw_len < min_text:
            reasons.append(f"visible_text_empty(raw_len={raw_len})")

        # Signal 2: main-document host has 4xx/5xx responses
        page_url = after_info.get("page_url") or ""
        page_host = ""
        try:
            page_host = (urlparse(page_url).hostname or "").lower()
        except Exception:  # noqa: BLE001
            page_host = ""
        failed_reqs = runtime_signals.get("failed_requests") or []
        for fr in failed_reqs:
            status = fr.get("status")
            if not isinstance(status, int):
                continue
            if status < min_4xx:
                continue
            fr_host = (fr.get("host") or "").lower()
            if page_host and fr_host == page_host:
                reasons.append(
                    f"main_resource_{status}({fr.get('url', '')[:120]})"
                )
                break

        # Signal 3: CUA self-reported "blank" / "white screen" wording in its
        # text. The prompt teaches it this phrase explicitly in J1.
        reasoning = (cua_result.get("cua_reasoning") or "").lower()
        if any(
            tok in reasoning for tok in (
                "page_load_failed", "blank page", "white screen", "页面空白",
                "页面未加载", "页面加载失败",
            )
        ):
            reasons.append("cua_reported_blank")

        # Signal 4: navigation errors stack from playwright (rare on a healthy
        # page since they would have been caught at goto-time, but they do
        # appear when the in-page router throws).
        for ce in runtime_signals.get("console_errors") or []:
            text = (ce.get("text") or "").lower()
            if any(
                tok in text for tok in (
                    "failed to fetch", "net::err_", "errnotfound",
                    "chunk load failed", "loading chunk", "navigation failed",
                )
            ):
                reasons.append(f"console_{(ce.get('type') or 'error')}:{text[:80]}")
                break

        return (bool(reasons), reasons)

    def _force_page_load_failure_verdict(
        self,
        claim: EvalClaim,
        fail_reasons: list[str],
        attempts: int,
    ) -> dict[str, Any]:
        """J1 cascade: produce the synthetic verdict when retries are spent.

        Presence-like claims become ``contradicted`` (the page is dead, so
        the asserted UI element is observably *not* there). Other claim
        types stay as ``insufficient_evidence`` (we couldn't exercise the
        feature at all, so we don't pretend to have evidence either way).
        """
        joined_reasons = ", ".join(fail_reasons) or "page never loaded"
        if claim.claim_type in J1_CONTRADICTED_TYPES:
            return {
                "verdict": "contradicted",
                "confidence": 0.9,
                "reason": (
                    f"[page_load_failure_cascade] Page failed to load after "
                    f"{attempts} attempts ({joined_reasons}); a presence/"
                    f"interaction claim cannot be satisfied by a dead page."
                ),
                "evidence_sufficient": True,
                "validator": "page_load_failure_cascade",
            }
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.3,
            "reason": (
                f"[page_load_failure_cascade] Page failed to load after "
                f"{attempts} attempts ({joined_reasons}); cannot evaluate "
                f"{claim.claim_type} without a working page."
            ),
            "evidence_sufficient": False,
            "validator": "page_load_failure_cascade",
        }

    def _route(self, claim: EvalClaim) -> str:
        if claim.claim_type in LLM_FIRST_TYPES:
            return "llm_first"
        return "programmatic_first"

    def _run_programmatic(
        self,
        claim: EvalClaim,
        before_info: dict[str, Any],
        after_info: dict[str, Any],
        cua_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the programmatic validator for the claim's verification method.

        Dispatch is keyed on ``claim_type`` (Axis 1). ``value_correctness`` and
        ``runtime_integrity`` additionally use ``semantic_type`` (Axis 2) for
        sub-dispatch (e.g. data_integration vs runtime_health vs accessibility).
        """
        ct = claim.claim_type
        semantic_type = getattr(claim, "semantic_type", None)
        action_trace = cua_result.get("action_trace", []) or []
        runtime_signals = cua_result.get("runtime_signals", {}) or {}

        if ct == "structural":
            result = prog.validate_structural(after_info, claim.checkpoint, claim.verification)
            result["validator"] = "structural_validator"
            return result
        elif ct == "navigation":
            result = prog.validate_navigation(before_info, after_info, action_trace)
            result["validator"] = "navigation_validator"
            return result
        elif ct == "state_transition":
            result = prog.validate_state_transition(
                before_info, after_info, action_trace,
                claim.checkpoint, claim.verification,
            )
            result["validator"] = "state_transition_validator"
            return result
        elif ct == "value_correctness":
            result = prog.validate_value_correctness(
                after_info, claim.checkpoint, claim.verification,
                semantic_type=semantic_type,
            )
            result["validator"] = "value_correctness_validator"
            return result
        elif ct == "runtime_integrity":
            result = prog.validate_runtime_integrity(
                runtime_signals, action_trace,
                claim.checkpoint, claim.verification,
                semantic_type=semantic_type,
            )
            result["validator"] = "runtime_integrity_validator"
            return result
        elif ct == "persistence":
            result = prog.validate_persistence(
                cua_result.get("persistence_states"),
                claim.checkpoint, claim.verification,
            )
            result["validator"] = "persistence_validator"
            return result
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.3,
            "reason": f"No programmatic validator for {ct}",
            "evidence_sufficient": False,
            "needs_llm": True,
            "validator": "none",
        }

    async def _run_llm_judge(
        self,
        claim: EvalClaim,
        page_info: dict[str, Any],
        cua_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Call the LLM judge with all collected evidence.

        Picks up to ``_JUDGE_MAX_SCREENSHOTS`` time-ordered screenshots
        (start / mid / end) instead of only the last one. This is
        critical for ``visual_semantics`` claims with time-series
        semantics (countdown rings, progress bars, animations,
        sequential UI states): a single final frame cannot prove a
        directional change, so the previous "last-frame only" payload
        forced the judge into ``insufficient_evidence`` even when the
        trace obviously contained the answer.
        """
        cfg = self._config.llm_judge
        screenshot_b64_list = self._pick_judge_screenshots(
            cua_result.get("screenshots", []) or [],
        )

        cua_reasoning = cua_result.get("cua_reasoning", "")
        geo_val = cua_result.get("geo_validation")
        if isinstance(geo_val, dict):
            summary = geo_val.get("geometry_summary") or {}
            geo_block = (
                "\n\n[geo_semantics_validator]\n"
                f"verdict={geo_val.get('verdict')} "
                f"confidence={geo_val.get('confidence')}\n"
                f"reason={geo_val.get('reason')}\n"
                f"summary={summary}"
            )
            cua_reasoning = (cua_reasoning or "") + geo_block

        return await llm_judge.judge_claim(
            checkpoint=claim.checkpoint,
            verification=claim.verification,
            claim_type=claim.claim_type,
            semantic_type=getattr(claim, "semantic_type", None),
            criticality=claim.criticality,
            page_info=page_info,
            cua_reasoning=cua_reasoning,
            action_trace=cua_result.get("action_trace", []),
            runtime_signals=cua_result.get("runtime_signals") or {},
            screenshot_b64_list=screenshot_b64_list,
            action_summary=cua_result.get("action_summary") or {},
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            thinking=getattr(cfg, "thinking", False),
        )

    # Cap on how many screenshots we forward to the LLM judge in a
    # single call. 3 is enough for first/mid/last comparison without
    # blowing up the request payload or the model's vision context.
    _JUDGE_MAX_SCREENSHOTS = 3

    @classmethod
    def _pick_judge_screenshots(cls, screenshots: list[str]) -> list[str]:
        """Pick up to ``_JUDGE_MAX_SCREENSHOTS`` representative frames.

        - 0 frames: returns ``[]``.
        - 1 frame: returns ``[frame0]``.
        - 2 frames: returns ``[frame0, frameN]`` (start + end).
        - >=3 frames: returns ``[frame0, middle, frameN]`` — the start
          / mid / end triple is what lets the judge see whether the
          UI actually changed over the CUA interaction window.

        Frames that fail to load from disk are silently skipped; if
        every pick fails we return an empty list (no screenshot sent).
        """
        from pathlib import Path
        import base64

        if not screenshots:
            return []

        n = len(screenshots)
        if n == 1:
            picks = [screenshots[0]]
        elif n == 2:
            picks = [screenshots[0], screenshots[-1]]
        else:
            picks = [
                screenshots[0],
                screenshots[n // 2],
                screenshots[-1],
            ]

        out: list[str] = []
        for ss in picks[: cls._JUDGE_MAX_SCREENSHOTS]:
            try:
                ss_bytes = Path(ss).read_bytes()
            except Exception as e:  # noqa: BLE001
                logger.warning("judge screenshot %s unreadable: %s", ss, e)
                continue
            out.append(base64.b64encode(ss_bytes).decode("utf-8"))
        return out

    # J3 caps. ``insufficient_evidence`` literally means "I don't know" — it
    # should never be paired with a high confidence number, which we have
    # seen happen when the underlying LLM emits a self-contradicting
    # ``{verdict: insufficient_evidence, confidence: 0.9}`` payload. We
    # therefore clamp it to ``INSUFFICIENT_CONFIDENCE_CAP``. ``contradicted``
    # similarly gets a soft ceiling so the report doesn't print
    # "100% contradicted" on shaky evidence; ``supported`` is left alone
    # because it's the natural high-confidence case.
    _INSUFFICIENT_CONFIDENCE_CAP = 0.5
    _CONTRADICTED_CONFIDENCE_CAP = 0.95

    def _merge_verdicts(
        self,
        cua_result: dict[str, Any],
        llm_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge CUA and LLM verdicts under the J3 sanity rules.

        Rules applied (in order):

        1. Pick the higher-confidence side as the base, but attach a
           ``conflict_marker`` flag when both sides are confident *and*
           disagree on the verdict — downstream report/meta-review can
           highlight these for manual review.
        2. ``insufficient_evidence`` verdicts cap the merged confidence at
           ``_INSUFFICIENT_CONFIDENCE_CAP`` (default 0.5). High confidence
           on "I don't know" is incoherent and inflates the report's
           ``insufficient_evidence_rate``.
        3. ``contradicted`` verdicts are capped at
           ``_CONTRADICTED_CONFIDENCE_CAP`` (default 0.95) so the dataset
           never displays "100% wrong" on a single LLM call.
        4. The merged ``reason`` always carries both sides so meta-review
           can audit why the final pick was made.
        """
        cua_verdict = str(cua_result.get("verdict") or "insufficient_evidence")
        llm_verdict = str(llm_result.get("verdict") or "insufficient_evidence")
        try:
            cua_conf = float(cua_result.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            cua_conf = 0.0
        try:
            llm_conf = float(llm_result.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            llm_conf = 0.0

        if llm_conf >= cua_conf:
            merged = dict(llm_result)
            base_verdict = llm_verdict
        else:
            merged = dict(cua_result)
            base_verdict = cua_verdict

        merged["reason"] = (
            f"CUA: {cua_result.get('reason', '') or '(none)'} "
            f"| LLM: {llm_result.get('reason', '') or '(none)'}"
        )
        # The judge's extended-thinking trace lives on llm_result; carry it
        # over even when the CUA side wins the confidence pick.
        if llm_result.get("judge_thinking_trace"):
            merged["judge_thinking_trace"] = llm_result["judge_thinking_trace"]

        merged_conf = max(cua_conf, llm_conf)
        sanity_notes: list[str] = []

        # Conflict marker: both sides confident, but they disagree.
        if (
            cua_conf >= 0.7
            and llm_conf >= 0.7
            and cua_verdict != llm_verdict
            and {cua_verdict, llm_verdict} & {"supported", "contradicted"}
        ):
            merged["conflict_marker"] = {
                "cua_verdict": cua_verdict,
                "cua_confidence": cua_conf,
                "llm_verdict": llm_verdict,
                "llm_confidence": llm_conf,
            }
            sanity_notes.append(
                f"conflict({cua_verdict}@{cua_conf:.2f} vs "
                f"{llm_verdict}@{llm_conf:.2f})"
            )

        # Confidence caps tied to verdict semantics.
        if (
            base_verdict == "insufficient_evidence"
            and merged_conf > self._INSUFFICIENT_CONFIDENCE_CAP
        ):
            sanity_notes.append(
                f"insufficient_conf_capped({merged_conf:.2f}->"
                f"{self._INSUFFICIENT_CONFIDENCE_CAP:.2f})"
            )
            merged_conf = self._INSUFFICIENT_CONFIDENCE_CAP
        elif (
            base_verdict == "contradicted"
            and merged_conf > self._CONTRADICTED_CONFIDENCE_CAP
        ):
            sanity_notes.append(
                f"contradicted_conf_capped({merged_conf:.2f}->"
                f"{self._CONTRADICTED_CONFIDENCE_CAP:.2f})"
            )
            merged_conf = self._CONTRADICTED_CONFIDENCE_CAP

        merged["confidence"] = merged_conf
        merged["verdict"] = base_verdict
        if sanity_notes:
            merged["sanity_notes"] = sanity_notes
            merged["reason"] = (merged.get("reason") or "") + (
                " [J3: " + "; ".join(sanity_notes) + "]"
            )

        return merged

    def _build_result(
        self,
        claim: EvalClaim,
        query_id: str,
        verdict_data: dict[str, Any],
        cua_result: dict[str, Any],
        used_validators: list[str],
    ) -> ClaimResult:
        native_stats = cua_result.get("native_tool_use_stats") or {}
        if not isinstance(native_stats, dict):
            native_stats = {}
        page_load_failure = cua_result.get("page_load_failure") or {}
        if not isinstance(page_load_failure, dict):
            page_load_failure = {}
        action_summary = cua_result.get("action_summary") or {}
        if not isinstance(action_summary, dict):
            action_summary = {}

        runtime_audit = self._build_runtime_audit(
            cua_result.get("runtime_signals") or {}
        )

        empty_state_signal = self._detect_empty_state(
            claim, verdict_data, cua_result,
        )

        verdict_attribution = self._attribute_verdict(verdict_data, cua_result)

        return ClaimResult(
            query_id=query_id,
            claim_id=claim.id,
            source_fc_id=claim.source_fc_id,
            checkpoint=claim.checkpoint,
            verification=claim.verification,
            claim_type=claim.claim_type,
            semantic_type=getattr(claim, "semantic_type", None),
            criticality=claim.criticality,
            verdict=verdict_data.get("verdict", "insufficient_evidence"),
            confidence=float(verdict_data.get("confidence", 0.0)),
            reason=verdict_data.get("reason", ""),
            evidence_sufficient=verdict_data.get("evidence_sufficient"),
            used_validators=used_validators,
            action_trace=cua_result.get("action_trace", []),
            action_summary=action_summary,
            dom_signals={},
            screenshots=cua_result.get("screenshots", []),
            cua_reasoning=cua_result.get("cua_reasoning", ""),
            native_tool_use_stats=native_stats,
            thinking_trace=cua_result.get("thinking_trace") or [],
            judge_thinking_trace=verdict_data.get("judge_thinking_trace") or [],
            page_load_failure=page_load_failure,
            runtime_audit=runtime_audit,
            empty_state_signal=empty_state_signal,
            verdict_attribution=verdict_attribution,
        )

    # Tools that do not count as "real" interaction with the app: pure
    # bookkeeping / observation steps. Mirrors the filter used by the
    # cua_verdict shortcut guard above.
    _NON_INTERACTION_TOOLS: frozenset[str] = frozenset(
        {"report_verdict", "__screenshot__", "__cursor__", "__unknown__"}
    )

    @classmethod
    def _attribute_verdict(
        cls,
        verdict_data: dict[str, Any],
        cua_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Classify *why* a non-supported verdict happened (diagnostic only).

        The key distinction for cross-CUA comparison: did the CUA actually
        attempt the required interaction? ``insufficient_evidence`` /
        ``contradicted`` with zero real action steps usually means the agent
        never drove the app (a CUA-capability problem, not an App defect),
        whereas the same verdict after real interaction is far more likely to
        be a genuine App issue. Does NOT change the verdict or score.
        """
        verdict = str(verdict_data.get("verdict", "")).lower()
        if verdict == "supported":
            return {}

        real_action_steps = sum(
            1 for a in (cua_result.get("action_trace") or [])
            if a.get("tool") not in cls._NON_INTERACTION_TOOLS
        )

        if verdict == "insufficient_evidence":
            if real_action_steps == 0:
                return {
                    "category": "cua_no_action",
                    "real_action_steps": 0,
                    "note": "CUA performed no real interaction; likely a "
                            "CUA-capability gap rather than an App defect",
                }
            return {
                "category": "inconclusive_after_action",
                "real_action_steps": real_action_steps,
                "note": "CUA interacted but evidence stayed inconclusive",
            }
        if verdict == "contradicted":
            if real_action_steps == 0:
                return {
                    "category": "contradicted_without_action",
                    "real_action_steps": 0,
                    "note": "contradicted with no interaction; treat as a "
                            "possible CUA-side false negative",
                }
            return {
                "category": "contradicted_after_action",
                "real_action_steps": real_action_steps,
                "note": "contradicted after real interaction; more likely a "
                        "genuine App defect",
            }
        return {}

    # Placeholder phrases the CUA tends to surface when a data-display
    # container is rendered but holds no user data yet. Kept lowercase for
    # case-insensitive matching against the reasoning text.
    _EMPTY_STATE_MARKERS: tuple[str, ...] = (
        "暂无", "还没有", "尚未", "没有记录", "没有数据", "无数据", "空状态",
        "空列表", "暂时没有", "开始第一次", "快去", "去添加", "添加第一",
        "no records", "no data", "nothing here", "empty list", "empty state",
        "get started", "no entries", "no items", "not yet", "placeholder",
        "container present, data empty", "data empty", "prerequisite not",
    )

    # Verification methods whose verification depends on prior user data being
    # present (data-display claims: structural existence, value correctness,
    # persistence, perceptual). Keyed on the Axis-1 ``claim_type``.
    _DATA_DISPLAY_CLAIM_TYPES: frozenset[str] = frozenset(
        {"structural", "value_correctness", "persistence", "perceptual"}
    )

    @classmethod
    def _detect_empty_state(
        cls,
        claim: EvalClaim,
        verdict_data: dict[str, Any],
        cua_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Flag (without changing the verdict) likely empty-state stalls.

        A diagnostic-only heuristic: when a data-display claim does NOT end
        ``supported`` and the CUA reasoning contains empty-state placeholder
        language, we record the matched markers so the next iteration can
        see which claims got stuck on "page had no user data yet".
        Returns ``{}`` when not applicable so consumers treat absence as
        "no signal".
        """
        verdict = str(verdict_data.get("verdict", "")).lower()
        if verdict == "supported":
            return {}
        claim_type = str(getattr(claim, "claim_type", "")).lower()
        if claim_type and claim_type not in cls._DATA_DISPLAY_CLAIM_TYPES:
            return {}
        haystack = " ".join(
            str(x or "") for x in (
                cua_result.get("cua_reasoning"),
                verdict_data.get("reason"),
            )
        ).lower()
        if not haystack.strip():
            return {}
        markers = [m for m in cls._EMPTY_STATE_MARKERS if m in haystack]
        if not markers:
            return {}
        return {
            "suspected": True,
            "markers": markers[:8],
            "verdict": verdict,
        }

    @staticmethod
    def _build_runtime_audit(runtime_signals: dict[str, Any]) -> dict[str, Any]:
        """Distil the per-claim Playwright snapshot into a compact audit dict.

        Keeps the full ``backend_calls`` list (already capped upstream) plus
        a few rollup counters so meta-review / scoring can reason about the
        runtime truth without re-walking the trace. Returns ``{}`` when no
        backend traffic was observed so downstream code can treat absence as
        "no signal" rather than "healthy".
        """
        if not isinstance(runtime_signals, dict):
            return {}
        backend_calls = runtime_signals.get("backend_calls") or []
        if not isinstance(backend_calls, list):
            backend_calls = []
        console_errors = runtime_signals.get("console_errors") or []
        failed_requests = runtime_signals.get("failed_requests") or []

        error_calls = [c for c in backend_calls
                       if isinstance(c, dict) and not c.get("ok", True)]
        ok_count = len(backend_calls) - len(error_calls)
        pg_codes: list[str] = []
        for c in error_calls:
            code = c.get("pg_code")
            if code and str(code) not in pg_codes:
                pg_codes.append(str(code))

        if not backend_calls and not console_errors and not failed_requests:
            return {}

        return {
            "backend_calls": backend_calls,
            "backend_error_count": len(error_calls),
            "backend_ok_count": ok_count,
            "pg_codes": pg_codes,
            "console_error_count": len(console_errors)
            if isinstance(console_errors, list) else 0,
            "failed_request_count": len(failed_requests)
            if isinstance(failed_requests, list) else 0,
        }
