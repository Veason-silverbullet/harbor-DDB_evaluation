"""
CUARunner: unified function-calling agent loop.

One loop handles both OpenAI-compatible and Anthropic SDKs; the same
``ToolRegistry`` + ``ActionExecutor`` / ``PageInspector`` powers the
execution side regardless of which LLM protocol is used.

Termination:
- The LLM calls the sentinel tool ``report_verdict`` (preferred), or
- The LLM returns text without any tool call (final-answer mode: we parse
  VERDICT/CONFIDENCE/... from the text as a fallback), or
- Max iterations reached, or
- Early-stop: no visible progress for ``stale_window`` consecutive steps.
"""
from __future__ import annotations
import base64
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .playwright_env import PlaywrightEnv
from .tools.executor import ActionExecutor
from .tools.inspector import PageInspector
from .tools.schemas import (
    Tool,
    ToolRegistry,
    build_default_registry,
    REPORT_VERDICT_TOOL_NAME,
    DELEGATE_TO_VIEWER_TOOL_NAME,
    MULTI_SESSION_TOOL_NAMES,
    SEARCH_TOOL_NAME,
    VERACITY_TOOL_NAMES,
)
from .multi_session import MultiSessionMixin, multi_session_block, veracity_block
from .transient import capture_transient_immediate
from types_.errors import QuotaExhaustedError, is_quota_error
import usage_tracker

logger = logging.getLogger(__name__)


# ── Prompts ─────────────────────────────────────────────────────

_GOAL_TEMPLATE = """\
You are a Computer-Using Agent evaluating a web application to verify a specific claim.

## Claim
{checkpoint}

## How to verify
{verification}

## Claim type
{claim_type}

## App URL
{app_url}

## How to act
1. Look at the screenshot of the current viewport (size: {w}x{h}).
2. Call ONE OR MORE tools to interact (click / type / scroll / drag / ...).
   After each turn a new screenshot is shown so you can inspect the effect.
3. Only base your judgement on what is directly observable.
4. When done, call the `report_verdict` tool EXACTLY ONCE with your verdict,
   a confidence in [0, 1], whether evidence_sufficient, and a short reasoning.
   Do NOT call any action tool in the same turn as report_verdict.
5. Coordinates must be integers inside [0, {w}) x [0, {h}).

## Empty-state vs missing-feature (READ for presence claims)
- Many `presence` claims describe data displays that REQUIRE prior
  user data — "displays today's completed focus rounds", "shows
  session history", "lists saved items", "shows weekly total", etc.
- When you open the page and see a WELL-FORMED placeholder card with
  copy like "还没有...", "暂无...", "尚未...", "开始第一次",
  "no records yet", "get started", "empty list", that is NOT evidence
  the feature is missing. It is evidence the feature's container IS
  rendered and just has no data yet because the prerequisite user
  action has not occurred (and may be too expensive to perform in your
  iteration budget — e.g. completing a 25-min countdown).
- Decision tree for these cases:
    * Container/region clearly matches the claim's subject (card
      titled "今日专注" for a claim about "今日完成轮次", "Recent"
      panel for "history list", etc.) AND shows an empty-state
      placeholder → call `report_verdict` with
      verdict="insufficient_evidence", confidence <= 0.45,
      reasoning="container present, data empty because prerequisite
      not exercised".
    * The prerequisite is a long timer/countdown/cooldown (e.g. a
      25-min focus timer must finish before "今日完成轮次" increments):
      START the timer, then call `advance_time` (minutes=...) to skip
      it to completion INSTANTLY, then re-check. Only if the data is
      STILL empty after that → verdict="insufficient_evidence",
      confidence<=0.45.
    * Any other prerequisite that cannot be exercised inside the
      iteration budget → verdict="insufficient_evidence", confidence<=0.45.
    * Reserve `contradicted` for cases where the claimed UI section
      is truly ABSENT (no such card/region anywhere on the page) or
      shows clearly unrelated content. Empty-state + correct title
      is NOT contradicted.

## Timers & countdowns — use `advance_time` (READ for any timer claim)
- The page runs on a controllable VIRTUAL CLOCK. The `advance_time`
  tool fast-forwards it (minutes/seconds) so a long countdown, timer,
  or cooldown reaches its END instantly — you NEVER need to wait real
  minutes for a 25-min focus timer, Pomodoro, exam clock, cooldown, etc.
- Canonical plan for "does the timer complete / what happens at the end"
  (e.g. completion popup, 00:00 state, round counter +1, fail dialog on
  reaching zero):
    1. Set the duration if the app asks (e.g. pick 25 min) and click
       START so the timer is actually running.
    2. Call `advance_time` with the full duration (e.g. minutes=25), or
       a bit more, to jump to the end.
    3. Look at the next screenshot for the end-state and `report_verdict`.
- IMPORTANT: `advance_time` only does something AFTER the timer has been
  started — advancing on a paused/unstarted timer changes nothing.
- If after advancing the expected end-state still does not appear, the
  feature is genuinely broken/missing → `contradicted` (timer-specific
  end behaviour absent) or `insufficient_evidence` per the rules above.

## Visual time-series claims (READ if claim_type == "visual_semantics")
- For claims about a UI element CHANGING OVER TIME (countdown ring
  decreasing, progress bar advancing, animation direction, blinking,
  carousel auto-rotation, etc.), pure repeated `wait` is NOT useful
  evidence — the judge sees the screenshots, not your inner monologue.
- Recommended micro-plan (3-5 steps total, NOT 10):
    1. Trigger the relevant interaction (e.g. click "Start" on the
       timer). One short `wait` (1-3s) is fine to let the UI settle.
    2. Note the visible state in your reasoning so it is captured for
       comparison (e.g. "ring shows 25:00, full circle, pointer at
       12 o'clock").
    3. Advance the clock a little with `advance_time` (e.g. seconds=10),
       or `wait` 5-15s of real time, so the UI has time to visibly change.
    4. Look at the new screenshot and compare against step-2 state.
    5. Call `report_verdict` with whichever direction you saw. The
       judge will independently receive start/mid/end frames from
       your trace and corroborate.
- Do NOT keep calling `wait` past step 4. If two samples have not
  produced a visible change relevant to the claim, call
  `report_verdict` with `insufficient_evidence` rather than burning
  the remaining iterations.

## TTS / audio claims (READ if the claim mentions 朗读 / TTS / speech / voice playback)
- Browser text-to-speech (speechSynthesis) produces NO visible change in
  the screenshot. An unchanged screenshot after clicking a speaker/play
  button is NORMAL for these claims — do NOT treat it as failure and do
  NOT keep re-clicking.
- The harness instruments speechSynthesis. When the app calls it, the
  tool result of your action will contain a line like:
    TTS event: speak(text="...", lang="ja-JP")
- Use that line as first-class evidence:
    * The line appears and its text matches the on-screen line you
      clicked -> the TTS claim is `supported`.
    * You clicked the speaker button (click success=true, correct
      coordinates) and NO "TTS event" line appears -> the app did not
      invoke browser TTS -> `contradicted`.
    * You could not locate/click any speaker button -> follow the normal
      presence/empty-state rules instead.

## Budget discipline (MUST follow)
- You have a limited iteration budget (see {claim_type} budget). The
  CALLER will silently fail this claim if iterations run out without
  a `report_verdict`. That is the WORST outcome — strictly worse than
  guessing.
- Therefore: BEFORE the last turn of your budget, you MUST call
  `report_verdict`, even if you are not fully sure. The correct
  uncertain fallback is:
    report_verdict(verdict="insufficient_evidence",
                   evidence_sufficient=false,
                   confidence <= 0.4,
                   reasoning="<what you saw, and what was missing>")
- Plan accordingly: do not stack 5+ `wait` calls in a row. Two well-
  placed waits + a `report_verdict` always beats ten waits + timeout.

## Page-load failure handling (J1)
- If the viewport is BLANK / pure white / shows only an error banner
  ("Cannot GET ...", "ERR_CONNECTION_RESET", a stack trace, an empty
  body), and especially if the page stays unchanged after a `wait`, you
  MUST first try the `reload` tool ONCE before giving up.
- After that one reload, if the page is STILL blank or still shows the
  same error banner, report:
    * `contradicted` (confidence 0.85+) for presence / navigation /
      visual_semantics / interaction_accessibility claims — a dead page
      is direct evidence that the claimed UI does NOT exist.
    * `insufficient_evidence` for computation / persistence /
      data_integration / runtime_health claims — without a working page
      we cannot exercise the feature, so neither verdict is justified.
- In the reasoning, explicitly write the phrase `page_load_failed` so
  the orchestrator can correlate your finding with its own page-load
  failure detector. Then the orchestrator will run additional reload
  retries on its own; you only need to attempt one reload yourself.

## Reasoning style (MUST follow)
- Write the `reasoning` in the SAME language as the Claim above:
  Chinese claim -> Chinese reasoning; English claim -> English reasoning.
  Do not mix languages in a single reasoning.
- Length: 1-2 sentences total. State (a) what you observed and (b) why it
  does or does not support the claim.
- Do NOT exceed 2 sentences. Do NOT collapse to a bare verdict word
  (e.g. just "supported" / "失败"). No bullet lists, no preamble like
  "Based on the screenshot..." / "根据截图...".

Available verdicts: supported | contradicted |
insufficient_evidence | not_applicable.
"""


def _guideline_block(guideline: str | None) -> str:
    """Render the per-query verification guideline as an appended prompt
    section, or "" when absent.

    The guideline is a model-agnostic, per-query setup/orchestration note
    (prerequisites to satisfy, fixtures/seeds available, ordering, app-wide
    conventions) produced by the Stage-3 generator and human-reviewed. It is
    injected verbatim AFTER the static goal so it overrides nothing but adds
    query-specific context. It MUST NOT leak the expected verdict.
    """
    text = (guideline or "").strip()
    if not text:
        return ""
    return (
        "\n\n## Per-query verification guideline (read carefully)\n"
        "The following query-specific notes describe prerequisites, available "
        "fixtures/seeds, and conventions for THIS app. Follow them when "
        "planning your actions. They do NOT tell you the answer — you still "
        "must observe the app and decide the verdict yourself.\n"
        f"{text}\n"
    )


@dataclass
class RunnerConfig:
    """Tuning knobs for the CUA loop."""

    agent_protocol: str = "function_calling"  # or "native_computer_20251124"
    tool_mode: str = "multi_tool"             # or "single_tool" (future)
    model: str = "claude-opus-4-6"
    api_key: str = ""
    base_url: str = ""
    max_iterations: int = 20
    max_iterations_by_claim_type: dict[str, int] = field(default_factory=dict)
    post_action_delay_ms: int = 500
    save_screenshots: bool = True
    # Run root that holds per-query subdirs. The runner writes
    # ``<screenshot_dir>/<query_id>/screenshots/<claim_id>_stepN.png`` so
    # each query's PNGs sit next to its result JSON.
    screenshot_dir: Path = field(default_factory=lambda: Path("./results"))
    enable_readonly_tools: bool = False
    temperature: float = 0.0
    max_output_tokens: int = 4096
    tool_fail_retries: int = 1
    stale_window: int = 3   # consecutive unchanged screenshots -> warn the model
    # P0-1: hard early-stop. After this many consecutive unchanged screenshots
    # the runner stops the loop and synthesizes an ``insufficient_evidence``
    # verdict instead of spinning to ``max_iterations`` (the type-2 timeout:
    # CUA repeatedly clicks because a prerequisite never succeeded). Must be
    # >= ``stale_window`` so the model gets the warning + a few steps to
    # change strategy before we force-converge.
    stale_hard_stop: int = 6
    request_retries: int = 1
    # P0-2: exponential backoff (ms) between transient LLM-request retries.
    # base * 2**attempt, capped at request_backoff_cap_ms. 0 disables backoff.
    request_backoff_base_ms: int = 800
    request_backoff_cap_ms: int = 8000
    # P0-3: capture screenshots only after the viewport visually settles
    # (consecutive identical frames or timeout). Avoids judging mid-animation
    # / mid-render frames. ``stable_wait_load`` adds a short networkidle wait
    # (intended for the L-DOM track).
    stable_capture: bool = True
    stable_max_wait_ms: int = 800
    stable_settle_ms: int = 120
    stable_wait_load: bool = False
    # Keep only the most recent N base64 screenshots in the message history.
    # Older screenshots are replaced with a small text placeholder, which
    # drastically cuts request payload size as the loop progresses.
    #
    # Set to 20 (== current max_iterations fallback) on 2026-04-27 so that
    # for every claim_type the agent retains the full per-claim screenshot
    # trace — this matches the pre-trim baseline behavior of the 04-26 run.
    #
    # History from this knob:
    #   * 2 (initial trim): cut payload aggressively but caused CUA to forget
    #     early evidence (e.g. "I already saw 3 product cards 5 steps ago"),
    #     reducing supported verdicts on multi-step presence claims.
    #   * 5: same regression pattern, just slower (LLM judge complained about
    #     "无 CUA 推理记录" because terminal screenshot was unrelated).
    #   * 20: covers the largest per-type budget (action_effect:20), so the
    #     trim is effectively a no-op for current configs while still capping
    #     payload if max_iterations is later raised much higher.
    # If you raise any per-type budget above this number, raise this too.
    keep_last_screenshots: int = 20

    # Extra HTTP headers attached to every LLM request. Mainly used by the
    # native_computer_20251124 adapter to force `anthropic-beta:
    # computer-use-2025-11-24` through gateways that strip non-whitelisted
    # headers. The function_calling path can also use this to pass
    # gateway-specific tags (x-vendor, x-route, ...). Empty by default so
    # FC users that go through stock OpenAI-compatible relays are unaffected.
    extra_headers: dict[str, str] = field(default_factory=dict)

    # Anthropic extended-thinking budget for the native_computer_20251124
    # path. 0 = disabled (default). See ``config.settings.CUAConfig`` for
    # the full motivation / suggested values. Ignored by CUARunner.
    thinking_budget_tokens: int = 0

    # OpenAI reasoning controls for the native_computer_openai path; both
    # ignored by CUARunner and the Anthropic adapter. See
    # ``config.settings.CUAConfig`` for semantics.
    reasoning_effort: str = "medium"
    reasoning_summary: str = "concise"

    # OpenAI reasoning round-trip toggle (native_computer_openai only).
    # False = server-side state via previous_response_id (default).
    # True  = stateless client-side echo of reasoning items carrying
    #         ``encrypted_content`` (store=False + include=...). See
    #         ``config.settings.CUAConfig.reasoning_passthrough`` for the
    #         full rationale. Ignored by every other runner.
    reasoning_passthrough: bool = False

    # ── Multi-session collaboration (0702) ──────────────────────────
    # Global feature switch. When False (default) the delegate_to_viewer tool
    # is never registered and the runner behaves exactly as before. When True,
    # the tool is registered but still only made VISIBLE per-claim for claims
    # the verifier flags ``multi_session: true``.
    enable_multi_session: bool = False
    # Iteration budget for each viewer subagent loop (kept small — the viewer
    # performs ONE delegated action then reports).
    viewer_max_iterations: int = 8
    # Model for the viewer subagent. Empty => reuse the main ``model``. Point
    # this at a cheaper model to cap the extra cost of collaboration claims.
    viewer_model: str = ""
    # Hard cap on how many viewer subagents a single claim may spawn, bounding
    # the extra cost (default 2 == viewer_a + viewer_b).
    max_viewers_per_claim: int = 2

    # ── Transient-feedback capture (弹幕/toast) ──────────────────────
    # For state_transition claims (发弹幕/发评论/投票/点赞…), take ONE extra
    # screenshot IMMEDIATELY after the action batch, WITHOUT waiting for the
    # viewport to settle, so a fast-disappearing element (danmaku/toast) is
    # captured before it flies off screen. The frame is added to the judge's
    # sampling pool and also shown to the CUA. Applies to every protocol.
    capture_transient_immediate: bool = True

    # ── Real-world data veracity check ──────────────────────────────
    # Master switch. When True, data-bearing claims (data_integration /
    # content / value_correctness) get the ``search_web`` tool and a
    # ground-truth verification prompt so the CUA can confirm the app's
    # real-world data is actually correct/current instead of stale or mock.
    # When False (default) the tool is never registered and behaviour is
    # unchanged.
    enable_veracity_check: bool = False
    # Pluggable search provider config for ``search_web``. When no api_key/
    # endpoint is set the tool returns "search unavailable" and veracity
    # degrades to insufficient_evidence — it NEVER false-fails a claim.
    search_provider: str = "tavily"   # tavily | serper | custom
    search_api_key: str = ""
    search_endpoint: str = ""         # empty => provider default endpoint
    search_max_results: int = 5


# ── Runner ──────────────────────────────────────────────────────

class CUARunner(MultiSessionMixin):
    """Single loop that works for OpenAI-compatible SDKs.

    For the Anthropic native computer_20251124 protocol use
    ``cua.native_adapter.NativeAnthropicRunner`` (shares the same ActionExecutor).

    Inherits viewer-subagent + veracity-search plumbing from
    :class:`cua.multi_session.MultiSessionMixin`.
    """

    def __init__(
        self,
        env: PlaywrightEnv,
        config: RunnerConfig,
        registry: ToolRegistry | None = None,
        executor: ActionExecutor | None = None,
        inspector: PageInspector | None = None,
    ) -> None:
        self._env = env
        self._cfg = config
        self._executor = executor or ActionExecutor(
            env, post_action_delay_ms=config.post_action_delay_ms
        )
        self._inspector = inspector or PageInspector(env)
        self._registry = registry or build_default_registry(
            include_multi_session=config.enable_multi_session,
            include_veracity=config.enable_veracity_check,
        )
        # Per-claim multi-session state (reset at the top of verify_claim).
        self._ms_reset()
        self._bind_handlers()

    def _bind_handlers(self) -> None:
        """Bind per-tool python handlers to the registry entries."""
        action_map = {
            "click": self._executor.click,
            "mouse_down": self._executor.mouse_down,
            "mouse_up": self._executor.mouse_up,
            "move": self._executor.move,
            "drag": self._executor.drag,
            "scroll": self._executor.scroll,
            "type_text": self._executor.type_text,
            "press_key": self._executor.press_key,
            "hold_key": self._executor.hold_key,
            "wait": self._executor.wait,
            "advance_time": self._executor.advance_time,
            "zoom": self._executor.zoom,
            "reload": self._executor.reload,
            "upload_file": self._executor.upload_file,
            "get_page_info": self._inspector.get_page_info,
            "get_accessibility_tree": self._inspector.get_accessibility_tree,
            # Only present in the registry when multi-session is enabled.
            DELEGATE_TO_VIEWER_TOOL_NAME: self._delegate_to_viewer,
            # Only present in the registry when veracity checking is enabled.
            SEARCH_TOOL_NAME: self._run_search,
        }
        for name, fn in action_map.items():
            tool = self._registry.get(name)
            if tool is not None:
                tool.handler = fn

    # ── public entry ───────────────────────────────────────────────

    async def verify_claim(
        self,
        claim_id: str,
        checkpoint: str,
        verification: str,
        claim_type: str,
        app_url: str,
        query_id: str = "",
        guideline: str = "",
        multi_session: bool = False,
        participants: list[str] | None = None,
        marker: str = "",
        session_urls: dict[str, str] | None = None,
        veracity_check: bool = False,
    ) -> dict[str, Any]:
        cfg = self._cfg
        # Multi-session is only actually armed when the feature is globally
        # enabled AND this specific claim asked for it.
        ms_active = self._ms_armed(multi_session)
        veracity_active = self._veracity_armed(veracity_check)
        # Reset per-claim collaboration state so a viewer handler invoked
        # during this claim has the right context.
        self._ms_reset(
            query_id=query_id,
            claim_id=claim_id,
            app_url=app_url,
            marker=marker,
            participants=participants,
            session_urls=session_urls,
        )

        vp = await self._env.get_viewport_size()
        w, h = vp["width"], vp["height"]
        goal = _GOAL_TEMPLATE.format(
            checkpoint=checkpoint,
            verification=verification,
            claim_type=claim_type,
            app_url=app_url,
            w=w, h=h,
        ) + _guideline_block(guideline)
        if ms_active:
            goal += multi_session_block(participants, marker)
        if veracity_active:
            goal += veracity_block()

        budget = cfg.max_iterations_by_claim_type.get(claim_type, cfg.max_iterations)

        screenshots: list[str] = []
        action_trace: list[dict[str, Any]] = []

        screenshot_b64 = await self._capture_step_screenshot(
            query_id, claim_id, 0, screenshots,
        )

        # Build initial messages (OpenAI-compatible shape).
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You act via tool calls. Prefer pixel observation; only use "
                    "readonly tools if pixels are ambiguous. End with report_verdict."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": goal},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                ],
            },
        ]

        include_readonly = cfg.enable_readonly_tools
        tool_schemas = self._registry.to_openai_format(include_readonly=include_readonly)
        # Per-claim visibility: the delegate_to_viewer / search_web tools exist
        # in the registry whenever their feature is enabled, but a claim that
        # did not opt in must never see them (zero behaviour/cost change).
        hidden_tools: set[str] = set()
        if not ms_active:
            hidden_tools |= set(MULTI_SESSION_TOOL_NAMES)
        if not veracity_active:
            hidden_tools |= set(VERACITY_TOOL_NAMES)
        if hidden_tools:
            tool_schemas = [
                t for t in tool_schemas
                if t.get("function", {}).get("name") not in hidden_tools
            ]

        client = self._make_openai_client()

        verdict_payload: dict[str, Any] | None = None
        raw_assistant_text = ""
        last_screenshot_hash = _sha1(screenshot_b64)
        stale_steps = 0

        for step in range(1, budget + 1):
            logger.info(
                "CUA step %d/%d claim=%s [%s]",
                step, budget, claim_id, claim_type,
            )
            _trim_screenshots_in_history(messages, keep_last=cfg.keep_last_screenshots)
            try:
                response = self._chat_with_retry(
                    client=client,
                    messages=messages,
                    tools=tool_schemas,
                )
            except QuotaExhaustedError:
                raise
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return self._error_result(
                    f"LLM error: {type(e).__name__}: {e}",
                    action_trace, screenshots,
                    evaluator_error=True,
                )

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if msg.content:
                raw_assistant_text = msg.content

            # 1) If no tool calls -> treat as verdict text (legacy behavior).
            if not tool_calls:
                logger.info("CUA step %d: text-only response, parsing verdict", step)
                return _compose_result(
                    _parse_text_verdict(raw_assistant_text or ""),
                    action_trace, screenshots, raw_assistant_text,
                )

            # 2) Record the assistant turn with its tool_calls into the message log.
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # 3) Dispatch each tool call. If report_verdict was called, we stop.
            hit_verdict = False
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == REPORT_VERDICT_TOOL_NAME:
                    verdict_payload = args
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "verdict recorded",
                    })
                    action_trace.append({
                        "step": step,
                        "tool": name,
                        "action": name,
                        "input": args,
                        "success": True,
                        "error": None,
                    })
                    logger.info(
                        "CUA step=%d tool=%s verdict=%s confidence=%.2f",
                        step, name,
                        args.get("verdict"), float(args.get("confidence", 0.0) or 0.0),
                    )
                    hit_verdict = True
                    continue

                tool_result = await self._call_tool(name, args)
                action_trace.append({
                    "step": step,
                    "tool": name,
                    "action": name,
                    "input": args,
                    "success": bool(tool_result.get("success")),
                    "error": tool_result.get("error"),
                    "clamped": bool(tool_result.get("clamped")),
                })
                logger.info(
                    "CUA step=%d tool=%s args=%s success=%s clamped=%s error=%s",
                    step, name, _args_preview(args),
                    bool(tool_result.get("success")),
                    bool(tool_result.get("clamped")),
                    tool_result.get("error"),
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _format_tool_result_text(name, args, tool_result),
                })

            if hit_verdict:
                break

            # 4) After the batch of tool calls, take a fresh screenshot and
            #    feed it back so the model can see the effect. We capture
            #    the bytes once and reuse for both the LLM payload and the
            #    on-disk artifact so we don't pay for two screenshots.
            #    For transient-feedback claims (弹幕/toast) grab ONE extra
            #    immediate (non-settling) frame FIRST so a fast-disappearing
            #    element is captured before the stable frame settles.
            imm_b64 = await capture_transient_immediate(
                self._env, cfg, query_id, claim_id, step, screenshots, claim_type,
            )
            new_b64 = await self._capture_step_screenshot(
                query_id, claim_id, step, screenshots,
            )
            new_hash = _sha1(new_b64)
            if new_hash == last_screenshot_hash:
                stale_steps += 1
            else:
                stale_steps = 0
            last_screenshot_hash = new_hash

            # P0-1: hard early-stop on a stuck loop (type-2 timeout). The
            # screen has not changed for stale_hard_stop steps despite the
            # warning below, so further iterations would only spin to
            # max_iterations and forfeit the claim. Synthesize an
            # insufficient verdict now instead.
            if stale_steps >= self._cfg.stale_hard_stop:
                logger.warning(
                    "CUA stale hard-stop: %d unchanged frames for %s (budget %d)",
                    stale_steps, claim_id, budget,
                )
                return _compose_result(
                    {
                        "verdict": "insufficient_evidence",
                        "confidence": 0.3,
                        "reasoning": (
                            f"screen unchanged for {stale_steps} consecutive steps; "
                            f"stopped early to avoid spinning (likely a prerequisite "
                            f"action had no effect)."
                        ),
                        "evidence_sufficient": False,
                    },
                    action_trace, screenshots,
                    raw_assistant_text or "stale hard-stop",
                )

            follow_up_text = (
                f"Viewport {w}x{h}. Inspect the new screenshot and decide the "
                f"next tool call. (iteration {step}/{budget})"
            )
            if stale_steps >= self._cfg.stale_window:
                follow_up_text += (
                    f"\nWARNING: the screenshot has not changed for "
                    f"{stale_steps} consecutive steps. Try a different strategy, "
                    f"or call report_verdict with insufficient_evidence."
                )
            # Budget-discipline reminder. Forcing `report_verdict` in the
            # final turn(s) is the single most effective fix for the
            # "CUA reached max iterations" failure mode where the LLM
            # judge fallback then has to operate on whatever fragments
            # of evidence happened to be captured.
            remaining = budget - step
            if remaining <= 1:
                follow_up_text += (
                    f"\nIMPORTANT: only {remaining} iteration(s) left in this "
                    f"budget ({budget} total). You MUST call `report_verdict` "
                    f"on your NEXT turn — do not call any other tool. If you "
                    f"are still uncertain, return "
                    f"verdict=\"insufficient_evidence\", "
                    f"evidence_sufficient=false, confidence<=0.4, and put "
                    f"what you actually saw in `reasoning`. Silently running "
                    f"out of iterations forfeits this claim entirely."
                )

            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": follow_up_text},
            ]
            if imm_b64:
                user_content.append({
                    "type": "text",
                    "text": (
                        "Extra frame captured IMMEDIATELY after the action "
                        "(before the screen settled) to catch transient UI "
                        "like danmaku/toast:"
                    ),
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{imm_b64}"},
                })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{new_b64}"},
            })
            messages.append({"role": "user", "content": user_content})
        else:
            logger.warning("CUA max iterations (%d) reached for %s", budget, claim_id)
            if verdict_payload is None:
                return self._error_result(
                    f"CUA reached max iterations ({budget})",
                    action_trace, screenshots,
                )

        # Final verdict from report_verdict tool.
        return _compose_result(
            _verdict_from_payload(verdict_payload or {}),
            action_trace, screenshots,
            raw_assistant_text or json.dumps(verdict_payload or {}, ensure_ascii=False),
        )

    # ── helpers ────────────────────────────────────────────────────

    async def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool: Tool | None = self._registry.get(name)
        if tool is None or tool.handler is None:
            return {
                "success": False,
                "error": f"unknown tool: {name}",
                "text": None,
                "image_b64": None,
                "clamped": False,
            }
        last: dict[str, Any] = {}
        for attempt in range(self._cfg.tool_fail_retries + 1):
            try:
                last = await tool.handler(args)
                if last.get("success"):
                    return await self._enrich_with_tts_events(last)
            except QuotaExhaustedError:
                # Propagate quota exhaustion (e.g. from a viewer subagent loop)
                # so the batch runner can checkpoint and re-raise cleanly.
                raise
            except Exception as e:
                last = {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "text": None,
                    "image_b64": None,
                    "clamped": False,
                }
            logger.warning(
                "tool %s failed (attempt %d/%d): %s",
                name, attempt + 1, self._cfg.tool_fail_retries + 1,
                last.get("error"),
            )
        return await self._enrich_with_tts_events(last)

    async def _enrich_with_tts_events(
        self, result: dict[str, Any],
    ) -> dict[str, Any]:
        """Append freshly-drained TTS events to the tool result text.

        Browser TTS produces no screenshot change, so without this the CUA
        cannot tell a working speaker button from a dead one. The harness
        shim records every ``speechSynthesis.speak()`` call; new events
        since the previous action are surfaced as ``TTS event:`` lines in
        the action feedback. Best-effort: any failure leaves the result
        untouched.
        """
        try:
            new_events = await self._env.drain_tts_events()
        except Exception as e:  # noqa: BLE001
            logger.debug("TTS drain after tool call failed: %s", e)
            return result
        if not new_events:
            return result
        lines = [_format_tts_event(ev) for ev in new_events]
        base = (result.get("text") or "").rstrip()
        result["text"] = (base + "\n" + "\n".join(lines)).strip()
        return result

    async def _capture_step_screenshot(
        self,
        query_id: str,
        claim_id: str,
        step: int,
        screenshots: list[str],
    ) -> str:
        """Take ONE screenshot, persist it (if configured), and return base64.

        The previous implementation took two screenshots per step (one for
        the LLM payload and one for the on-disk artifact). This version
        captures bytes once and reuses them for both purposes.

        ``step > 0`` frames follow a tool batch (click/drag/navigate), so we
        wait for the viewport to settle first (P0-3). Step 0 is the initial
        load, already warmed up by the orchestrator's post-goto wait.
        """
        if self._cfg.stable_capture and step > 0:
            raw = await self._env.screenshot_bytes_stable(
                max_wait_ms=self._cfg.stable_max_wait_ms,
                settle_ms=self._cfg.stable_settle_ms,
                wait_load=self._cfg.stable_wait_load,
            )
        else:
            raw = await self._env.screenshot_bytes()
        if self._cfg.save_screenshots:
            try:
                # Per-query subdir so a single query's result JSON and PNGs
                # live together. The claim_id+step filename stays unique
                # without re-prefixing the qid.
                path = (
                    self._cfg.screenshot_dir
                    / query_id
                    / "screenshots"
                    / f"{claim_id}_step{step}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                screenshots.append(str(path))
            except Exception as e:
                logger.warning("failed to persist screenshot step=%d: %s", step, e)
        return base64.b64encode(raw).decode("utf-8")

    def _make_openai_client(self):
        from openai import OpenAI
        kwargs: dict[str, Any] = {
            "api_key": self._cfg.api_key,
            "base_url": self._cfg.base_url or None,
        }
        # `default_headers` is attached to every outgoing request by the
        # OpenAI SDK. We use the same `cua.extra_headers` knob as the
        # native_computer_20251124 path so that gateway-specific tags
        # (vendor routing, anthropic-beta passthrough, ...) only need to
        # be configured once in yaml regardless of agent_protocol.
        if self._cfg.extra_headers:
            kwargs["default_headers"] = dict(self._cfg.extra_headers)
        return OpenAI(**kwargs)

    def _chat_with_retry(self, client, messages, tools):
        last_exc: Exception | None = None
        for attempt in range(self._cfg.request_retries + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self._cfg.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": self._cfg.temperature,
                    "max_tokens": self._cfg.max_output_tokens,
                }
                if self._cfg.extra_headers:
                    request_kwargs["extra_headers"] = dict(self._cfg.extra_headers)
                response = client.chat.completions.create(**request_kwargs)
                usage_tracker.record_response("cua", response, model=self._cfg.model)
                return response
            except Exception as e:
                if is_quota_error(e):
                    raise QuotaExhaustedError("cua", str(e)) from e
                last_exc = e
                logger.warning(
                    "chat.completions attempt %d/%d failed: %s",
                    attempt + 1, self._cfg.request_retries + 1, e,
                )
                # P0-2: exponential backoff before the next retry so a
                # transient gateway hiccup (timeout / 5xx) gets a real
                # chance to recover instead of immediately exhausting the
                # retry budget and forfeiting the claim.
                if attempt < self._cfg.request_retries:
                    self._backoff_sleep(attempt)
        assert last_exc is not None
        raise last_exc

    def _backoff_sleep(self, attempt: int) -> None:
        """Exponential backoff between transient request retries (P0-2)."""
        base = max(0, self._cfg.request_backoff_base_ms)
        if base <= 0:
            return
        cap = max(base, self._cfg.request_backoff_cap_ms)
        delay_ms = min(base * (2 ** attempt), cap)
        time.sleep(delay_ms / 1000.0)

    @staticmethod
    def _error_result(
        msg: str,
        action_trace: list[dict],
        screenshots: list[str],
        *,
        evaluator_error: bool = False,
    ) -> dict[str, Any]:
        # ``evaluator_error`` (P0-2) flags failures of the EVALUATION harness
        # itself (LLM/gateway down, request timeout) rather than a real app
        # defect, so downstream scoring can exclude them instead of charging
        # the app an ``insufficient_evidence``. The verdict stays
        # insufficient for back-compat with consumers that ignore the flag.
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.0,
            "reason": msg,
            "evidence_sufficient": False,
            "evaluator_error": bool(evaluator_error),
            "action_trace": action_trace,
            "action_summary": _compute_action_summary(action_trace),
            "screenshots": screenshots,
            "cua_reasoning": "",
            "raw_response": msg,
        }


# ── Tool-result / verdict helpers ───────────────────────────────

def _format_tts_event(ev: dict[str, Any]) -> str:
    """One-line summary of a drained TTS event for the action feedback."""
    text = (ev.get("text") or "").replace("\n", " ")
    if len(text) > 120:
        text = text[:119] + "…"
    parts = [f'TTS event: speak(text="{text}"']
    if ev.get("lang"):
        parts.append(f', lang="{ev["lang"]}"')
    if ev.get("voice"):
        parts.append(f', voice="{ev["voice"]}"')
    parts.append(")")
    if ev.get("error"):
        parts.append(f" [engine error: {ev['error']}]")
    return "".join(parts)


def _format_tool_result_text(name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    ok = "ok" if result.get("success") else "fail"
    head = f"[{name}] {ok}"
    if result.get("error"):
        head += f" error={result['error']}"
    if result.get("clamped"):
        head += " (coords clamped)"
    body = result.get("text") or ""
    # Note: we don't forward image_b64 here; the next user turn carries the
    # fresh screenshot. Zoom's cropped image is also mirrored into the
    # next user turn via the caller if ever needed.
    return f"{head}\n{body}".strip()


def _parse_text_verdict(text: str) -> dict[str, Any]:
    """Fallback parser for when the model returns a text verdict instead of calling report_verdict."""
    verdict = "insufficient_evidence"
    confidence = 0.0
    evidence_sufficient = False
    reasoning = text

    for line in text.split("\n"):
        s = line.strip()
        u = s.upper()
        if u.startswith("VERDICT:"):
            raw = s.split(":", 1)[1].strip().lower()
            if raw in ("partial", "partially_supported"):
                raw = "contradicted"
            if raw in ("supported", "contradicted",
                       "insufficient_evidence", "not_applicable"):
                verdict = raw
        elif u.startswith("CONFIDENCE:"):
            try:
                confidence = max(0.0, min(1.0, float(s.split(":", 1)[1].strip())))
            except ValueError:
                pass
        elif u.startswith("EVIDENCE_SUFFICIENT:"):
            val = s.split(":", 1)[1].strip().lower()
            evidence_sufficient = val in ("true", "yes", "1")
        elif u.startswith("REASONING:"):
            reasoning = s.split(":", 1)[1].strip()

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence_sufficient": evidence_sufficient,
    }


def _verdict_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    verdict = str(payload.get("verdict", "insufficient_evidence"))
    if verdict in ("partial", "partially_supported"):
        verdict = "contradicted"
    if verdict not in ("supported", "contradicted",
                       "insufficient_evidence", "not_applicable"):
        verdict = "insufficient_evidence"
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": str(payload.get("reasoning") or ""),
        "evidence_sufficient": bool(payload.get("evidence_sufficient", False)),
    }


def _compute_action_summary(action_trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts over an ``action_trace``.

    Returns a flat dict with:

    - ``by_tool``: per-tool counter (raw tool names, including sentinels
      such as ``__screenshot__`` / ``__cursor__`` / ``__unknown__`` so
      downstream readers can spot abnormal patterns).
    - ``total_steps``: same as ``len(action_trace)``.
    - ``failed_count`` / ``clamped_count`` / ``synthetic_count`` /
      ``unknown_count``: diagnostic counters. ``synthetic`` is only ever
      set by the native adapter's text-action salvage path; FC traces
      will report 0 for it. ``clamped`` is currently only populated by
      the FC path because ``ActionExecutor`` flags it there; the field
      will simply read 0 on native traces.

    The summary is intentionally derivable purely from ``action_trace``
    (no Playwright / LLM dependencies) so it can be re-computed by any
    consumer that only persists the trace.
    """
    by_tool: dict[str, int] = {}
    failed = 0
    clamped = 0
    synthetic = 0
    unknown = 0
    for entry in action_trace:
        tool = str(entry.get("tool") or "")
        by_tool[tool] = by_tool.get(tool, 0) + 1
        if not bool(entry.get("success", True)):
            failed += 1
        if bool(entry.get("clamped")):
            clamped += 1
        if bool(entry.get("synthetic")):
            synthetic += 1
        if tool == "__unknown__":
            unknown += 1
    return {
        "by_tool": by_tool,
        "total_steps": len(action_trace),
        "failed_count": failed,
        "clamped_count": clamped,
        "synthetic_count": synthetic,
        "unknown_count": unknown,
    }


def _compose_result(
    verdict: dict[str, Any],
    action_trace: list[dict[str, Any]],
    screenshots: list[str],
    raw_text: str,
) -> dict[str, Any]:
    return {
        "verdict": verdict["verdict"],
        "confidence": float(verdict.get("confidence", 0.0)),
        "reason": verdict.get("reasoning", ""),
        "evidence_sufficient": bool(verdict.get("evidence_sufficient", False)),
        "action_trace": action_trace,
        "action_summary": _compute_action_summary(action_trace),
        "screenshots": screenshots,
        "cua_reasoning": raw_text,
        "raw_response": raw_text,
    }


def _sha1(data: str) -> str:
    return hashlib.sha1(data.encode("utf-8", errors="ignore")).hexdigest()


def _trim_screenshots_in_history(
    messages: list[dict[str, Any]],
    keep_last: int = 2,
) -> None:
    """Drop base64 image payloads from older user turns in-place.

    The CUA loop appends a fresh screenshot to the message history every
    iteration; this lets the model "see" the page state but causes the
    request payload to grow linearly. We keep the screenshot on the most
    recent ``keep_last`` user turns and replace older ones with a short
    text placeholder so the model still has structural context but the
    big base64 blobs are dropped.
    """
    if keep_last < 0:
        keep_last = 0

    user_image_indices: list[int] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        ):
            user_image_indices.append(idx)

    if len(user_image_indices) <= keep_last:
        return

    to_strip = user_image_indices[: len(user_image_indices) - keep_last]
    for idx in to_strip:
        msg = messages[idx]
        new_content: list[dict[str, Any]] = []
        stripped = False
        for part in msg["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                stripped = True
                continue
            new_content.append(part)
        if stripped:
            new_content.append({
                "type": "text",
                "text": "[earlier screenshot omitted to save tokens]",
            })
        msg["content"] = new_content


def _args_preview(args: dict[str, Any], maxlen: int = 80) -> str:
    """Compact single-line rendering of tool args for logs."""
    try:
        s = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(args)
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"
