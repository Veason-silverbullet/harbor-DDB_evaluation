"""
OpenAI native computer-use adapter (Responses API + GA ``computer`` tool).

Mirror of :mod:`cua.native_adapter` but for OpenAI GPT-5.5 (and any
future GPT model trained for the GA computer tool). Shares the same
:class:`cua.tools.executor.ActionExecutor` / :class:`cua.playwright_env.PlaywrightEnv`
backend so action execution and screenshot capture behave identically
across the three protocols.

Loop shape (mirrors the official guide
https://platform.openai.com/docs/guides/tools-computer-use):

    OpenAI computer_call(actions=[click(x,y), type(text), scroll(...), ...])
           │
           ▼  normalize to ActionExecutor calls
    ActionExecutor.<handler>(args) for each action in actions[] (in order)
           │
           ▼  capture ONE fresh screenshot, wrap as computer_call_output
    responses.create(previous_response_id=..., input=[{type:"computer_call_output", ...}])

Reasoning continuity — two modes (``cfg.reasoning_passthrough``):

* **Server-side state (default, ``reasoning_passthrough=False``)** — we
  send ``previous_response_id`` and let OpenAI preserve the conversation
  state, including ``reasoning`` items (the model's internal
  chain-of-thought), at no extra upload cost. Correct ONLY when the
  endpoint actually persists responses (genuine ``api.openai.com`` /
  Azure OpenAI). Most third-party Responses relays are stateless, so
  ``previous_response_id`` silently resolves to nothing and the model
  loses all prior context.
* **Client-side round-trip (``reasoning_passthrough=True``)** — mirror
  the Anthropic native adapter: run statelessly with ``store=False`` +
  ``include=["reasoning.encrypted_content"]``, then echo each turn's
  ``reasoning`` items (carrying their opaque ``encrypted_content`` blob,
  the GPT-5.5 analog of Anthropic ``redacted_thinking``) back into the
  next ``input`` ourselves. Everything rides in the wire payload, so it
  works through stateless relays — at the cost of re-uploading reasoning
  items and prior screenshots each turn. Use ``probe_openai_reasoning.py``
  to verify the relay forwards ``encrypted_content`` in both directions.

If we manually rebuild ``input`` (passthrough mode) we MUST transcribe
reasoning items back verbatim; missing them causes GPT-5.5 to lose state
and significantly degrades multi-step CUA performance. Under
``store=False`` a reasoning item echoed back with ``encrypted_content``
null + only its ``id`` triggers a hard ``Item with id 'rs_...' not
found`` error, which is exactly why we request the encrypted blob.

Termination:

* Model emits an ``output_message`` item that contains the
  VERDICT/CONFIDENCE/EVIDENCE_SUFFICIENT/REASONING block — we parse it
  via :func:`cua.runner._parse_text_verdict` and return.
* ``response.output`` contains no further ``computer_call`` items.
* ``max_iterations_by_claim_type[claim_type]`` is reached.

This adapter does NOT use the legacy ``computer_use_preview`` tool. If
your model id is ``computer-use-preview``, switch back to the FC path
or call the preview tool directly — schemas differ between the two.
"""
from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from .playwright_env import PlaywrightEnv
from .runner import (
    RunnerConfig,
    _compose_result,
    _compute_action_summary,
    _guideline_block,
    _parse_text_verdict,
)
from .multi_session import MultiSessionMixin, multi_session_block, veracity_block
from .transient import capture_transient_immediate
from .tools.executor import ActionExecutor
from .tools.schemas import (
    DELEGATE_TO_VIEWER_TOOL_NAME,
    SEARCH_TOOL_NAME,
    responses_function_tool_schema,
)
from types_.errors import QuotaExhaustedError, is_quota_error
import usage_tracker

logger = logging.getLogger(__name__)


# Modifier-key normalization for OpenAI computer tool. The model emits
# upper-case names like "CTRL"/"ALT"/"META"; Playwright's keyboard.down
# accepts "Control"/"Alt"/"Meta". Anything we don't recognize is passed
# through unchanged so newer/unusual modifiers still work.
_OAI_MOD_MAP = {
    "CTRL": "Control",
    "CONTROL": "Control",
    "ALT": "Alt",
    "OPTION": "Alt",
    "SHIFT": "Shift",
    "META": "Meta",
    "CMD": "Meta",
    "COMMAND": "Meta",
    "SUPER": "Meta",
    "WIN": "Meta",
}


def _norm_mods(keys: Any) -> list[str]:
    if not isinstance(keys, (list, tuple)):
        return []
    out: list[str] = []
    for k in keys:
        if not isinstance(k, str):
            continue
        out.append(_OAI_MOD_MAP.get(k.upper(), k))
    return out


# Special-key name normalization for the ``keypress`` action. Covers
# both Anthropic / OpenAI's common alias set (ENTER/RETURN/arrows/...)
# AND the modifier-key family (CTRL/ALT/META/...) so combo keypresses
# like ["CTRL","C"] -> "Control+C" deserialize correctly for Playwright
# (which expects exactly those capitalized canonical names). Without
# the modifier rows here we previously emitted "CTRL+C", which
# Playwright treats as a literal key sequence and silently no-ops.
_OAI_KEY_MAP = {
    # Editing / control keys.
    "ENTER": "Enter",
    "RETURN": "Enter",
    "TAB": "Tab",
    "ESC": "Escape",
    "ESCAPE": "Escape",
    "BACKSPACE": "Backspace",
    "DELETE": "Delete",
    "DEL": "Delete",
    "SPACE": "Space",
    "HOME": "Home",
    "END": "End",
    "PAGEUP": "PageUp",
    "PAGE_UP": "PageUp",
    "PAGEDOWN": "PageDown",
    "PAGE_DOWN": "PageDown",
    # Arrows.
    "ARROWUP": "ArrowUp",
    "ARROW_UP": "ArrowUp",
    "UP": "ArrowUp",
    "ARROWDOWN": "ArrowDown",
    "ARROW_DOWN": "ArrowDown",
    "DOWN": "ArrowDown",
    "ARROWLEFT": "ArrowLeft",
    "ARROW_LEFT": "ArrowLeft",
    "LEFT": "ArrowLeft",
    "ARROWRIGHT": "ArrowRight",
    "ARROW_RIGHT": "ArrowRight",
    "RIGHT": "ArrowRight",
    # Modifiers. Mirrored from _OAI_MOD_MAP so a combo keypress
    # arrives at Playwright in canonical form.
    "CTRL": "Control",
    "CONTROL": "Control",
    "ALT": "Alt",
    "OPTION": "Alt",
    "SHIFT": "Shift",
    "META": "Meta",
    "CMD": "Meta",
    "COMMAND": "Meta",
    "SUPER": "Meta",
    "WIN": "Meta",
}


def _norm_key(k: str) -> str:
    """Map an OpenAI-style key name to its Playwright equivalent."""
    if not isinstance(k, str):
        return ""
    return _OAI_KEY_MAP.get(k.upper(), k)


# ── Goal prompt ────────────────────────────────────────────────
#
# OpenAI's GA computer tool does not have a built-in "report verdict"
# affordance, so we instruct the model to emit a structured text block
# when it is done, just like the Anthropic native adapter does. The
# orchestrator's downstream parsers (``_parse_text_verdict``) already
# accept this exact shape.
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
1. Look at the current screenshot of the {w}x{h} viewport.
2. Call the `computer` tool with one or more actions (click / double_click /
   scroll / type / keypress / drag / move / wait / screenshot). Multiple
   actions in a single turn are allowed and encouraged when they form a
   coherent micro-plan.
3. Only base your judgement on what is directly observable in the screenshots.
4. When you have enough evidence, STOP calling the computer tool and send
   a final text message in EXACTLY this format:

       VERDICT: <supported|contradicted|insufficient_evidence|not_applicable>
       CONFIDENCE: <0..1 float>
       EVIDENCE_SUFFICIENT: <true|false>
       REASONING: <one or two sentences>

   Do NOT mix the VERDICT message with a computer_call in the same turn.

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
      placeholder → emit:
          VERDICT: insufficient_evidence
          CONFIDENCE: <=0.45
          EVIDENCE_SUFFICIENT: false
          REASONING: container present, data empty because
                     prerequisite not exercised.
    * The prerequisite is a long timer/countdown/cooldown (e.g. a
      25-min focus timer must finish before a counter updates): START
      it, then fast-forward via an `advance_time` action
      (minutes=25) to skip to completion, and re-check. Only if the
      data is STILL empty afterwards →
          VERDICT: insufficient_evidence
          CONFIDENCE: <=0.45
          EVIDENCE_SUFFICIENT: false
    * Any other prerequisite that cannot be exercised inside the
      iteration budget →
          VERDICT: insufficient_evidence
          CONFIDENCE: <=0.45
          EVIDENCE_SUFFICIENT: false
    * Reserve `contradicted` for cases where the claimed UI section
      is truly ABSENT (no such card/region anywhere on the page) or
      shows clearly unrelated content. Empty-state + correct title
      is NOT contradicted.

## Timers & countdowns — fast-forward instead of waiting
- The page runs on a controllable VIRTUAL CLOCK. To skip a long
  countdown/timer/cooldown to its END instantly (so you never wait
  real minutes), emit a computer action of type `advance_time` with
  `minutes` / `seconds` (e.g. {"type":"advance_time","minutes":25}).
  All in-page timers and Date/now reads jump forward together so the
  app renders its real end-state (00:00, completion popup, +1 round).
- ONLY advance AFTER the timer is started. If your runtime cannot emit
  `advance_time`, fall back to the empty-state rule above
  (insufficient_evidence) rather than burning the whole budget on
  `wait`.

## Visual time-series claims (READ if claim_type == "visual_semantics")
- For claims about a UI element CHANGING OVER TIME (countdown ring
  decreasing, progress bar advancing, animation direction, blinking,
  carousel auto-rotation, etc.), pure repeated `wait` is NOT useful
  evidence — the judge will see your screenshots, not your reasoning.
- Recommended micro-plan (3-5 actions total, NOT 10):
    1. Trigger the relevant interaction (e.g. click "Start" on the
       timer). One short `wait` (1-3s) is fine to let the UI settle.
    2. Take a `screenshot` and mentally note the visible state
       (digits, ring orientation, percentage) for later comparison.
    3. `wait` 5-15s of real time so the UI has time to visibly change.
    4. Take a new `screenshot` and compare against step-2 state.
    5. Send the final VERDICT block based on whichever direction you
       saw. The judge will independently receive start/mid/end frames
       and corroborate.
- Do NOT keep `wait`ing past step 4. If two waits have produced no
  visible change relevant to the claim, send VERDICT=
  insufficient_evidence rather than burning the remaining iterations.

## Budget discipline (MUST follow)
- You have a limited iteration budget. If iterations run out before
  you emit a VERDICT message, the CALLER silently fails this claim —
  strictly worse than guessing.
- Therefore: emit your VERDICT message BEFORE the last turn of the
  budget, even when not fully sure. The correct uncertain fallback is:
      VERDICT: insufficient_evidence
      CONFIDENCE: <=0.4
      EVIDENCE_SUFFICIENT: false
      REASONING: <what you saw, and what was missing>
- Plan accordingly: do not stack many `wait` actions in a row. Two
  well-placed waits + a VERDICT block always beats ten waits +
  timeout.

## Reasoning style (MUST follow)
- Write REASONING in the SAME language as the Claim above:
  Chinese claim -> Chinese reasoning; English claim -> English reasoning.
  Do not mix languages in a single reasoning.
- Length: 1-2 sentences total. State (a) what you observed and (b) why it
  does or does not support the claim.
- Do NOT exceed 2 sentences. Do NOT collapse to a bare verdict word.

## Page-load failure handling (J1)
- If the viewport is BLANK / pure white / shows only an error banner
  ("Cannot GET /", "ERR_CONNECTION_RESET", a stack trace, an empty body)
  and stays unchanged after a `wait` action, issue ONE reload-equivalent
  action chain (keypress F5, or scroll to top + screenshot) before
  giving up.
- After that one reload, if the page is STILL blank or still shows the
  same error banner, set VERDICT according to claim_type:
    * `contradicted` (confidence 0.85+) for presence / navigation /
      visual_semantics / interaction_accessibility claims — a dead page
      is direct evidence that the claimed UI does NOT exist.
    * `insufficient_evidence` for computation / persistence /
      data_integration / runtime_health claims.
  In REASONING, include the literal phrase `page_load_failed` so the
  orchestrator can correlate. The orchestrator will run additional
  reload retries on its own; you only need to attempt one yourself.
"""


# ── Action translation ────────────────────────────────────────


def _translate_action(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Translate one OpenAI computer-tool action dict into (tool_name, args).

    Returns ``("__screenshot__", {})`` when the model just wants a fresh
    screenshot (we don't actually execute anything on the page; the
    adapter loop captures + replies on its own). Returns
    ``("__unknown__", {"action": <raw>})`` for action types our
    executor cannot handle so the caller can echo an error string back.
    """
    a = (action.get("type") or "").lower()

    if a == "click":
        # OpenAI uses ``button`` values: "left" / "right" / "wheel" /
        # "back" / "forward". Playwright accepts "left"/"right"/"middle";
        # "wheel" maps to a middle-click style and we fold "back"/
        # "forward" onto the closest mouse button so the action at least
        # registers (Playwright has no first-class back/forward button).
        button = (action.get("button") or "left").lower()
        if button == "wheel":
            button = "middle"
        elif button in ("back", "forward"):
            button = "left"
        return "click", {
            "x": action.get("x", 0),
            "y": action.get("y", 0),
            "button": button,
            "clicks": 1,
            "modifiers": _norm_mods(action.get("keys")),
        }
    if a == "double_click":
        return "click", {
            "x": action.get("x", 0),
            "y": action.get("y", 0),
            "button": "left",
            "clicks": 2,
            "modifiers": _norm_mods(action.get("keys")),
        }
    if a == "scroll":
        sx = int(action.get("scroll_x") or 0)
        sy = int(action.get("scroll_y") or 0)
        # We translate (scroll_x, scroll_y) deltas into one direction +
        # discrete "amount" of mouse-wheel ticks (100 px / tick on our
        # executor). Vertical takes precedence when both are non-zero;
        # the model rarely emits combined diagonals in practice.
        if abs(sy) >= abs(sx) and sy != 0:
            direction = "down" if sy > 0 else "up"
            amount = max(1, round(abs(sy) / 100))
        elif sx != 0:
            direction = "right" if sx > 0 else "left"
            amount = max(1, round(abs(sx) / 100))
        else:
            direction = "down"
            amount = 1
        return "scroll", {
            "x": action.get("x", 0),
            "y": action.get("y", 0),
            "direction": direction,
            "amount": amount,
            "modifiers": _norm_mods(action.get("keys")),
        }
    if a == "type":
        return "type_text", {"text": action.get("text", "")}
    if a == "wait":
        # OpenAI's wait action has no duration field; default to 1s
        # which matches Anthropic's implicit behavior on the same tool.
        return "wait", {"duration_ms": int(action.get("duration_ms") or 1000)}
    if a == "keypress":
        # keys is always an array. Single key -> press_key; multi-key
        # combos (e.g. CTRL+L) -> use the executor's hold_key on the
        # last key with modifiers prefixed via clicks. We approximate
        # the combo by sending the joined Playwright "+"-separated form
        # which page.keyboard.press accepts natively (e.g. "Control+L").
        raw_keys = action.get("keys") or []
        if not isinstance(raw_keys, list):
            raw_keys = [str(raw_keys)]
        normed = [_norm_key(str(k)) for k in raw_keys if str(k).strip()]
        if not normed:
            return "__unknown__", {"action": "keypress(empty)"}
        if len(normed) == 1:
            return "press_key", {"key": normed[0]}
        return "press_key", {"key": "+".join(normed)}
    if a == "drag":
        path = action.get("path") or []
        if not isinstance(path, list) or len(path) < 2:
            return "__unknown__", {"action": "drag(<2 path points)"}
        start = path[0]
        end = path[-1]
        return "drag", {
            "start_x": int(start.get("x", 0) or 0),
            "start_y": int(start.get("y", 0) or 0),
            "end_x": int(end.get("x", 0) or 0),
            "end_y": int(end.get("y", 0) or 0),
            "button": "left",
        }
    if a == "move":
        return "move", {
            "x": int(action.get("x", 0) or 0),
            "y": int(action.get("y", 0) or 0),
        }
    if a == "screenshot":
        return "__screenshot__", {}
    if a == "advance_time":
        # Not part of OpenAI's official computer-tool action types; best-effort
        # support in case the model emits it (or a wrapper injects it) to
        # fast-forward a long countdown/timer instead of waiting real time.
        return "advance_time", {
            "minutes": action.get("minutes", 0) or 0,
            "seconds": action.get("seconds", 0) or 0,
        }

    return "__unknown__", {"action": a or "(missing)"}


class NativeOpenAIRunner(MultiSessionMixin):
    """Drop-in alternative to ``CUARunner`` for the OpenAI native CUA path.

    Uses :class:`openai.OpenAI` (Responses API) with the GA
    ``{"type": "computer"}`` tool. Continuation uses
    ``previous_response_id`` so the server preserves reasoning items
    across turns.

    Inherits viewer-subagent + veracity-search plumbing from
    :class:`cua.multi_session.MultiSessionMixin`.
    """

    def __init__(
        self,
        env: PlaywrightEnv,
        config: RunnerConfig,
        executor: ActionExecutor | None = None,
    ) -> None:
        self._env = env
        self._cfg = config
        self._executor = executor or ActionExecutor(
            env, post_action_delay_ms=config.post_action_delay_ms
        )
        self._ms_reset()

    # ── public entry ────────────────────────────────────────────

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
        ms_active = self._ms_armed(multi_session)
        veracity_active = self._veracity_armed(veracity_check)
        self._ms_reset(
            query_id=query_id, claim_id=claim_id, app_url=app_url,
            marker=marker, participants=participants, session_urls=session_urls,
        )
        vp = await self._env.get_viewport_size()
        w, h = vp["width"], vp["height"]
        budget = cfg.max_iterations_by_claim_type.get(claim_type, cfg.max_iterations)
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

        screenshots: list[str] = []
        action_trace: list[dict[str, Any]] = []

        screenshot_b64 = await self._env.screenshot_base64()
        await self._save_step_screenshot(query_id, claim_id, 0, screenshots)

        # Tool spec for the GA ``computer`` tool. The Responses API uses
        # ``display_width`` / ``display_height`` (NOT the ``_px`` suffix
        # that the deprecated ``computer_use_preview`` tool used). The
        # ``environment`` value affects how the model interprets coords
        # and which actions it favors; we pin "browser" since our
        # ActionExecutor drives a Playwright Chromium instance.
        computer_tool: dict[str, Any] = {
            "type": "computer",
            "display_width": w,
            "display_height": h,
            "environment": "browser",
        }
        # Opt-in custom function tools hung next to the GA computer tool.
        tools_for_request: list[dict[str, Any]] = [computer_tool]
        if ms_active:
            tools_for_request.append(
                responses_function_tool_schema(DELEGATE_TO_VIEWER_TOOL_NAME)
            )
        if veracity_active:
            tools_for_request.append(
                responses_function_tool_schema(SEARCH_TOOL_NAME)
            )

        # ``reasoning`` is passed on EVERY request so behavior is stable
        # whether we send a fresh ``input`` or a ``previous_response_id``
        # continuation. Empty effort/summary fields are dropped (server
        # defaults stay in effect).
        reasoning_kwargs: dict[str, str] = {}
        if cfg.reasoning_effort:
            reasoning_kwargs["effort"] = cfg.reasoning_effort
        if cfg.reasoning_summary:
            reasoning_kwargs["summary"] = cfg.reasoning_summary

        # Reasoning round-trip mode. When enabled we stop relying on
        # server-side state and instead echo reasoning items ourselves
        # (see module docstring). ``include`` asks the server for the
        # encrypted CoT blob; ``store=False`` makes the call stateless so
        # the only place that state can live is the payload WE control.
        passthrough = bool(getattr(cfg, "reasoning_passthrough", False))
        include = ["reasoning.encrypted_content"] if passthrough else None
        store = False if passthrough else None

        client = self._make_openai_client()

        # Diagnostics: counts of computer_calls and reasoning summaries.
        # The orchestrator surfaces these in the per-claim JSON so we can
        # tell at a glance whether GPT-5.5 actually reasoned about the
        # page (reasoning_count == 0 across all steps almost always
        # means the gateway is stripping reasoning items, similar to the
        # `synthetic_only` failure mode on the Claude native path).
        stats: dict[str, Any] = {
            "steps_total": 0,
            "computer_call_steps": 0,
            "computer_call_count": 0,
            "action_count": 0,
            "reasoning_item_count": 0,
            # How many reasoning items came back carrying a non-empty
            # ``encrypted_content`` blob. In passthrough mode this should
            # track ``reasoning_item_count``; a gap (count>0 but
            # encrypted==0) means the relay is stripping the blob and the
            # next-turn echo will fail server-side validation.
            "encrypted_reasoning_count": 0,
            # How many reasoning items we echoed back into ``input``
            # across the whole run (passthrough mode only).
            "reasoning_carried_back": 0,
            "reasoning_passthrough": passthrough,
            "text_only_steps": 0,
        }
        # Verbatim reasoning items (summary + encrypted_content), persisted
        # into the result JSON so trajectories can be replayed later.
        thinking_trace: list[dict[str, Any]] = []

        # First turn: send the goal text + initial screenshot. Subsequent
        # turns will use ``previous_response_id`` so the server retains
        # reasoning state.
        initial_input: list[dict[str, Any]] = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": goal},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                    "detail": "original",
                },
            ],
        }]

        previous_response_id: str | None = None
        # ``next_input`` is what we send on the upcoming request.
        #
        # * Server-side mode: on turn 1 it is the goal+screenshot; on
        #   later turns it is just a ``computer_call_output`` referencing
        #   the last computer_call's call_id (the server holds the rest).
        # * Passthrough mode: ``next_input`` IS the full running
        #   conversation. We append the model's output items (reasoning +
        #   computer_call + message) and the computer_call_output to it
        #   each turn, and resend the whole thing (``previous_response_id``
        #   stays None). This is what carries the reasoning blob back.
        next_input: list[dict[str, Any]] = list(initial_input)
        final_text = ""

        for step in range(1, budget + 1):
            logger.info(
                "OpenAINativeCUA step %d/%d claim=%s [%s] mode=%s",
                step, budget, claim_id, claim_type,
                "passthrough" if passthrough else "server_state",
            )
            try:
                response = await self._responses_create(
                    client=client,
                    tools=tools_for_request,
                    reasoning_kwargs=reasoning_kwargs,
                    previous_response_id=None if passthrough else previous_response_id,
                    inp=next_input,
                    store=store,
                    include=include,
                )
            except QuotaExhaustedError:
                raise
            except Exception as e:
                logger.error("OpenAI Responses call failed: %s", e)
                return _error_result(
                    f"OpenAI Responses error: {type(e).__name__}: {e}",
                    action_trace, screenshots,
                )

            previous_response_id = getattr(response, "id", None) or previous_response_id

            output_items = list(getattr(response, "output", []) or [])
            computer_calls = [
                it for it in output_items
                if getattr(it, "type", None) == "computer_call"
            ]
            function_calls = [
                it for it in output_items
                if getattr(it, "type", None) == "function_call"
            ]
            messages = [
                it for it in output_items
                if getattr(it, "type", None) == "message"
            ]
            reasoning_items = [
                it for it in output_items
                if getattr(it, "type", None) == "reasoning"
            ]

            stats["steps_total"] += 1
            stats["reasoning_item_count"] += len(reasoning_items)
            for _ri in reasoning_items:
                if _coerce_action_dict(_ri).get("encrypted_content"):
                    stats["encrypted_reasoning_count"] += 1
            if reasoning_items:
                thinking_trace.extend(
                    _reasoning_items_to_trace(step, reasoning_items)
                )

            # In passthrough mode, append THIS turn's model output
            # (reasoning + computer_call + message items) to the running
            # conversation BEFORE we either break or build the next
            # computer_call_output. Echoing them back verbatim is what
            # preserves the reasoning chain when ``store=False``.
            if passthrough:
                # Once a decrypt failure has degraded this run, stop echoing
                # reasoning items back — they would only fail decryption
                # again next turn. Tool calls / messages are still echoed.
                degraded = getattr(self, "_reasoning_degraded", False)
                for it in output_items:
                    d = _coerce_action_dict(it)
                    if degraded and d.get("type") == "reasoning":
                        continue
                    next_input.append(d)
                if not degraded:
                    stats["reasoning_carried_back"] += len(reasoning_items)

            # 0) Dispatch any custom function tools (viewer subagent /
            #    veracity search). These are plain Responses function calls
            #    answered with a ``function_call_output`` item; they may
            #    arrive with or without a computer_call in the same turn.
            fc_outputs: list[dict[str, Any]] = []
            for fc in function_calls:
                d = _coerce_action_dict(fc)
                fname = str(d.get("name") or "")
                fcall_id = d.get("call_id")
                try:
                    fargs = json.loads(d.get("arguments") or "{}")
                    if not isinstance(fargs, dict):
                        fargs = {}
                except (json.JSONDecodeError, TypeError):
                    fargs = {}
                if fname == DELEGATE_TO_VIEWER_TOOL_NAME:
                    r = await self._delegate_to_viewer(fargs)
                elif fname == SEARCH_TOOL_NAME:
                    r = await self._run_search(fargs)
                else:
                    r = {"success": False, "error": f"unknown tool: {fname}",
                         "text": f"unknown tool: {fname}"}
                success = bool(r.get("success"))
                out_text = r.get("text") or (
                    "ok" if success else (r.get("error") or "failed")
                )
                action_trace.append({
                    "step": step, "tool": fname, "action": fname,
                    "input": fargs, "success": success, "error": r.get("error"),
                })
                fc_outputs.append({
                    "type": "function_call_output",
                    "call_id": fcall_id,
                    "output": out_text,
                })

            # 1) Final-answer turn: no computer_call AND no function_call ->
            #    parse VERDICT text out of any output_message items and return.
            if not computer_calls and not function_calls:
                stats["text_only_steps"] += 1
                final_text = _extract_message_text(messages) or final_text
                break

            # Function-only turn (no computer_call): feed the tool outputs
            # back and continue — do NOT try to build a computer_call_output.
            if not computer_calls:
                extra_msg = _extract_message_text(messages)
                if extra_msg:
                    final_text = extra_msg
                if passthrough:
                    next_input.extend(fc_outputs)
                else:
                    next_input = list(fc_outputs)
                continue

            stats["computer_call_steps"] += 1
            stats["computer_call_count"] += len(computer_calls)

            # 2) Execute every action of every computer_call in order.
            #    Multiple computer_calls per turn is rare but legal; we
            #    only reply to the LAST one's call_id because the
            #    Responses API expects a single computer_call_output per
            #    outstanding computer_call. In practice we have only
            #    ever observed one computer_call per turn.
            last_call = computer_calls[-1]
            last_call_id = getattr(last_call, "call_id", None)

            # Collect action text from message blocks as a partial text
            # trail for the final reasoning fallback. If the model also
            # emits a message alongside the computer_call (rare) we keep
            # it as breadcrumbs.
            extra_msg = _extract_message_text(messages)
            if extra_msg:
                final_text = extra_msg

            for cc in computer_calls:
                actions = list(getattr(cc, "actions", []) or [])
                for raw_action in actions:
                    action_dict = _coerce_action_dict(raw_action)
                    stats["action_count"] += 1
                    tool_name, args = _translate_action(action_dict)

                    if tool_name == "__screenshot__":
                        action_trace.append({
                            "step": step,
                            "tool": tool_name,
                            "action": "screenshot",
                            "input": action_dict,
                            "success": True,
                            "error": None,
                        })
                        continue

                    if tool_name == "__unknown__":
                        err = f"unsupported OpenAI action: {args.get('action')}"
                        logger.warning(err)
                        action_trace.append({
                            "step": step,
                            "tool": tool_name,
                            "action": str(args.get("action") or "(unknown)"),
                            "input": action_dict,
                            "success": False,
                            "error": err,
                        })
                        continue

                    handler = getattr(self._executor, tool_name, None)
                    if handler is None:
                        action_trace.append({
                            "step": step,
                            "tool": tool_name,
                            "action": tool_name,
                            "input": action_dict,
                            "success": False,
                            "error": f"no handler for {tool_name}",
                        })
                        continue

                    try:
                        r = await handler(args)
                    except Exception as ex:  # noqa: BLE001
                        r = {
                            "success": False,
                            "error": f"{type(ex).__name__}: {ex}",
                            "clamped": False,
                        }
                    action_trace.append({
                        "step": step,
                        "tool": tool_name,
                        "action": tool_name,
                        "input": action_dict,
                        "success": bool(r.get("success")),
                        "error": r.get("error"),
                        "clamped": bool(r.get("clamped")),
                    })

            # 3) Acknowledge any pending safety checks the model raised
            #    on the last computer_call. We auto-acknowledge inside
            #    this evaluation harness because the target apps are
            #    sandboxed seed apps we control; a production agent
            #    would defer to a human here.
            pending_checks = getattr(last_call, "pending_safety_checks", None) or []
            ack_checks: list[dict[str, Any]] = []
            for chk in pending_checks:
                ack_checks.append(_coerce_safety_check(chk))
            if ack_checks:
                logger.info(
                    "OpenAINativeCUA: auto-acknowledging %d safety check(s) on "
                    "claim=%s step=%d (sandboxed eval env)",
                    len(ack_checks), claim_id, step,
                )

            # 4) Capture a fresh screenshot (the artifact + the LLM
            #    payload reuse the same bytes) and build the
            #    computer_call_output that will become next turn's input.
            #    For transient-feedback claims (弹幕/toast) also grab ONE
            #    immediate (non-settling) frame into the saved screenshots
            #    so a fast-disappearing element enters the judge's pool.
            await capture_transient_immediate(
                self._env, cfg, query_id, claim_id, step, screenshots, claim_type,
            )
            new_b64 = await self._env.screenshot_base64()
            await self._save_step_screenshot(query_id, claim_id, step, screenshots)

            cco: dict[str, Any] = {
                "type": "computer_call_output",
                "call_id": last_call_id,
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{new_b64}",
                    "detail": "original",
                },
            }
            if ack_checks:
                cco["acknowledged_safety_checks"] = ack_checks
            if passthrough:
                # Append the tool output to the accumulating conversation;
                # the reasoning + computer_call items for this turn were
                # already appended above, so order stays
                # reasoning -> computer_call -> computer_call_output.
                next_input.extend(fc_outputs)
                next_input.append(cco)
            else:
                # Server holds the history; only the new outputs are needed.
                next_input = [*fc_outputs, cco]
        else:
            logger.warning(
                "OpenAINativeCUA max iterations reached (%d) for %s",
                budget, claim_id,
            )

        # Per-claim summary log mirrors NativeAnthropicRunner so grep
        # patterns work across protocols. `mode` is intentionally
        # different here ("native_only" vs "text_only") to surface the
        # rare case where the model never issues a computer_call.
        if stats["computer_call_steps"] > 0 and stats["text_only_steps"] > 0:
            mode = "native_with_message"
        elif stats["computer_call_steps"] > 0:
            mode = "native_only"
        else:
            mode = "text_only"
        logger.info(
            "OpenAINativeCUA claim=%s done: mode=%s steps=%d computer_calls=%d "
            "actions=%d reasoning_items=%d encrypted=%d carried_back=%d "
            "passthrough=%s",
            claim_id, mode, stats["steps_total"], stats["computer_call_count"],
            stats["action_count"], stats["reasoning_item_count"],
            stats["encrypted_reasoning_count"], stats["reasoning_carried_back"],
            passthrough,
        )
        # Loud warning if passthrough was requested but the relay never
        # returned an encrypted blob: the next-turn echo would silently
        # degrade to "no reasoning continuity" (or hard-fail under
        # store=False). Mirrors the Anthropic adapter's thinking_steps==0
        # gateway-strip detector.
        if (
            passthrough
            and stats["reasoning_item_count"] > 0
            and stats["encrypted_reasoning_count"] == 0
        ):
            logger.warning(
                "OpenAINativeCUA claim=%s: reasoning_passthrough is ON and the "
                "model returned %d reasoning item(s) but ZERO carried "
                "encrypted_content. The relay is stripping the encrypted CoT "
                "blob (include=reasoning.encrypted_content not honored); "
                "round-trip cannot work. Cross-check with "
                "probe_openai_reasoning.py.",
                claim_id, stats["reasoning_item_count"],
            )

        result = _compose_result(
            _parse_text_verdict(final_text),
            action_trace, screenshots, final_text,
        )
        # We reuse the ``native_tool_use_stats`` field so downstream
        # reports (which already render it for Claude native) display
        # GPT-5.5 stats too. Field semantics are kept compatible:
        # ``steps_total`` and a ``mode`` are present in both; extras
        # are protocol-specific.
        result["native_tool_use_stats"] = {
            **stats,
            "mode": mode,
            "protocol": "native_computer_openai",
        }
        result["thinking_trace"] = thinking_trace
        return result

    # ── helpers ─────────────────────────────────────────────────

    async def _responses_create(
        self,
        client,
        tools: list[dict[str, Any]],
        reasoning_kwargs: dict[str, str],
        previous_response_id: str | None,
        inp: list[dict[str, Any]],
        store: bool | None = None,
        include: list[str] | None = None,
    ):
        """Call ``client.responses.create`` with retries on transient errors.

        We wrap the sync OpenAI client call in ``run_in_executor`` so it
        does not block the event loop.

        ``store`` / ``include`` are only set in reasoning-passthrough mode:
        ``store=False`` + ``include=["reasoning.encrypted_content"]`` make
        the call stateless and ask the server to hand back the encrypted
        chain-of-thought so we can echo it into the next ``input``. Both
        are omitted (server defaults) otherwise.
        """
        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        # ``truncation`` was REQUIRED for the deprecated computer-use-
        # preview integration but is "not necessary" for the GA
        # ``computer`` tool (per OpenAI's migration table). Some gateway
        # proxies (notably az.gptplus5) ship a strict schema validator
        # that REJECTS the field outright with "Unsupported parameter:
        # truncation", so we omit it by default and only re-add it if
        # we observe an upstream that explicitly asks for it. If you
        # need it later, plumb a config knob — do NOT hard-code it back
        # in or the strict gateways will break again.
        attempts_left = self._cfg.request_retries + 1
        # Extra retries reserved ONLY for the decrypt-degradation path
        # below; they do not consume the normal transient-retry budget.
        degrade_retries_left = 2
        while attempts_left > 0:
            attempts_left -= 1
            kwargs: dict[str, Any] = {
                "model": self._cfg.model,
                "tools": list(tools),
                "input": inp,
            }
            if reasoning_kwargs:
                kwargs["reasoning"] = dict(reasoning_kwargs)
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            if include:
                kwargs["include"] = list(include)
            if store is not None:
                kwargs["store"] = bool(store)
            if self._cfg.extra_headers:
                kwargs["extra_headers"] = dict(self._cfg.extra_headers)
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: client.responses.create(**kwargs),
                )
                usage_tracker.record_response(
                    "cua", response, model=self._cfg.model,
                )
                return response
            except Exception as e:  # noqa: BLE001
                if is_quota_error(e):
                    raise QuotaExhaustedError("cua", str(e)) from e
                last_exc = e
                logger.warning(
                    "responses.create attempt failed (%d normal retr(ies) left): %s",
                    attempts_left, e,
                )
                msg = str(e).lower()
                # Code path: pyopenai surfaces HTTP code on BadRequestError;
                # some gateways wrap a 400 in a 500 with the code/message in
                # the JSON body, so we also pattern-match the text.
                code = getattr(e, "status_code", None) or getattr(e, "code", None)

                # ── Decrypt-degradation (relay key-multiplexing) ──────
                # A 400 "encrypted content could not be decrypted/verified"
                # means the echoed reasoning blob (store=False passthrough)
                # was minted by a DIFFERENT upstream org than the one
                # handling this turn. Strip the echoed reasoning items in
                # place (so the caller's accumulating conversation also
                # stops carrying them) and retry the SAME turn, degrading
                # to no cross-turn reasoning continuity instead of
                # hard-failing the whole claim.
                if degrade_retries_left > 0 and _is_decrypt_error(code, msg):
                    removed = _strip_encrypted_reasoning(inp)
                    if removed:
                        degrade_retries_left -= 1
                        attempts_left += 1  # do not bill the normal budget
                        self._reasoning_degraded = True
                        logger.warning(
                            "responses.create decrypt-degrade: stripped %d echoed "
                            "reasoning item(s) and retrying turn WITHOUT reasoning "
                            "continuity. Relay cannot decrypt cross-org encrypted "
                            "CoT — prefer base_url=official OpenAI, or set "
                            "reasoning_passthrough=false.", removed,
                        )
                        continue

                # Fail fast on permanent 4xx schema rejections — these
                # never recover via retry and otherwise stack with the SDK
                # retry layer, turning one logical failure into many wasted
                # requests (and quota hits).
                permanent_markers = (
                    "unsupported parameter",
                    "unsupported tool",          # "Unsupported tool type: computer"
                    "invalid_request_error",
                    "invalid parameter",
                    "model_not_found",
                    "no such tool",
                    "does not support",          # "model X does not support computer"
                    "upstream request failed",
                    "upstream_error",
                    "bad gateway",
                )
                if code == 400:
                    logger.error(
                        "responses.create returned HTTP 400; aborting retries. "
                        "Permanent schema error: %s", e,
                    )
                    raise
                if any(marker in msg for marker in permanent_markers):
                    logger.error(
                        "responses.create rejected our schema (markers matched); "
                        "aborting retries. Full error: %s", e,
                    )
                    raise
        assert last_exc is not None
        raise last_exc

    async def _save_step_screenshot(
        self,
        query_id: str,
        claim_id: str,
        step: int,
        screenshots: list[str],
    ) -> None:
        if not self._cfg.save_screenshots:
            return
        # Per-query subdir: <screenshot_dir>/<qid>/screenshots/<claim>_stepN.png
        # so each query owns one folder holding both result JSON and PNGs.
        path = (
            self._cfg.screenshot_dir
            / query_id
            / "screenshots"
            / f"{claim_id}_step{step}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        await self._env.save_screenshot(path)
        screenshots.append(str(path))

    def _make_openai_client(self):
        """Build the OpenAI client used for Responses API calls.

        Note: the GA ``computer`` tool is only available on the official
        OpenAI endpoint and on Azure OpenAI's matching Responses API
        deployment. Third-party Chat-Completions-only gateways (e.g.
        most ``/v1/chat/completions`` relays) WILL NOT WORK on this
        protocol — point ``cua.base_url`` at the genuine
        ``https://api.openai.com/v1`` endpoint (or leave empty for the
        SDK default), or pick a relay that explicitly supports
        ``POST /v1/responses``.
        """
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "api_key": self._cfg.api_key,
            "base_url": self._cfg.base_url or None,
            # Disable the SDK's own retry layer so it does not stack on
            # top of our ``request_retries``. The SDK default is 2 retries
            # per call which, combined with our outer loop, turned a
            # single 400 schema error into 6 visible HTTP attempts (see
            # 2026-05-20 az.gptplus5 truncation/computer-tool incidents).
            # We do all retry policy ourselves in ``_responses_create``.
            "max_retries": 0,
            # Cap each request at 90s. OpenAI's SDK default is 600s
            # (10 min) which is catastrophic on broken gateways: the
            # 2026-05-20 yibuapi incident hung for 2m45s per attempt
            # before returning 502, so a single claim with the default
            # retry policy would wait >8 minutes to fail. 90s is enough
            # for any real CUA turn (typical: 8-30s on the official
            # endpoint, up to 60s on first-turn cold starts with
            # ``reasoning_effort=high``) while bounding the failure
            # latency on dead upstreams.
            "timeout": 90.0,
        }
        if self._cfg.extra_headers:
            kwargs["default_headers"] = dict(self._cfg.extra_headers)
        return OpenAI(**kwargs)


# ── module-private helpers ──────────────────────────────────────


_DECRYPT_ERROR_MARKERS = (
    "could not be decrypted",
    "could not be verified",
    "encrypted content could not",
)


def _is_decrypt_error(code: Any, msg: str) -> bool:
    """True for the OpenAI Responses "encrypted reasoning blob cannot be
    decrypted/verified" failure.

    Happens in ``reasoning_passthrough`` mode when a multi-key relay routes
    a later turn to a DIFFERENT upstream org than the one that generated the
    echoed ``encrypted_content`` — the new org cannot decrypt it (HTTP 400).
    """
    m = (msg or "").lower()
    if "encrypted" in m and ("decrypt" in m or "could not be verified" in m):
        return True
    return any(k in m for k in _DECRYPT_ERROR_MARKERS)


def _strip_encrypted_reasoning(inp: list[Any]) -> int:
    """Remove ``reasoning`` items from an echoed ``input`` list IN PLACE.

    Returns the count removed. Mutating in place means the caller's
    accumulating conversation list (same object) also stops carrying the
    un-decryptable blobs on subsequent turns. Tool calls / outputs are kept
    so the function/computer-call chain stays consistent; only cross-turn
    reasoning continuity is dropped.
    """
    keep: list[Any] = []
    removed = 0
    for it in inp:
        t = it.get("type") if isinstance(it, dict) else getattr(it, "type", None)
        if t == "reasoning":
            removed += 1
            continue
        keep.append(it)
    if removed:
        inp[:] = keep
    return removed


def _reasoning_items_to_trace(step: int, items: list[Any]) -> list[dict[str, Any]]:
    """Serialize OpenAI Responses ``reasoning`` items for persistence.

    Mirrors the Anthropic adapters' ``_thinking_blocks_to_trace``: the
    human-readable part is the ``summary`` list (present when
    ``reasoning.summary`` is requested); the replay-critical part is the
    ``encrypted_content`` blob (present in passthrough mode with
    ``include=["reasoning.encrypted_content"]``), which only OpenAI can
    decrypt but MUST be echoed back verbatim to restore a conversation.
    Both are saved so trajectories are readable AND restorable.

    ``ts`` is the wall-clock time the item was received (i.e. when the
    LLM response for this step arrived), so trajectory latency can be
    reconstructed step-by-step.
    """
    ts = datetime.now().isoformat(timespec="milliseconds")
    out: list[dict[str, Any]] = []
    for it in items:
        d = _coerce_action_dict(it)
        parts: list[str] = []
        summary = d.get("summary") or []
        if isinstance(summary, list):
            for s in summary:
                text = (
                    s.get("text", "") if isinstance(s, dict)
                    else getattr(s, "text", "") or ""
                )
                if text:
                    parts.append(str(text))
        out.append({
            "step": step,
            "ts": ts,
            "type": "reasoning",
            "id": str(d.get("id") or ""),
            "summary": " ".join(parts),
            "encrypted_content": str(d.get("encrypted_content") or ""),
        })
    return out


def _coerce_action_dict(raw: Any) -> dict[str, Any]:
    """Best-effort conversion of an SDK action object into a plain dict.

    Newer OpenAI SDKs return strongly-typed Pydantic models for action
    items. We normalize to a plain dict so the rest of the adapter can
    treat them uniformly without binding to a specific SDK minor
    version.
    """
    if isinstance(raw, dict):
        return dict(raw)
    md = getattr(raw, "model_dump", None)
    if callable(md):
        try:
            d = md()
            if isinstance(d, dict):
                return d
        except Exception:  # noqa: BLE001
            pass
    # Last-resort: copy public attributes into a dict.
    out: dict[str, Any] = {}
    for name in dir(raw):
        if name.startswith("_"):
            continue
        try:
            val = getattr(raw, name)
        except Exception:  # noqa: BLE001
            continue
        if callable(val):
            continue
        out[name] = val
    return out


def _coerce_safety_check(raw: Any) -> dict[str, Any]:
    """Same shape-coercion idea as :func:`_coerce_action_dict`, for
    ``pending_safety_checks`` items.

    The schema the server expects back is the verbatim entry it sent us,
    so we reflect the input dict as-is (after coercion).
    """
    base = _coerce_action_dict(raw)
    # Keep only fields the server accepts back. Extra fields are
    # tolerated but the ``id`` is the only one strictly required for
    # acknowledgement.
    out = {}
    for k in ("id", "code", "message"):
        if k in base and base[k] is not None:
            out[k] = base[k]
    return out


def _extract_message_text(messages: list[Any]) -> str:
    """Concatenate text content from one or more ``message`` output items.

    Each ``message`` item has a ``content`` array of typed blocks; we
    care about ``output_text`` blocks. The text fields across SDK
    versions are either ``text`` (typed model) or ``value`` (older
    name); we accept both.
    """
    chunks: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", None) or []
        for block in content:
            btype = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if btype not in ("output_text", "text"):
                continue
            txt = getattr(block, "text", None)
            if txt is None and isinstance(block, dict):
                txt = block.get("text") or block.get("value")
            if isinstance(txt, str) and txt.strip():
                chunks.append(txt.strip())
    return "\n".join(chunks)


def _error_result(msg: str, action_trace: list[dict], screenshots: list[str]) -> dict[str, Any]:
    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.0,
        "reason": msg,
        "evidence_sufficient": False,
        "action_trace": action_trace,
        "action_summary": _compute_action_summary(action_trace),
        "screenshots": screenshots,
        "cua_reasoning": "",
        "raw_response": msg,
    }


# ── pathlib re-export to keep the public namespace tidy ────────
#
# ``RunnerConfig.screenshot_dir`` is a ``Path`` and adapter callers
# already have access to ``Path`` via :mod:`cua.runner`. Re-exporting
# here avoids an extra import on the call site.
__all__ = [
    "NativeOpenAIRunner",
    "Path",
]
