"""
Anthropic native computer_20251124 adapter.

Shares the same ActionExecutor / PlaywrightEnv backend as the function-calling
CUARunner. The adapter's job is Protocol-layer translation only:

    Claude tool_use(name="computer", input={action: "left_click", ...})
           │
           ▼  normalize to our tool call
    ActionExecutor.<handler>(args)
           │
           ▼  wrap into Anthropic tool_result (image + text)
    messages.append(user/tool_result)

Requires ``anthropic-beta: computer-use-2025-11-24`` — enable only when the
runtime endpoint is known to pass this header through.
"""
from __future__ import annotations
import json
import logging
import re
from datetime import datetime
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
    anthropic_tool_schema,
)
from types_.errors import QuotaExhaustedError, is_quota_error
import usage_tracker

logger = logging.getLogger(__name__)


# Fallback parser: when the upstream gateway/Bedrock fails to register the
# beta `computer` tool, Claude returns its intended action as JSON-shaped
# *text* with stop_reason=end_turn. We salvage those into synthetic
# tool_use payloads so the loop can still drive the browser.
#
# Supported text shapes (any of):
#   {"name": "computer", "input": {"action": ..., ...}}
#   {"action": "...", "coordinate": [x, y], ...}
# Lines that look like a `VERDICT:` block are NOT treated as actions.
_VERDICT_RE = re.compile(r"^\s*VERDICT\s*:", re.IGNORECASE | re.MULTILINE)


def _try_loads(blob: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_text_as_actions(text: str) -> list[dict[str, Any]]:
    """Best-effort extractor of computer-tool action dicts from text.

    Returns a list of ``action_input`` dicts ready for ``_translate_action``.
    Never raises; returns ``[]`` if nothing recognizable is found.
    """
    if not text or _VERDICT_RE.search(text):
        return []
    actions: list[dict[str, Any]] = []

    # Pattern 1: {"name": "computer", "input": {...}}
    for m in re.finditer(
        r"\{[^{}]*\"name\"\s*:\s*\"computer\"[^{}]*\"input\"\s*:\s*(\{[^{}]*\})",
        text, re.DOTALL,
    ):
        inp = _try_loads(m.group(1))
        if inp and "action" in inp:
            actions.append(inp)

    # Pattern 2: bare {"action": "...", ...} (e.g. FC-2-A2 / FC-3-A3 in logs)
    if not actions:
        for m in re.finditer(r"\{[^{}]*\"action\"\s*:\s*\"[a-z_]+\"[^{}]*\}", text, re.DOTALL):
            inp = _try_loads(m.group(0))
            if inp and "action" in inp:
                actions.append(inp)

    return actions


def _thinking_blocks_to_trace(step: int, blocks: list[Any]) -> list[dict[str, Any]]:
    """Serialize Anthropic thinking/redacted_thinking blocks for persistence.

    The ``signature`` token is what Anthropic requires to be echoed back
    verbatim to continue a conversation, so saving it alongside the thinking
    text is the difference between a trajectory that can be replayed and one
    that can only be read. Shared by both the beta (native) and non-beta
    (FC) Anthropic adapters.

    ``ts`` is the wall-clock time the block was received (i.e. when the
    LLM response for this step arrived), so trajectory latency can be
    reconstructed step-by-step.
    """
    ts = datetime.now().isoformat(timespec="milliseconds")
    out: list[dict[str, Any]] = []
    for b in blocks:
        btype = getattr(b, "type", "")
        if btype == "thinking":
            out.append({
                "step": step,
                "ts": ts,
                "type": "thinking",
                "thinking": getattr(b, "thinking", "") or "",
                "signature": getattr(b, "signature", "") or "",
            })
        elif btype == "redacted_thinking":
            out.append({
                "step": step,
                "ts": ts,
                "type": "redacted_thinking",
                "data": getattr(b, "data", "") or "",
            })
    return out


COMPUTER_TOOL_TYPE = "computer_20251124"
COMPUTER_BETA_FLAG = "computer-use-2025-11-24"
# COMPUTER_TOOL_TYPE = "computer_20250124"
# COMPUTER_BETA_FLAG = "computer-use-2025-01-24"


# Anthropic "computer" tool action → our per-tool handler invocation.
# Field names follow the 2025-11-24 schema (scroll_direction / scroll_amount,
# modifier `text`, etc.). Unknown fields are ignored.
def _translate_action(
    action_input: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    action = (action_input.get("action") or "").lower()
    coord = action_input.get("coordinate") or [0, 0]
    mods_text = action_input.get("text") if action in (
        "left_click", "right_click", "middle_click",
        "double_click", "triple_click", "scroll",
    ) else None
    modifiers = [mods_text] if mods_text else []

    if action in ("left_click", "click"):
        return "click", {"x": coord[0], "y": coord[1], "button": "left", "clicks": 1, "modifiers": modifiers}
    if action == "right_click":
        return "click", {"x": coord[0], "y": coord[1], "button": "right", "clicks": 1, "modifiers": modifiers}
    if action == "middle_click":
        return "click", {"x": coord[0], "y": coord[1], "button": "middle", "clicks": 1, "modifiers": modifiers}
    if action == "double_click":
        return "click", {"x": coord[0], "y": coord[1], "button": "left", "clicks": 2, "modifiers": modifiers}
    if action == "triple_click":
        return "click", {"x": coord[0], "y": coord[1], "button": "left", "clicks": 3, "modifiers": modifiers}
    if action == "mouse_move":
        return "move", {"x": coord[0], "y": coord[1]}
    if action == "left_click_drag":
        start = action_input.get("start_coordinate") or [0, 0]
        return "drag", {
            "start_x": start[0], "start_y": start[1],
            "end_x": coord[0], "end_y": coord[1], "button": "left",
        }
    if action == "left_mouse_down":
        return "mouse_down", {"x": coord[0], "y": coord[1], "button": "left"}
    if action == "left_mouse_up":
        return "mouse_up", {"x": coord[0], "y": coord[1], "button": "left"}
    if action == "scroll":
        direction = action_input.get("scroll_direction") or "down"
        amount = int(action_input.get("scroll_amount") or 3)
        return "scroll", {
            "x": coord[0], "y": coord[1],
            "direction": direction, "amount": amount, "modifiers": modifiers,
        }
    if action == "type":
        return "type_text", {"text": action_input.get("text", "")}
    if action == "key":
        return "press_key", {"key": action_input.get("text", "")}
    if action == "hold_key":
        return "hold_key", {
            "key": action_input.get("text") or action_input.get("key", ""),
            "duration_ms": int(
                (action_input.get("duration") or 0) * 1000
                if isinstance(action_input.get("duration"), (int, float))
                else action_input.get("duration_ms") or 500
            ),
        }
    if action == "wait":
        raw = action_input.get("duration") or action_input.get("duration_ms") or 500
        if isinstance(raw, float) and raw < 50:
            raw = int(raw * 1000)  # Anthropic sends seconds in some versions
        return "wait", {"duration_ms": int(raw)}
    if action == "zoom":
        region = action_input.get("region") or [0, 0, 0, 0]
        return "zoom", {"region": region}
    if action == "reload":
        # ``reload`` is not part of Anthropic's official computer-use action
        # vocabulary; we expose it via our synthetic-text path so the CUA
        # prompt can instruct the model to call it when the page looks dead.
        return "reload", {"retries": int(action_input.get("retries", 1) or 0)}
    if action == "advance_time":
        # Not part of the official vocabulary either; surfaced via the same
        # synthetic-text path so the prompt can instruct the model to
        # fast-forward a long countdown/timer instead of waiting real time.
        return "advance_time", {
            "minutes": action_input.get("minutes", 0) or 0,
            "seconds": action_input.get("seconds", 0) or 0,
        }
    if action == "screenshot":
        return "__screenshot__", {}  # sentinel: runner just returns fresh shot
    if action == "cursor_position":
        return "__cursor__", {}

    return "__unknown__", {"action": action}


class NativeAnthropicRunner(MultiSessionMixin):
    """Drop-in alternative to ``CUARunner`` when running native computer-use.

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

        goal = (
            f"You are evaluating a web app to verify:\n"
            f"CLAIM: {checkpoint}\nVERIFY: {verification}\n"
            f"TYPE: {claim_type}\nAPP: {app_url}\n"
            f"Screen: {w}x{h}. Use the computer tool to interact. "
            f"When done, send a final text message in the format:\n"
            f"VERDICT: <supported|contradicted|insufficient_evidence|not_applicable>\n"
            f"CONFIDENCE: <0..1>\nEVIDENCE_SUFFICIENT: <true|false>\nREASONING: <why>\n"
            f"\n"
            f"Reasoning style (MUST follow):\n"
            f"- Write the REASONING in the SAME language as the CLAIM above:\n"
            f"  Chinese claim -> Chinese reasoning; English claim -> English reasoning.\n"
            f"  Do not mix languages in a single reasoning.\n"
            f"- Length: 1-2 sentences total. State (a) what you observed and\n"
            f"  (b) why it does or does not support the claim.\n"
            f"- Do NOT exceed 2 sentences. Do NOT collapse to a bare verdict word.\n"
            f"\n"
            f"Empty-state vs missing-feature (read for presence claims):\n"
            f"- Many `presence` claims describe data displays that REQUIRE\n"
            f"  prior user data ('shows today's completed focus rounds',\n"
            f"  'shows session history', 'lists saved items', etc.). When the\n"
            f"  page shows a well-formed placeholder card with copy like\n"
            f"  '还没有...', '暂无...', '尚未...', '开始第一次',\n"
            f"  'no records yet', 'get started', 'empty list', that is NOT\n"
            f"  evidence the feature is missing — the container IS rendered\n"
            f"  and just has no data yet because the prerequisite user\n"
            f"  action has not occurred.\n"
            f"- Decision:\n"
            f"    * Container/region clearly matches the claim's subject\n"
            f"      AND shows an empty-state placeholder →\n"
            f"      VERDICT=insufficient_evidence, CONFIDENCE<=0.45,\n"
            f"      EVIDENCE_SUFFICIENT=false, REASONING explains 'container\n"
            f"      present, data empty because prerequisite not exercised'.\n"
            f"    * Prerequisite is a long timer/countdown/cooldown (e.g. a\n"
            f"      25-min focus timer must finish before a counter updates):\n"
            f"      START it, then fast-forward with the synthetic action\n"
            f"      {{\"name\":\"computer\",\"input\":{{\"action\":\"advance_time\",\"minutes\":25}}}}\n"
            f"      and re-check. Only if STILL empty afterwards →\n"
            f"      VERDICT=insufficient_evidence, CONFIDENCE<=0.45.\n"
            f"    * Any other prerequisite that cannot be exercised inside the\n"
            f"      iteration budget → VERDICT=insufficient_evidence,\n"
            f"      CONFIDENCE<=0.45.\n"
            f"    * Reserve `contradicted` for cases where the claimed UI\n"
            f"      section is truly ABSENT or shows clearly unrelated\n"
            f"      content. Empty-state + correct title is NOT contradicted.\n"
            f"\n"
            f"Timers & countdowns — fast-forward instead of waiting:\n"
            f"- The page runs on a controllable VIRTUAL CLOCK. To skip a long\n"
            f"  countdown/timer/cooldown to its END instantly (so you never\n"
            f"  wait real minutes), output the synthetic action:\n"
            f"  {{\"name\":\"computer\",\"input\":{{\"action\":\"advance_time\",\"minutes\":25,\"seconds\":0}}}}\n"
            f"  All in-page timers (setInterval/setTimeout/rAF) and Date/now\n"
            f"  reads jump forward together, so the app renders its real\n"
            f"  end-state (00:00, completion popup, round counter +1, etc.).\n"
            f"- ONLY advance AFTER you have started the timer; advancing a\n"
            f"  paused/unstarted timer changes nothing. For 'is it counting\n"
            f"  down' use a small advance (seconds=10) and compare frames.\n"
            f"\n"
            f"Visual time-series claims (read if TYPE == visual_semantics):\n"
            f"- For claims about a UI element changing OVER TIME (countdown ring\n"
            f"  decreasing, progress bar advancing, animation direction,\n"
            f"  blinking, carousel auto-rotation, etc.), pure repeated `wait`\n"
            f"  is NOT useful evidence on its own — the judge will see the\n"
            f"  screenshots, not your reasoning.\n"
            f"- Recommended micro-plan (3-5 actions, NOT 10):\n"
            f"    1. Trigger the relevant interaction (e.g. click \"Start\").\n"
            f"       One short wait (1-3s) is fine to let the UI settle.\n"
            f"    2. Take a `screenshot` and mentally note the state.\n"
            f"    3. `wait` 5-15s of real time so the UI visibly changes.\n"
            f"    4. Take a new `screenshot` and compare against step 2.\n"
            f"    5. Send the final VERDICT block — do NOT keep waiting.\n"
            f"\n"
            f"Budget discipline (MUST follow):\n"
            f"- You have a limited iteration budget. If iterations run out\n"
            f"  before you emit a VERDICT, the CALLER silently fails this\n"
            f"  claim — strictly worse than guessing.\n"
            f"- Therefore: emit VERDICT BEFORE the last turn of the budget,\n"
            f"  even when not fully sure. The correct uncertain fallback is\n"
            f"  VERDICT=insufficient_evidence, CONFIDENCE<=0.4,\n"
            f"  EVIDENCE_SUFFICIENT=false, with REASONING describing what\n"
            f"  you actually saw and what was missing.\n"
            f"\n"
            f"Page-load failure handling (J1):\n"
            f"- If the viewport is BLANK / pure white / only shows an error\n"
            f"  banner (e.g. 'Cannot GET /', 'ERR_CONNECTION_RESET', a stack\n"
            f"  trace, an empty body), output ONE synthetic action call:\n"
            f"  {{\"name\":\"computer\",\"input\":{{\"action\":\"reload\"}}}}\n"
            f"  then re-evaluate the new screenshot before sending VERDICT.\n"
            f"- If the page is STILL blank after that reload, set:\n"
            f"  * VERDICT=contradicted (presence / navigation / visual_semantics /\n"
            f"    interaction_accessibility) — a dead page is direct evidence\n"
            f"    that the claimed UI does NOT exist.\n"
            f"  * VERDICT=insufficient_evidence (computation / persistence /\n"
            f"    data_integration / runtime_health) — we cannot exercise the\n"
            f"    feature without a working page.\n"
            f"  AND include the literal phrase `page_load_failed` in REASONING\n"
            f"  so the orchestrator can correlate. The orchestrator will then\n"
            f"  run additional reload-retries on its own — you only need to\n"
            f"  attempt one reload yourself."
        )
        goal += _guideline_block(guideline)
        if ms_active:
            goal += multi_session_block(participants, marker)
        if veracity_active:
            goal += veracity_block()

        screenshots: list[str] = []
        action_trace: list[dict[str, Any]] = []
        screenshot_b64 = await self._env.screenshot_base64()
        await self._save_step_screenshot(query_id, claim_id, 0, screenshots)

        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": goal},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": screenshot_b64,
                }},
            ],
        }]

        tool_cfg = {
            "type": COMPUTER_TOOL_TYPE,
            "name": "computer",
            "display_width_px": w,
            "display_height_px": h,
            "display_number": 1,
        }
        # Hang the opt-in tools next to the native `computer` tool.
        tools_for_request: list[dict[str, Any]] = [tool_cfg]
        if ms_active:
            tools_for_request.append(anthropic_tool_schema(DELEGATE_TO_VIEWER_TOOL_NAME))
        if veracity_active:
            tools_for_request.append(anthropic_tool_schema(SEARCH_TOOL_NAME))

        client = self._make_anthropic_client()
        per_request_headers = self._request_extra_headers()
        final_text = ""

        # Diagnostics: counts of real vs synthetic tool_uses across the
        # whole verify_claim run. If `real_tool_use_steps` stays 0 across
        # multiple steps it almost certainly means the gateway is stripping
        # the `anthropic-beta` header and the computer tool is never
        # registered on the model side. We expose this in the returned
        # result so downstream reports / regressions can flag it.
        stats = {
            "steps_total": 0,
            "real_tool_use_steps": 0,
            "synthetic_tool_use_steps": 0,
            "text_only_steps": 0,
            "real_tool_use_count": 0,
            "synthetic_tool_use_count": 0,
            # Extended-thinking observability. Both stay 0 when
            # ``thinking`` is disabled (the default) so existing
            # downstream reports are unaffected. When enabled,
            # ``thinking_steps`` should equal ``steps_total`` and
            # ``thinking_blocks`` should be ≥ steps_total; if either
            # stays 0 across the whole run while thinking is on, the
            # gateway is silently dropping the thinking field.
            "thinking_steps": 0,
            "thinking_blocks": 0,
        }
        # Verbatim thinking blocks (text + signature), persisted into the
        # result JSON so trajectories can be replayed later.
        thinking_trace: list[dict[str, Any]] = []

        # ── Extended thinking knobs ──────────────────────────────────
        #
        # When ``cfg.thinking_budget_tokens > 0`` we ask Claude to emit
        # `thinking` blocks alongside its tool_use / text response. The
        # blocks live inside ``response.content`` and are echoed back
        # unchanged in the next request's assistant message (see the
        # ``messages.append({"role": "assistant", "content":
        # response.content})`` line below) — Anthropic mandates that
        # the assistant message starts with thinking/redacted_thinking
        # blocks BEFORE any tool_use blocks, and round-tripping
        # ``response.content`` as-is satisfies that ordering for free.
        #
        # Two important guardrails (per Anthropic's docs):
        #   1. ``thinking.budget_tokens`` MUST be strictly less than
        #      ``max_tokens``. We clamp to ``max_tokens - 256`` to keep
        #      a small headroom for the visible response/tool_use.
        #   2. ``temperature`` MUST be 1.0 when thinking is enabled.
        #      The SDK default is 1.0 and we never override it here, so
        #      no extra plumbing is required — just don't add a
        #      ``temperature=...`` kwarg below.
        thinking_kwargs: dict[str, Any] = {}
        if cfg.thinking_budget_tokens > 0:
            requested = int(cfg.thinking_budget_tokens)
            max_allowed = max(1024, cfg.max_output_tokens - 256)
            budget_tokens = min(requested, max_allowed)
            if budget_tokens != requested:
                logger.warning(
                    "thinking.budget_tokens=%d clamped to %d (max_output_tokens=%d)",
                    requested, budget_tokens, cfg.max_output_tokens,
                )
            thinking_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }

        for step in range(1, budget + 1):
            logger.info("NativeCUA step %d/%d claim=%s", step, budget, claim_id)
            try:
                response = client.beta.messages.create(
                    model=cfg.model,
                    max_tokens=cfg.max_output_tokens,
                    tools=tools_for_request,
                    messages=messages,
                    betas=[COMPUTER_BETA_FLAG],
                    extra_headers=per_request_headers,
                    **thinking_kwargs,
                )
            except Exception as e:
                if is_quota_error(e):
                    raise QuotaExhaustedError("cua", str(e)) from e
                logger.error("Anthropic native call failed: %s", e)
                return _error_result(
                    f"Native API error: {type(e).__name__}: {e}",
                    action_trace, screenshots,
                )

            usage_tracker.record_response("cua", response, model=cfg.model)

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]
            thinking_blocks = [
                b for b in response.content
                if b.type in ("thinking", "redacted_thinking")
            ]
            if text_blocks:
                final_text = "\n".join(b.text for b in text_blocks)

            stats["steps_total"] += 1
            if thinking_blocks:
                stats["thinking_steps"] += 1
                stats["thinking_blocks"] += len(thinking_blocks)
                thinking_trace.extend(
                    _thinking_blocks_to_trace(step, thinking_blocks)
                )
            if tool_uses:
                stats["real_tool_use_steps"] += 1
                stats["real_tool_use_count"] += len(tool_uses)
            # The "first real tool_use seen" event is the clearest signal
            # that the beta header made it through. Log it once, loudly,
            # so users running with a new gateway can verify the fix.
            if tool_uses and stats["real_tool_use_steps"] == 1:
                logger.info(
                    "NativeCUA: real computer tool_use observed (beta header is "
                    "honored by the gateway). step=%d claim=%s",
                    step, claim_id,
                )

            # Some gateways (notably the AWS-Bedrock-backed relays we use) do
            # not honour the `anthropic-beta: computer-use-*` flag, so Claude
            # never gets a real `computer` tool registered. The model still
            # emits the intended action, but as JSON-shaped *text* with
            # stop_reason="end_turn". Recover those into synthetic tool_uses
            # so the loop can still drive the browser; otherwise every
            # action_effect / navigation claim would be unverifiable.
            synthetic_tool_uses: list[dict[str, Any]] = []
            if not tool_uses and final_text and not _VERDICT_RE.search(final_text):
                parsed = _parse_text_as_actions(final_text)
                if parsed:
                    for idx, action_input in enumerate(parsed):
                        synthetic_tool_uses.append({
                            "id": f"synthetic_{step}_{idx}",
                            "input": action_input,
                            "_synthetic": True,
                        })

            if synthetic_tool_uses:
                stats["synthetic_tool_use_steps"] += 1
                stats["synthetic_tool_use_count"] += len(synthetic_tool_uses)
                # First time we have to salvage actions from text -> WARN
                # exactly once with full context so the gateway issue is
                # not buried under per-step INFO logs.
                if (
                    stats["synthetic_tool_use_steps"] == 1
                    and stats["real_tool_use_steps"] == 0
                ):
                    logger.warning(
                        "NativeCUA: model returned JSON actions as TEXT, not as "
                        "tool_use blocks. This means the gateway/model did NOT "
                        "register the `computer` tool — almost always because "
                        "`anthropic-beta: %s` was stripped. Falling back to "
                        "synthetic action salvage. claim=%s step=%d "
                        "actions_recovered=%d",
                        COMPUTER_BETA_FLAG, claim_id, step, len(synthetic_tool_uses),
                    )

            if not tool_uses and not synthetic_tool_uses:
                stats["text_only_steps"] += 1
                break

            tool_results_block: list[dict[str, Any]] = []
            # Unified iteration over real Anthropic tool_use objects and our
            # synthetic ones (which use dict access).
            dispatch_list: list[tuple[str, str, dict[str, Any], bool]] = []
            for tu in tool_uses:
                dispatch_list.append((tu.id, tu.name, tu.input or {}, False))
            for stu in synthetic_tool_uses:
                # Salvaged text actions are always `computer` actions.
                dispatch_list.append((stu["id"], "computer", stu["input"], True))

            for tu_id, tu_name, tu_input, is_synth in dispatch_list:
                # Opt-in non-computer tools (viewer subagent / veracity search)
                # are dispatched by name and reply with a TEXT tool_result.
                if tu_name in (DELEGATE_TO_VIEWER_TOOL_NAME, SEARCH_TOOL_NAME):
                    if tu_name == DELEGATE_TO_VIEWER_TOOL_NAME:
                        r = await self._delegate_to_viewer(tu_input)
                    else:
                        r = await self._run_search(tu_input)
                    success = bool(r.get("success"))
                    result_text = r.get("text") or (
                        "ok" if success else (r.get("error") or "failed")
                    )
                    action_trace.append({
                        "step": step, "tool": tu_name, "action": tu_name,
                        "input": tu_input, "success": success,
                        "error": r.get("error"), "synthetic": is_synth,
                    })
                    if not is_synth:
                        tool_results_block.append({
                            "type": "tool_result",
                            "tool_use_id": tu_id,
                            "content": [{"type": "text", "text": result_text}],
                            "is_error": not success,
                        })
                    continue

                tool_name, args = _translate_action(tu_input)
                # Some sentinel actions reply with text rather than a fresh
                # screenshot — see the per-branch assignments below. When
                # ``sentinel_text`` is set we send a ``text`` tool_result;
                # otherwise we fall through to the default screenshot path.
                sentinel_text: str | None = None
                if tool_name == "__screenshot__":
                    success = True
                    err = None
                elif tool_name == "__cursor__":
                    # Anthropic's ``cursor_position`` action expects the
                    # current cursor coordinate back, not a screenshot.
                    # Returning an image here was protocol-incorrect and
                    # left Claude with no positional answer (it would just
                    # see the same scene it already had). We now reply
                    # with the cached coordinate from PlaywrightEnv.
                    success = True
                    err = None
                    try:
                        pos = self._env.get_cursor_position()
                        sentinel_text = f"cursor at ({pos['x']},{pos['y']})"
                    except Exception as e:  # noqa: BLE001
                        sentinel_text = "cursor position unavailable"
                        logger.warning("get_cursor_position failed: %s", e)
                elif tool_name == "__unknown__":
                    # Previously we logged a warning on our side and still
                    # handed Claude a regular screenshot — Claude could not
                    # tell its requested action was unsupported and would
                    # cheerfully continue on a stale world model. We now
                    # echo back an explicit error string so the model has
                    # the information it needs to replan.
                    success = False
                    raw_action = args.get("action") or "(unknown)"
                    err = f"unsupported native action: {raw_action}"
                    logger.warning(err)
                    sentinel_text = (
                        f"ERROR: action '{raw_action}' is not supported by "
                        f"this executor. Supported actions: left_click, "
                        f"right_click, middle_click, double_click, "
                        f"triple_click, mouse_move, left_click_drag, "
                        f"left_mouse_down, left_mouse_up, scroll, type, "
                        f"key, hold_key, wait, zoom, screenshot, "
                        f"cursor_position, reload."
                    )
                else:
                    handler = getattr(self._executor, tool_name, None)
                    if handler is None:
                        success = False
                        err = f"no handler for {tool_name}"
                    else:
                        r = await handler(args)
                        success = bool(r.get("success"))
                        err = r.get("error")
                action_trace.append({
                    "step": step,
                    "tool": tool_name,
                    "action": tu_input.get("action", tool_name),
                    "input": tu_input,
                    "success": success,
                    "error": err,
                    "synthetic": is_synth,
                })

                # Only real tool_use ids can be referenced from the assistant
                # message; for synthetic actions there is no `tool_use_id`
                # Claude knows about, so we feed the fresh screenshot back via
                # a plain user-image turn after the dispatch loop instead.
                if not is_synth:
                    if sentinel_text is not None:
                        tool_results_block.append({
                            "type": "tool_result",
                            "tool_use_id": tu_id,
                            "content": [{
                                "type": "text",
                                "text": sentinel_text,
                            }],
                            "is_error": tool_name == "__unknown__",
                        })
                    else:
                        new_b64 = await self._env.screenshot_base64()
                        tool_results_block.append({
                            "type": "tool_result",
                            "tool_use_id": tu_id,
                            "content": [{
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": new_b64,
                                },
                            }],
                        })

            # Transient-feedback claims (弹幕/toast): capture ONE immediate
            # (non-settling) frame and show it to the model alongside the
            # tool_result screenshots so a fast-disappearing element is caught.
            imm_b64 = await capture_transient_immediate(
                self._env, cfg, query_id, claim_id, step, screenshots, claim_type,
            )
            if imm_b64 and tool_results_block:
                tool_results_block.append({
                    "type": "text",
                    "text": (
                        "Extra frame captured IMMEDIATELY after the action "
                        "(before the screen settled) to catch transient UI "
                        "like danmaku/toast:"
                    ),
                })
                tool_results_block.append({
                    "type": "image",
                    "source": {
                        "type": "base64", "media_type": "image/png", "data": imm_b64,
                    },
                })

            await self._save_step_screenshot(query_id, claim_id, step, screenshots)
            messages.append({"role": "assistant", "content": response.content})
            if tool_results_block:
                messages.append({"role": "user", "content": tool_results_block})
            if synthetic_tool_uses:
                # Synthetic path: feed a fresh screenshot + nudge back as a
                # plain user turn so Claude can see the effect of the actions
                # we executed on its behalf.
                fresh_b64 = await self._env.screenshot_base64()
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "I executed the action(s) you described "
                            f"({', '.join(s['input'].get('action', '?') for s in synthetic_tool_uses)}). "
                            "Here is the updated screenshot. Continue with the next action, "
                            "or send the final VERDICT/CONFIDENCE/EVIDENCE_SUFFICIENT/REASONING text."
                        )},
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/png", "data": fresh_b64,
                        }},
                    ],
                })

            # When Claude returned only text and we recovered no actions, the
            # turn is over. With real or synthetic tool_uses we keep looping.
            if response.stop_reason == "end_turn" and not synthetic_tool_uses and not tool_uses:
                break
        else:
            logger.warning("NativeCUA max iterations reached (%d)", budget)

        # Per-claim summary: makes it trivial to grep logs for
        # "synthetic_only" claims (= gateway didn't honor the beta flag).
        synth_ratio = (
            stats["synthetic_tool_use_steps"] / stats["steps_total"]
            if stats["steps_total"] else 0.0
        )
        if stats["steps_total"] and stats["real_tool_use_steps"] == 0 and stats["synthetic_tool_use_steps"] > 0:
            mode = "synthetic_only"
        elif stats["real_tool_use_steps"] > 0 and stats["synthetic_tool_use_steps"] == 0:
            mode = "native_only"
        elif stats["real_tool_use_steps"] > 0:
            mode = "mixed"
        else:
            mode = "text_only"
        logger.info(
            "NativeCUA claim=%s done: mode=%s steps=%d real=%d synth=%d "
            "synth_ratio=%.2f thinking_steps=%d thinking_blocks=%d",
            claim_id, mode, stats["steps_total"],
            stats["real_tool_use_steps"], stats["synthetic_tool_use_steps"],
            synth_ratio,
            stats["thinking_steps"], stats["thinking_blocks"],
        )
        if (
            cfg.thinking_budget_tokens > 0
            and stats["steps_total"] > 0
            and stats["thinking_steps"] == 0
        ):
            # Same diagnostic story as the synthetic_only branch above:
            # the request asked for extended thinking, but every turn
            # came back without any thinking block. That points at the
            # gateway silently dropping the `thinking` request field
            # (some Bedrock-backed relays do this) or at a model id that
            # does not actually support extended thinking.
            logger.warning(
                "NativeCUA claim=%s: extended thinking was requested "
                "(budget_tokens=%d) but the model returned ZERO thinking "
                "blocks across %d steps. The gateway likely dropped the "
                "`thinking` field, or the model id %r does not support it.",
                claim_id, cfg.thinking_budget_tokens, stats["steps_total"],
                cfg.model,
            )

        result = _compose_result(
            _parse_text_verdict(final_text),
            action_trace, screenshots, final_text,
        )
        result["native_tool_use_stats"] = {
            **stats,
            "mode": mode,
            "synthetic_ratio": round(synth_ratio, 3),
        }
        result["thinking_trace"] = thinking_trace
        return result

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

    def _make_anthropic_client(self):
        import anthropic
        kwargs: dict[str, Any] = {"api_key": self._cfg.api_key}
        if self._cfg.base_url:
            url = self._cfg.base_url.rstrip("/")
            if url.endswith("/v1"):
                url = url[:-3]
            kwargs["base_url"] = url

        # `default_headers` is attached to every outgoing request, including
        # retries. We always force `anthropic-beta: computer-use-2025-11-24`
        # here so the computer tool gets registered even if the SDK's own
        # `betas=[...]` path is stripped by a gateway middleware. User-
        # supplied headers in `extra_headers` win on conflict so they can
        # override e.g. when piloting a newer beta tag.
        merged_default_headers: dict[str, str] = {
            "anthropic-beta": COMPUTER_BETA_FLAG,
        }
        for k, v in (self._cfg.extra_headers or {}).items():
            merged_default_headers[str(k)] = str(v)
        kwargs["default_headers"] = merged_default_headers
        return anthropic.Anthropic(**kwargs)

    def _request_extra_headers(self) -> dict[str, str]:
        """Per-request headers used as a second injection point.

        Some gateways drop the SDK's automatically-generated
        `anthropic-beta` header (which is what causes Claude to never see
        the `computer` tool and forces this adapter into the synthetic
        text-action fallback). We therefore pass the same beta tag through
        the public `extra_headers` API of `beta.messages.create`, which
        the SDK merges on top of its own headers per request. Yaml-
        configured `extra_headers` are layered on top so users can
        whitelist additional gateway-specific tags without touching code.
        """
        out: dict[str, str] = {"anthropic-beta": COMPUTER_BETA_FLAG}
        for k, v in (self._cfg.extra_headers or {}).items():
            out[str(k)] = str(v)
        return out


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
