"""
Anthropic non-beta function-calling adapter with thinking round-trip.

Sibling of ``cua.native_adapter.NativeAnthropicRunner``. Same tool surface
(single ``computer`` tool with an ``action`` discriminator that mirrors
Anthropic's official ``computer_20251124`` action vocabulary), same
``ActionExecutor`` backend, same ``VERDICT:`` text contract, same synthetic
salvage path. The only material differences from the beta adapter:

  * SDK call uses ``client.messages.create`` (stable, non-beta) instead of
    ``client.beta.messages.create``. No ``betas=[...]``, no
    ``anthropic-beta`` header.
  * The ``computer`` tool is registered as a vanilla function-calling tool
    with a hand-authored JSON schema (action enum + all per-action fields).
    The model still emits ``tool_use(name="computer", input={action: ...})``
    so the dispatch loop is byte-identical to native_adapter's.
  * Extended thinking uses ``{"type": "adaptive", "display": "summarized"}``
    + ``output_config={"effort": "high"}`` (the Claude 4.7-recommended
    shape that ``probe_anthropic_fc.py`` validated end-to-end on the
    real gateway). Round-trips ``response.content`` verbatim so the
    ``thinking`` blocks (with ``signature``) chain across turns and the
    CUA does NOT lose memory of its earlier reasoning the way the
    beta path can on gateways that strip ``signature``.

Use this protocol when:
  * The gateway/Bedrock relay 504s on ``/v1/messages?beta=true`` (the
    computer-use beta endpoint) but is fine on the stable
    ``/v1/messages`` endpoint.
  * You verified with ``probe_anthropic_fc.py`` that the gateway round-
    trips ``thinking.signature`` non-empty (otherwise extended thinking
    silently becomes a no-op even on this path).

What we deliberately *reuse* from native_adapter (single source of truth
for the action vocabulary; one PR adds an action everywhere):

  * ``_translate_action``      : action name + input dict -> executor call
  * ``_parse_text_as_actions`` : salvage JSON-shaped actions from text
  * ``_VERDICT_RE``            : prefix regex used to gate salvage
"""
from __future__ import annotations
import logging
from typing import Any
from .playwright_env import PlaywrightEnv
from .runner import (
    RunnerConfig,
    _compose_result,
    _compute_action_summary,
    _guideline_block,
    _parse_text_verdict,
)
# Single source of truth for the computer-tool action vocabulary +
# synthetic-text salvage. Both adapters share these so adding a new
# action is one PR, not two.
from .native_adapter import (
    _translate_action,
    _parse_text_as_actions,
    _thinking_blocks_to_trace,
    _VERDICT_RE,
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


# ── Tool definition ─────────────────────────────────────────────────
#
# Anthropic's official beta computer tool (``computer_20251124``) ships
# its action schema server-side; we don't see it. On the non-beta path
# we have to author the schema ourselves so Claude knows which fields
# go with which action.
#
# Design notes:
#   * Single ``computer`` tool with an ``action`` enum discriminator;
#     same shape as the beta tool from the model's perspective so
#     ``_translate_action`` keeps working unchanged.
#   * Enum lists every action name ``_translate_action`` handles
#     (including ``reload`` and ``cursor_position`` which Anthropic's
#     vocab doesn't formally include but the executor supports).
#   * Per-action fields are flattened into the top-level properties
#     dict. Claude is good at picking the right subset; constraining
#     this with JSON Schema ``oneOf`` per-action makes the schema
#     enormous and gives worse model behaviour in practice.
#   * ``additionalProperties: false`` keeps the model from inventing
#     fields that ``_translate_action`` would silently ignore.
_COMPUTER_FC_TOOL: dict[str, Any] = {
    "name": "computer",
    "description": (
        "Drive a web app in a real browser via screenshots + keyboard + mouse. "
        "Pick exactly one `action` per call and fill ONLY the fields that "
        "action needs:\n"
        "  - screenshot                                  : (no fields) take a fresh viewport screenshot\n"
        "  - left_click / right_click / middle_click /\n"
        "    double_click / triple_click                  : coordinate=[x,y]; optional text=<modifier like 'shift'>\n"
        "  - mouse_move                                  : coordinate=[x,y]\n"
        "  - left_click_drag                             : start_coordinate=[x,y], coordinate=[x,y]\n"
        "  - left_mouse_down / left_mouse_up             : coordinate=[x,y]\n"
        "  - scroll                                       : coordinate=[x,y], scroll_direction in {up,down,left,right}, scroll_amount=<int wheel ticks>\n"
        "  - type                                         : text=<string to type into focused element>\n"
        "  - key                                          : text=<single key name like 'Enter' or 'ctrl+a'>\n"
        "  - hold_key                                     : text=<key>, duration=<seconds, float> OR duration_ms=<int>\n"
        "  - wait                                         : duration=<seconds> OR duration_ms=<int>\n"
        "  - zoom                                         : region=[x1,y1,x2,y2] to crop+upscale a viewport patch\n"
        "  - reload                                      : (no fields) full page reload; use when viewport is blank\n"
        "  - cursor_position                              : (no fields) read current cursor coordinate\n"
        "Coordinates are integer viewport pixels in [0, width) x [0, height). "
        "Use one `action` per tool call; chain calls across turns."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "screenshot",
                    "left_click",
                    "right_click",
                    "double_click",
                    "triple_click",
                    "middle_click",
                    "mouse_move",
                    "left_click_drag",
                    "left_mouse_down",
                    "left_mouse_up",
                    "scroll",
                    "type",
                    "key",
                    "hold_key",
                    "wait",
                    "zoom",
                    "reload",
                    "cursor_position",
                ],
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Pixel [x, y] in viewport. Required by clicks/moves/scrolls.",
            },
            "start_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Drag origin [x, y]; pair with `coordinate` as the drag target.",
            },
            "text": {
                "type": "string",
                "description": (
                    "For `type`: literal text to type. "
                    "For `key`/`hold_key`: key name. "
                    "For click/scroll: modifier name (e.g. 'shift')."
                ),
            },
            "duration": {
                "type": "number",
                "description": "Duration in seconds for `wait` or `hold_key`.",
            },
            "duration_ms": {
                "type": "integer",
                "description": "Alternative to `duration`, in milliseconds.",
            },
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Wheel ticks for `scroll`.",
            },
            "region": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": "[x1, y1, x2, y2] viewport rect for `zoom`.",
            },
            "retries": {
                "type": "integer",
                "description": "Number of additional reload attempts for `reload`.",
                "minimum": 0,
                "maximum": 3,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


class AnthropicFCRunner(MultiSessionMixin):
    """Drop-in alternative to ``NativeAnthropicRunner`` that uses the
    stable (non-beta) ``client.messages.create`` API + a custom FC
    ``computer`` tool, while still round-tripping extended-thinking
    blocks (signature + reasoning text) across turns.

    Same ``verify_claim`` contract as the other runners so
    ``CUAAgent`` can dispatch to it transparently. Inherits viewer-subagent
    + veracity-search plumbing from :class:`cua.multi_session.MultiSessionMixin`.
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

        # Prompt is copied verbatim from native_adapter.NativeAnthropicRunner
        # so model behaviour stays comparable across protocols. The literal
        # `{"name":"computer","input":...}` template the model is told to
        # emit for synthetic-action fallback also matches our tool name.
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

        # Hang the opt-in tools next to the hand-authored `computer` FC tool.
        tools_for_request: list[dict[str, Any]] = [_COMPUTER_FC_TOOL]
        if ms_active:
            tools_for_request.append(anthropic_tool_schema(DELEGATE_TO_VIEWER_TOOL_NAME))
        if veracity_active:
            tools_for_request.append(anthropic_tool_schema(SEARCH_TOOL_NAME))

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

        client = self._make_anthropic_client()
        final_text = ""

        # Diagnostics — same shape as NativeAnthropicRunner so downstream
        # reports / log greps work for both protocols. ``synthetic_*`` and
        # ``mode`` stay meaningful here: on the non-beta path, the model
        # should always emit real tool_use, so any non-zero synthetic
        # count is a useful signal that something went wrong upstream
        # (e.g. the model thought it was on a Chat-Completions backend).
        stats = {
            "steps_total": 0,
            "real_tool_use_steps": 0,
            "synthetic_tool_use_steps": 0,
            "text_only_steps": 0,
            "real_tool_use_count": 0,
            "synthetic_tool_use_count": 0,
            "thinking_steps": 0,
            "thinking_blocks": 0,
        }
        # Verbatim thinking blocks (text + signature), persisted into the
        # result JSON so trajectories can be replayed later.
        thinking_trace: list[dict[str, Any]] = []

        # ── Extended thinking knobs (adaptive + summarized + effort) ──
        #
        # Differs from native_adapter, which uses
        # ``{"type": "enabled", "budget_tokens": N}`` (legacy 3.7-style).
        # For Claude 4.7+ the recommended shape is adaptive + summarized
        # (per Anthropic docs and verified end-to-end by
        # probe_anthropic_fc.py against multiple gateways).
        #
        # We keep the ``thinking_budget_tokens > 0`` config field as the
        # ON/OFF switch so existing eval_config_claude_cua.yaml works
        # unchanged. The actual numeric budget is set by the server in
        # adaptive mode and is NOT controlled by us; budget_tokens would
        # be rejected by the API in adaptive mode.
        thinking_kwargs: dict[str, Any] = {}
        if cfg.thinking_budget_tokens > 0:
            thinking_kwargs["thinking"] = {
                "type": "adaptive",
                "display": "summarized",
            }
            thinking_kwargs["output_config"] = {"effort": "high"}

        for step in range(1, budget + 1):
            logger.info("AnthropicFC step %d/%d claim=%s", step, budget, claim_id)
            try:
                response = client.messages.create(
                    model=cfg.model,
                    max_tokens=cfg.max_output_tokens,
                    tools=tools_for_request,
                    messages=messages,
                    **thinking_kwargs,
                )
            except Exception as e:
                if is_quota_error(e):
                    raise QuotaExhaustedError("cua", str(e)) from e
                logger.error("Anthropic fc call failed: %s", e)
                return _error_result(
                    f"Anthropic FC API error: {type(e).__name__}: {e}",
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
            if tool_uses and stats["real_tool_use_steps"] == 1:
                logger.info(
                    "AnthropicFC: first real computer tool_use observed. "
                    "step=%d claim=%s",
                    step, claim_id,
                )

            # Synthetic salvage: same logic as native_adapter. Less likely
            # to trigger on the non-beta path (the model normally emits
            # real tool_use even without the beta header because the tool
            # is registered via the stable FC schema), but kept for
            # defence-in-depth.
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
                if (
                    stats["synthetic_tool_use_steps"] == 1
                    and stats["real_tool_use_steps"] == 0
                ):
                    logger.warning(
                        "AnthropicFC: model emitted JSON actions as TEXT, not "
                        "as tool_use. On the non-beta FC path this is "
                        "unexpected — the `computer` tool is registered via "
                        "stable FC schema and the model should call it. "
                        "Possible causes: (1) gateway is downcasting the "
                        "tool schema in transit, (2) model id %r does not "
                        "actually understand the `computer` tool. Falling "
                        "back to synthetic action salvage. claim=%s step=%d "
                        "actions_recovered=%d",
                        cfg.model, claim_id, step, len(synthetic_tool_uses),
                    )

            if not tool_uses and not synthetic_tool_uses:
                stats["text_only_steps"] += 1
                break

            tool_results_block: list[dict[str, Any]] = []
            dispatch_list: list[tuple[str, str, dict[str, Any], bool]] = []
            for tu in tool_uses:
                dispatch_list.append((tu.id, tu.name, tu.input or {}, False))
            for stu in synthetic_tool_uses:
                # Salvaged text actions are always `computer` actions.
                dispatch_list.append((stu["id"], "computer", stu["input"], True))

            for tu_id, tu_name, tu_input, is_synth in dispatch_list:
                # Opt-in non-computer tools (viewer subagent / veracity search)
                # are dispatched by name and return a TEXT tool_result.
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
                sentinel_text: str | None = None
                if tool_name == "__screenshot__":
                    success = True
                    err = None
                elif tool_name == "__cursor__":
                    success = True
                    err = None
                    try:
                        pos = self._env.get_cursor_position()
                        sentinel_text = f"cursor at ({pos['x']},{pos['y']})"
                    except Exception as e:  # noqa: BLE001
                        sentinel_text = "cursor position unavailable"
                        logger.warning("get_cursor_position failed: %s", e)
                elif tool_name == "__unknown__":
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

            # ── thinking round-trip ─────────────────────────────────
            #
            # Append response.content VERBATIM. Two reasons:
            #   1. Anthropic mandates that an assistant message starts
            #      with thinking/redacted_thinking blocks before any
            #      tool_use blocks; round-tripping response.content
            #      preserves the ordering for free.
            #   2. Each thinking block carries a server-issued
            #      ``signature`` token. We MUST echo it back unchanged
            #      next turn or Anthropic rejects the request (or, on
            #      gateways that strip signature, silently drops the
            #      thinking continuity and the CUA loses memory of its
            #      own reasoning — which is the bug this whole adapter
            #      exists to avoid).
            messages.append({"role": "assistant", "content": response.content})
            if tool_results_block:
                messages.append({"role": "user", "content": tool_results_block})
            if synthetic_tool_uses:
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

            if response.stop_reason == "end_turn" and not synthetic_tool_uses and not tool_uses:
                break
        else:
            logger.warning("AnthropicFC max iterations reached (%d)", budget)

        # Per-claim summary log — same fields and labels as native_adapter
        # so `grep "AnthropicFC.*mode=" results/*.log` and
        # `grep "NativeCUA.*mode=" results/*.log` are symmetric.
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
            "AnthropicFC claim=%s done: mode=%s steps=%d real=%d synth=%d "
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
            # Same diagnostic as native_adapter: the request asked for
            # extended thinking but every turn came back empty. On the
            # non-beta path this almost always means the gateway is
            # stripping the `thinking` field from responses (yibu's
            # pre-fix behaviour) — re-run probe_anthropic_fc.py to
            # confirm signature_chars==0 and forward records.json to
            # the gateway operator.
            logger.warning(
                "AnthropicFC claim=%s: extended thinking was requested but "
                "the model returned ZERO thinking blocks across %d steps. "
                "The gateway likely strips `thinking` from responses, or "
                "the model id %r does not actually support extended "
                "thinking on this endpoint. Cross-check with "
                "probe_anthropic_fc.py.",
                claim_id, stats["steps_total"], cfg.model,
            )

        result = _compose_result(
            _parse_text_verdict(final_text),
            action_trace, screenshots, final_text,
        )
        # Field name kept as ``native_tool_use_stats`` for log/report
        # compatibility with NativeAnthropicRunner — reports already
        # know how to render this key.
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
            # Anthropic SDK appends /v1 itself; yaml configs commonly keep
            # the /v1 suffix for OpenAI-compat reuse, so strip it here.
            if url.endswith("/v1"):
                url = url[:-3]
            kwargs["base_url"] = url

        # Drop any `anthropic-beta` header the user may have left in
        # config for the beta path. On the non-beta FC path it is at
        # best a no-op and at worst confuses gateways that route by
        # header. Other extra headers (vendor routing tags, etc.) are
        # passed through unchanged.
        cleaned_headers: dict[str, str] = {}
        for k, v in (self._cfg.extra_headers or {}).items():
            if str(k).lower() == "anthropic-beta":
                logger.info(
                    "AnthropicFC: dropping configured extra_header %s=%r "
                    "(this adapter is non-beta).", k, v,
                )
                continue
            cleaned_headers[str(k)] = str(v)
        if cleaned_headers:
            kwargs["default_headers"] = cleaned_headers
        return anthropic.Anthropic(**kwargs)


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
