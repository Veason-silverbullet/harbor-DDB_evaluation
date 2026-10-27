"""
OpenAI function-calling adapter on the **Responses API** with reasoning
round-trip (agent_protocol = ``function_calling_openai``).

Why this exists
===============
Three pre-existing OpenAI-ish paths each miss something for GPT-5.5:

* ``function_calling`` (Chat Completions) — works through any relay, but
  Chat Completions does NOT expose GPT-5.5's reasoning/CoT, so it can
  never be round-tripped. The model re-thinks from scratch every turn.
* ``native_computer_openai`` (Responses + GA ``computer`` tool) — exposes
  encrypted reasoning, but the GA ``computer`` tool is rejected by most
  third-party ``/v1/responses`` relays (e.g. az.gptplus5 returns
  ``Unknown parameter: tools[0].display_height``).

This adapter is the middle path ("path C"): it drives the **Responses
API** with **our own custom function tools** (the same click / type /
scroll / report_verdict set as the Chat-Completions FC path), so it
sidesteps the GA computer-tool rejection, while STILL getting GPT-5.5's
encrypted reasoning and echoing it back across turns.

``probe_responses_minimal.py`` confirmed a strict relay (az.gptplus5):
DOWNLOAD ok (1k-char ``encrypted_content``), TOOL ok (custom function
tools accepted), ROUND-TRIP ok — **provided we sanitize echoed output
items** (strip output-only keys like ``metadata`` / ``status`` and null
values, which strict input validators reject). That sanitation lives in
:func:`_sanitize_echo`.

Reasoning continuity (``cfg.reasoning_passthrough``)
====================================================
* ``True`` (recommended for relays) — stateless: ``store=False`` +
  ``include=["reasoning.encrypted_content"]``; we accumulate the whole
  conversation and echo each turn's reasoning + function_call items
  (sanitized) back into ``input`` ourselves.
* ``False`` — server-side state via ``previous_response_id`` (only works
  on endpoints that actually persist responses).

Action vocabulary, screenshot cadence, verdict parsing and result shape
are intentionally identical to :class:`cua.runner.CUARunner` so
downstream reports render this protocol with zero special-casing.
"""
from __future__ import annotations
import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any
from .playwright_env import PlaywrightEnv
from .runner import (
    RunnerConfig,
    _GOAL_TEMPLATE,
    _compose_result,
    _format_tool_result_text,
    _guideline_block,
    _parse_text_verdict,
    _sha1,
    _verdict_from_payload,
)
from .openai_native_adapter import (
    _coerce_action_dict,
    _extract_message_text,
    _reasoning_items_to_trace,
)
from .multi_session import MultiSessionMixin, multi_session_block, veracity_block
from .transient import capture_transient_immediate
from .tools.executor import ActionExecutor
from .tools.inspector import PageInspector
from .tools.schemas import (
    REPORT_VERDICT_TOOL_NAME,
    DELEGATE_TO_VIEWER_TOOL_NAME,
    SEARCH_TOOL_NAME,
    MULTI_SESSION_TOOL_NAMES,
    VERACITY_TOOL_NAMES,
    ToolRegistry,
    build_default_registry,
)
from types_.errors import QuotaExhaustedError, is_quota_error
import usage_tracker

logger = logging.getLogger(__name__)


# Output-only fields the (strict) Responses input validator rejects when
# we echo output items back as input. Official api.openai.com ignores
# them; strict relays 400 with "Unknown parameter: input[N].<key>".
# Extend this set if a relay complains about another key.
_ECHO_DENY_KEYS = frozenset({"metadata", "status"})


def _sanitize_echo(obj: Any) -> Any:
    """Recursively drop output-only keys and null values so an echoed
    output item is accepted as an INPUT item by a strict validator."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _ECHO_DENY_KEYS or v is None:
                continue
            out[k] = _sanitize_echo(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_echo(x) for x in obj]
    return obj


def _tools_responses_format(
    registry: ToolRegistry, include_readonly: bool,
) -> list[dict[str, Any]]:
    """Convert the registry into the **Responses API** function-tool shape.

    Responses uses a FLAT function tool (``{type, name, description,
    parameters, strict}``), unlike Chat Completions which nests under a
    ``function`` key. ``strict`` is forced False: our schemas use
    optional properties with defaults, which strict mode forbids.
    """
    out: list[dict[str, Any]] = []
    for t in registry.visible(include_readonly=include_readonly):
        out.append({
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
            "strict": False,
        })
    return out


def _trim_input_images(conversation: list[dict[str, Any]], keep_last: int) -> None:
    """Drop base64 image bytes from older user turns in-place (passthrough
    mode only). Mirrors ``cua.runner._trim_screenshots_in_history`` but for
    the Responses input shape (``input_image`` blocks in user messages).
    """
    if keep_last < 0:
        keep_last = 0
    image_msg_indices: list[int] = []
    for idx, item in enumerate(conversation):
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "input_image" for p in content
        ):
            image_msg_indices.append(idx)
    if len(image_msg_indices) <= keep_last:
        return
    for idx in image_msg_indices[: len(image_msg_indices) - keep_last]:
        item = conversation[idx]
        new_content: list[dict[str, Any]] = []
        for part in item["content"]:
            if isinstance(part, dict) and part.get("type") == "input_image":
                new_content.append({
                    "type": "input_text",
                    "text": "[screenshot omitted to save tokens]",
                })
            else:
                new_content.append(part)
        item["content"] = new_content


def _error_result(
    msg: str,
    action_trace: list[dict],
    screenshots: list[str],
    *,
    evaluator_error: bool = False,
) -> dict[str, Any]:
    # ``evaluator_error`` (P0-2) flags failures of the EVALUATION harness
    # itself (Responses/gateway down, timeout) rather than a real app defect,
    # so downstream scoring can exclude them instead of charging the app an
    # ``insufficient_evidence``.
    from .runner import _compute_action_summary
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


class OpenAIFCRunner(MultiSessionMixin):
    """Responses-API + custom-function-tools runner with reasoning round-trip.

    Drop-in alternative to :class:`cua.runner.CUARunner` for GPT-5.5 when
    you want encrypted-reasoning continuity but the endpoint rejects the
    GA ``computer`` tool. Shares ``ActionExecutor`` / ``PageInspector`` /
    ``ToolRegistry`` with the Chat-Completions FC path.

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
        self._ms_reset()
        self._bind_handlers()

    def _bind_handlers(self) -> None:
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
            # Present only when their feature flag registered the tool.
            DELEGATE_TO_VIEWER_TOOL_NAME: self._delegate_to_viewer,
            SEARCH_TOOL_NAME: self._run_search,
        }
        for name, fn in action_map.items():
            tool = self._registry.get(name)
            if tool is not None:
                tool.handler = fn

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
            checkpoint=checkpoint, verification=verification,
            claim_type=claim_type, app_url=app_url, w=w, h=h,
        ) + _guideline_block(guideline)
        if ms_active:
            goal += multi_session_block(participants, marker)
        if veracity_active:
            goal += veracity_block()

        screenshots: list[str] = []
        action_trace: list[dict[str, Any]] = []
        screenshot_b64 = await self._capture_step_screenshot(
            query_id, claim_id, 0, screenshots,
        )

        include_readonly = cfg.enable_readonly_tools
        tools = _tools_responses_format(self._registry, include_readonly)
        # Per-claim visibility: hide opt-in tools the claim didn't request.
        hidden_tools: set[str] = set()
        if not ms_active:
            hidden_tools |= set(MULTI_SESSION_TOOL_NAMES)
        if not veracity_active:
            hidden_tools |= set(VERACITY_TOOL_NAMES)
        if hidden_tools:
            tools = [t for t in tools if t.get("name") not in hidden_tools]

        passthrough = bool(getattr(cfg, "reasoning_passthrough", False))
        include = ["reasoning.encrypted_content"] if passthrough else None
        store = False if passthrough else None
        reasoning_kwargs: dict[str, str] = {}
        if cfg.reasoning_effort:
            reasoning_kwargs["effort"] = cfg.reasoning_effort
        if cfg.reasoning_summary:
            reasoning_kwargs["summary"] = cfg.reasoning_summary

        client = self._make_openai_client()

        stats: dict[str, Any] = {
            "steps_total": 0,
            "function_call_steps": 0,
            "function_call_count": 0,
            "action_count": 0,
            "reasoning_item_count": 0,
            "encrypted_reasoning_count": 0,
            "reasoning_carried_back": 0,
            "reasoning_passthrough": passthrough,
        }
        # Verbatim reasoning items (summary + encrypted_content), persisted
        # into the result JSON so trajectories can be replayed later.
        thinking_trace: list[dict[str, Any]] = []

        # Initial user turn (goal + first screenshot). In passthrough mode
        # ``conversation`` accumulates the whole dialogue; in server-state
        # mode ``next_input`` only carries the latest items.
        initial_user: dict[str, Any] = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": goal},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                    "detail": "original",
                },
            ],
        }
        conversation: list[dict[str, Any]] = [initial_user]
        next_input: list[dict[str, Any]] = [initial_user]
        previous_response_id: str | None = None
        raw_assistant_text = ""
        last_hash = _sha1(screenshot_b64)
        stale_steps = 0

        for step in range(1, budget + 1):
            logger.info(
                "OpenAIFC step %d/%d claim=%s [%s] mode=%s",
                step, budget, claim_id, claim_type,
                "passthrough" if passthrough else "server_state",
            )
            if passthrough:
                _trim_input_images(conversation, cfg.keep_last_screenshots)
                inp = conversation
                prev = None
            else:
                inp = next_input
                prev = previous_response_id

            try:
                response = await self._responses_create(
                    client=client, tools=tools, reasoning_kwargs=reasoning_kwargs,
                    previous_response_id=prev, inp=inp, store=store, include=include,
                )
            except QuotaExhaustedError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("OpenAIFC Responses call failed: %s", e)
                return _error_result(
                    f"OpenAI Responses error: {type(e).__name__}: {e}",
                    action_trace, screenshots,
                    evaluator_error=True,
                )

            previous_response_id = getattr(response, "id", None) or previous_response_id
            output_items = list(getattr(response, "output", []) or [])
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

            # Echo this turn's model output back (sanitized) so the
            # reasoning + function_call chain survives under store=False.
            if passthrough:
                # Once a decrypt failure has degraded this run, stop echoing
                # reasoning items back — they would only fail decryption
                # again next turn. Function calls / messages still echoed.
                degraded = getattr(self, "_reasoning_degraded", False)
                for it in output_items:
                    d = _coerce_action_dict(it)
                    if degraded and d.get("type") == "reasoning":
                        continue
                    conversation.append(_sanitize_echo(d))
                if not degraded:
                    stats["reasoning_carried_back"] += len(reasoning_items)

            txt = _extract_message_text(messages)
            if txt:
                raw_assistant_text = txt

            # No tool call this turn -> treat any text as a verdict.
            if not function_calls:
                logger.info("OpenAIFC step %d: text-only response, parsing verdict", step)
                result = _compose_result(
                    _parse_text_verdict(raw_assistant_text or ""),
                    action_trace, screenshots, raw_assistant_text,
                )
                result["thinking_trace"] = thinking_trace
                return result

            stats["function_call_steps"] += 1
            stats["function_call_count"] += len(function_calls)

            fc_outputs: list[dict[str, Any]] = []
            verdict_payload: dict[str, Any] | None = None
            hit_verdict = False

            for fc in function_calls:
                d = _coerce_action_dict(fc)
                name = str(d.get("name") or "")
                call_id = d.get("call_id")
                try:
                    args = json.loads(d.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except (json.JSONDecodeError, TypeError):
                    args = {}

                if name == REPORT_VERDICT_TOOL_NAME:
                    verdict_payload = args
                    hit_verdict = True
                    action_trace.append({
                        "step": step, "tool": name, "action": name,
                        "input": args, "success": True, "error": None,
                    })
                    fc_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": "verdict recorded",
                    })
                    logger.info(
                        "OpenAIFC step=%d verdict=%s confidence=%.2f",
                        step, args.get("verdict"),
                        float(args.get("confidence", 0.0) or 0.0),
                    )
                    continue

                stats["action_count"] += 1
                tool_result = await self._call_tool(name, args)
                action_trace.append({
                    "step": step, "tool": name, "action": name,
                    "input": args, "success": bool(tool_result.get("success")),
                    "error": tool_result.get("error"),
                    "clamped": bool(tool_result.get("clamped")),
                })
                logger.info(
                    "OpenAIFC step=%d tool=%s success=%s error=%s",
                    step, name, bool(tool_result.get("success")),
                    tool_result.get("error"),
                )
                fc_outputs.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _format_tool_result_text(name, args, tool_result),
                })

            if hit_verdict:
                result = _compose_result(
                    _verdict_from_payload(verdict_payload or {}),
                    action_trace, screenshots,
                    raw_assistant_text or json.dumps(
                        verdict_payload or {}, ensure_ascii=False),
                )
                result["thinking_trace"] = thinking_trace
                return result

            # Fresh screenshot fed back as a new user turn so the model
            # can see the effect of its actions. For transient-feedback
            # claims (弹幕/toast) grab ONE immediate (non-settling) frame
            # first so a fast-disappearing element is not missed.
            imm_b64 = await capture_transient_immediate(
                self._env, cfg, query_id, claim_id, step, screenshots, claim_type,
            )
            new_b64 = await self._capture_step_screenshot(
                query_id, claim_id, step, screenshots,
            )
            new_hash = _sha1(new_b64)
            stale_steps = stale_steps + 1 if new_hash == last_hash else 0
            last_hash = new_hash

            # P0-1: hard early-stop on a stuck loop (type-2 timeout) instead
            # of spinning to max_iterations and forfeiting the claim.
            if stale_steps >= cfg.stale_hard_stop:
                logger.warning(
                    "OpenAIFC stale hard-stop: %d unchanged frames for %s (budget %d)",
                    stale_steps, claim_id, budget,
                )
                result = _compose_result(
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
                result["thinking_trace"] = thinking_trace
                return result

            follow_up = (
                f"Viewport {w}x{h}. Inspect the new screenshot and decide the "
                f"next tool call. (iteration {step}/{budget})"
            )
            if stale_steps >= cfg.stale_window:
                follow_up += (
                    f"\nWARNING: the screenshot has not changed for {stale_steps} "
                    f"consecutive steps. Try a different strategy, or call "
                    f"report_verdict with insufficient_evidence."
                )
            if budget - step <= 1:
                follow_up += (
                    f"\nIMPORTANT: only {budget - step} iteration(s) left. You MUST "
                    f"call `report_verdict` on your NEXT turn — no other tool. If "
                    f"uncertain, return verdict=\"insufficient_evidence\", "
                    f"evidence_sufficient=false, confidence<=0.4, and put what you "
                    f"actually saw in `reasoning`."
                )

            user_content: list[dict[str, Any]] = [
                {"type": "input_text", "text": follow_up},
            ]
            if imm_b64:
                user_content.append({
                    "type": "input_text",
                    "text": (
                        "Extra frame captured IMMEDIATELY after the action "
                        "(before the screen settled) to catch transient UI "
                        "like danmaku/toast:"
                    ),
                })
                user_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{imm_b64}",
                    "detail": "original",
                })
            user_content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{new_b64}",
                "detail": "original",
            })
            user_msg = {"role": "user", "content": user_content}
            if passthrough:
                conversation.extend(fc_outputs)
                conversation.append(user_msg)
            else:
                next_input = [*fc_outputs, user_msg]
        else:
            logger.warning("OpenAIFC max iterations (%d) reached for %s", budget, claim_id)
            result = _error_result(
                f"OpenAIFC reached max iterations ({budget})",
                action_trace, screenshots,
            )
            result["thinking_trace"] = thinking_trace
            return result

        # Unreachable (loop returns on verdict or via else), but keep a
        # safe fallback for type-checkers.
        return _error_result("OpenAIFC ended unexpectedly", action_trace, screenshots)

    # ── helpers ─────────────────────────────────────────────────

    async def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self._registry.get(name)
        if tool is None or tool.handler is None:
            return {"success": False, "error": f"unknown tool: {name}",
                    "text": None, "image_b64": None, "clamped": False}
        last: dict[str, Any] = {}
        for attempt in range(self._cfg.tool_fail_retries + 1):
            try:
                last = await tool.handler(args)
                if last.get("success"):
                    return last
            except Exception as e:  # noqa: BLE001
                last = {"success": False, "error": f"{type(e).__name__}: {e}",
                        "text": None, "image_b64": None, "clamped": False}
            logger.warning(
                "tool %s failed (attempt %d/%d): %s",
                name, attempt + 1, self._cfg.tool_fail_retries + 1, last.get("error"),
            )
        return last

    async def _capture_step_screenshot(
        self, query_id: str, claim_id: str, step: int, screenshots: list[str],
    ) -> str:
        # P0-3: post-action frames (step > 0) wait for the viewport to settle
        # so we never judge a mid-animation / mid-render frame. Step 0 is the
        # initial load, already warmed up by the orchestrator's post-goto wait.
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
                path = (
                    self._cfg.screenshot_dir / query_id / "screenshots"
                    / f"{claim_id}_step{step}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                screenshots.append(str(path))
            except Exception as e:  # noqa: BLE001
                logger.warning("failed to persist screenshot step=%d: %s", step, e)
        return base64.b64encode(raw).decode("utf-8")

    def _make_openai_client(self):
        from openai import OpenAI
        kwargs: dict[str, Any] = {
            "api_key": self._cfg.api_key,
            "base_url": self._cfg.base_url or None,
            "max_retries": 0,
            "timeout": 90.0,
        }
        if self._cfg.extra_headers:
            kwargs["default_headers"] = dict(self._cfg.extra_headers)
        return OpenAI(**kwargs)

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
        """Call ``client.responses.create`` with fast-fail on permanent
        schema errors (mirrors ``NativeOpenAIRunner._responses_create``)."""
        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        attempts_left = self._cfg.request_retries + 1
        # Extra retries reserved ONLY for the decrypt-degradation path
        # below; they do not consume the normal transient-retry budget.
        degrade_retries_left = 2
        while attempts_left > 0:
            attempts_left -= 1
            kwargs: dict[str, Any] = {
                "model": self._cfg.model,
                "tools": tools,
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
                    None, lambda: client.responses.create(**kwargs),
                )
                usage_tracker.record_response("cua", response, model=self._cfg.model)
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

                permanent_markers = (
                    "unsupported parameter", "unsupported tool", "unknown parameter",
                    "invalid_request_error", "invalid parameter", "model_not_found",
                    "no such tool", "does not support", "upstream request failed",
                    "upstream_error", "bad gateway",
                )
                if code == 400 or any(m in msg for m in permanent_markers):
                    logger.error(
                        "responses.create permanent error; aborting retries: %s", e,
                    )
                    raise

                # P0-2: exponential backoff before the next transient retry.
                if attempts_left > 0:
                    base = max(0, self._cfg.request_backoff_base_ms)
                    if base > 0:
                        cap = max(base, self._cfg.request_backoff_cap_ms)
                        used = max(0, self._cfg.request_retries - attempts_left)
                        delay_ms = min(base * (2 ** used), cap)
                        await asyncio.sleep(delay_ms / 1000.0)
        assert last_exc is not None
        raise last_exc


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
    un-decryptable blobs on subsequent turns. Function calls / outputs are
    kept so the tool-call chain stays consistent; only cross-turn reasoning
    continuity is dropped.
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


__all__ = ["OpenAIFCRunner", "Path"]
