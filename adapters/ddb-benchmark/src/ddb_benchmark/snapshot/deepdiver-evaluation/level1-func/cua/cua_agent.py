"""
CUAAgent: thin facade kept for backward compatibility.

Historical signatures accepted a ``provider`` string plus a handful of kwargs.
The actual loop now lives in one of four runners, selected by
``agent_protocol``:

- ``cua.runner.CUARunner``                          ``function_calling``
- ``cua.native_adapter.NativeAnthropicRunner``      ``native_computer_20251124``
- ``cua.openai_native_adapter.NativeOpenAIRunner``  ``native_computer_openai``
- ``cua.anthropic_fc_adapter.AnthropicFCRunner``    ``function_calling_anthropic``

All four share the same :class:`cua.tools.executor.ActionExecutor`
backend so action semantics and screenshot artifacts stay identical
across protocols.

The ``provider`` kwarg is mapped to ``agent_protocol`` via the same logic as
:mod:`config.settings`. New code should construct runners directly.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
from .playwright_env import PlaywrightEnv
from .runner import CUARunner, RunnerConfig
from .tools.executor import ActionExecutor
from .tools.inspector import PageInspector

logger = logging.getLogger(__name__)


_PROVIDER_TO_PROTOCOL = {
    "openai": "function_calling",
    "anthropic": "native_computer_20251124",
}


class CUAAgent:
    """Thin facade that dispatches to the appropriate runner based on config."""

    def __init__(
        self,
        env: PlaywrightEnv,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "",
        agent_protocol: str | None = None,
        tool_mode: str = "multi_tool",
        enable_readonly_tools: bool = False,
        base_url: str = "",
        max_iterations: int = 20,
        max_iterations_by_claim_type: dict[str, int] | None = None,
        screenshot_delay_ms: int = 500,
        save_screenshots: bool = True,
        # Run root that holds per-query subdirs. The runner writes
        # ``<screenshot_dir>/<query_id>/screenshots/<claim_id>_stepN.png``.
        screenshot_dir: str | Path = "./results",
        extra_headers: dict[str, str] | None = None,
        max_output_tokens: int = 4096,
        thinking_budget_tokens: int = 0,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "concise",
        reasoning_passthrough: bool = False,
        enable_multi_session: bool = False,
        viewer_max_iterations: int = 8,
        viewer_model: str = "",
        max_viewers_per_claim: int = 2,
        capture_transient_immediate: bool = True,
        enable_veracity_check: bool = False,
        search_provider: str = "tavily",
        search_api_key: str = "",
        search_endpoint: str = "",
        search_max_results: int = 5,
    ) -> None:
        # Resolve protocol with legacy mapping.
        if agent_protocol is None:
            if provider:
                agent_protocol = _PROVIDER_TO_PROTOCOL.get(
                    provider.lower(), "function_calling",
                )
                logger.warning(
                    "CUAAgent(provider=%r) is deprecated; "
                    "pass agent_protocol=%r directly.",
                    provider, agent_protocol,
                )
            else:
                agent_protocol = "function_calling"

        self._env = env
        self._cfg = RunnerConfig(
            agent_protocol=agent_protocol,
            tool_mode=tool_mode,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_iterations=max_iterations,
            max_iterations_by_claim_type=dict(max_iterations_by_claim_type or {}),
            post_action_delay_ms=screenshot_delay_ms,
            save_screenshots=save_screenshots,
            screenshot_dir=Path(screenshot_dir),
            enable_readonly_tools=enable_readonly_tools,
            extra_headers=dict(extra_headers or {}),
            max_output_tokens=int(max_output_tokens or 4096),
            thinking_budget_tokens=int(thinking_budget_tokens or 0),
            reasoning_effort=str(reasoning_effort or ""),
            reasoning_summary=str(reasoning_summary or ""),
            reasoning_passthrough=bool(reasoning_passthrough),
            enable_multi_session=bool(enable_multi_session),
            viewer_max_iterations=int(viewer_max_iterations or 8),
            viewer_model=str(viewer_model or ""),
            max_viewers_per_claim=int(max_viewers_per_claim or 2),
            capture_transient_immediate=bool(capture_transient_immediate),
            enable_veracity_check=bool(enable_veracity_check),
            search_provider=str(search_provider or "tavily"),
            search_api_key=str(search_api_key or ""),
            search_endpoint=str(search_endpoint or ""),
            search_max_results=int(search_max_results or 5),
        )

        self._executor = ActionExecutor(env, post_action_delay_ms=screenshot_delay_ms)
        self._inspector = PageInspector(env)

        if agent_protocol == "native_computer_20251124":
            # Lazy import to avoid forcing anthropic dependency for FC users.
            from .native_adapter import NativeAnthropicRunner
            self._runner: Any = NativeAnthropicRunner(
                env=env, config=self._cfg, executor=self._executor,
            )
        elif agent_protocol == "native_computer_openai":
            # Lazy import so users who never enable the OpenAI native
            # path don't need a Responses-API-capable openai SDK
            # version installed (older 1.x SDKs still work for the FC
            # path even if they lack ``client.responses``).
            from .openai_native_adapter import NativeOpenAIRunner
            self._runner = NativeOpenAIRunner(
                env=env, config=self._cfg, executor=self._executor,
            )
        elif agent_protocol == "function_calling_openai":
            # Responses API + our custom function tools (not the GA
            # computer tool). Reuses the same ActionExecutor / registry /
            # inspector as the Chat-Completions FC path, but talks
            # /v1/responses so it can round-trip GPT-5.5 encrypted
            # reasoning. Lazy import to avoid requiring a Responses-capable
            # openai SDK for users who never select this protocol.
            from .openai_fc_adapter import OpenAIFCRunner
            self._runner = OpenAIFCRunner(
                env=env, config=self._cfg,
                executor=self._executor, inspector=self._inspector,
            )
        elif agent_protocol == "function_calling_anthropic":
            # Lazy import: same anthropic SDK dependency as the beta
            # native path, but a separate adapter so the two can evolve
            # independently. This protocol skips the computer-use beta
            # endpoint entirely and uses ``client.messages.create`` with
            # a hand-authored FC ``computer`` tool; extended thinking
            # is sent as adaptive + summarized so signature round-trip
            # keeps the CUA's memory across turns even on gateways
            # that 504 on /v1/messages?beta=true.
            from .anthropic_fc_adapter import AnthropicFCRunner
            self._runner = AnthropicFCRunner(
                env=env, config=self._cfg, executor=self._executor,
            )
        else:
            self._runner = CUARunner(
                env=env, config=self._cfg,
                executor=self._executor, inspector=self._inspector,
            )

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
        kwargs: dict[str, Any] = dict(
            claim_id=claim_id,
            checkpoint=checkpoint,
            verification=verification,
            claim_type=claim_type,
            app_url=app_url,
            query_id=query_id,
            guideline=guideline,
        )
        # All five runners now accept the multi-session + veracity args, but
        # we still probe the signature so this facade tolerates any future
        # runner that keeps the original narrower signature (it degrades to
        # single-session / no-veracity transparently).
        try:
            import inspect
            params = inspect.signature(self._runner.verify_claim).parameters
            if "multi_session" in params:
                kwargs["multi_session"] = bool(multi_session)
                kwargs["participants"] = list(participants or [])
                kwargs["marker"] = marker or ""
            if "session_urls" in params:
                kwargs["session_urls"] = dict(session_urls or {})
            if "veracity_check" in params:
                kwargs["veracity_check"] = bool(veracity_check)
        except (TypeError, ValueError):
            pass
        return await self._runner.verify_claim(**kwargs)
