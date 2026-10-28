"""Local-app adaptations around DDB's evaluator; leave upstream scoring intact."""

from __future__ import annotations
import importlib
from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Any, NoReturn
from urllib.parse import urljoin, urlparse


class EvaluatorError(RuntimeError):
    """The evaluation service failed, so no benchmark reward can be assigned."""


class EvaluatorHealth:
    def __init__(self) -> None:
        self.failure: EvaluatorError | None = None

    def fail(self, component: str, reason: str) -> NoReturn:
        if self.failure is None:
            self.failure = EvaluatorError(f"{component}: {reason}")
        raise self.failure

    def check(self) -> None:
        if self.failure is not None:
            raise self.failure

    def check_result(self, component: str, result: dict[str, Any]) -> None:
        # These are upstream's explicit infrastructure signals, not ordinary
        # insufficient_evidence verdicts caused by an inconclusive UI check.
        reason = str(result.get("reason", ""))
        prefixes = (
            "[LLM error]",
            "[LLM API key not configured]",
            "[openai package not installed]",
            "Native API error:",
            "Anthropic FC API error:",
            "OpenAI Responses error:",
        )
        if result.get("evaluator_error") or reason.startswith(prefixes):
            self.fail(component, reason or "upstream evaluator_error")
        self.check()

    def guard(self, component: str, function, *, result: bool = False):
        @wraps(function)
        async def checked(*args, **kwargs):
            self.check()
            try:
                value = await function(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - SDK exception types differ by protocol.
                self.fail(component, f"{type(exc).__name__}: {exc}")
            if result:
                self.check_result(component, value)
            self.check()
            return value

        return checked


def local_url(raw: str, base: str) -> str:
    url = urljoin(base, raw)
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Participant entry URLs must point to the local application")
    return url


def collaboration_prompt(participants: list[str] | None, marker: str) -> str:
    roles = ", ".join(participants or ["creator", "viewer_a"])
    return (
        f"\n\n## Local-app collaboration: {roles}\n"
        "You are the creator. Establish your identity using the app's normal UI. "
        "No participant is pre-authenticated and no shared room is pre-joined. "
        "For shared-room checks, create/open the relevant room and discover its "
        "actual invite URL or join code through the UI. Do not invent one. "
        "Use delegate_to_viewer with an explicit role, an observed invite URL "
        "when needed, and instructions for signing up/logging in with a DISTINCT "
        "test identity, joining the room, and performing the action. You may use "
        "the supplied local entry URL as a starting point; it does not establish identity. "
        "For private-data checks, the viewer must log in as a different user and "
        "reach the corresponding app screen before checking visibility. A login "
        "screen, failed join, or inaccessible screen is inconclusive, NOT proof "
        "that private data is isolated. Do not tell the viewer the expected verdict. "
        "Each role keeps its own cookies and browser context for follow-up calls; "
        "omit url to continue that role's session. Viewer sessions remain open "
        "while you observe the creator's screen. Never impersonate both users "
        "in the creator session. "
        f"Use the unique marker {marker!r} in cross-session test data. "
        "Only report supported after observing the requested behavior across "
        "the established identities. Report setup failures explicitly.\n"
    )


DELEGATION_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {
            "type": "string",
            "description": "Declared participant, e.g. viewer_a. Reuse the role to continue its session.",
        },
        "url": {
            "type": "string",
            "description": "Optional local invite/entry URL observed in the app. Omit to continue the role's session or use its entry hint.",
        },
        "instruction": {
            "type": "string",
            "description": "The action and any necessary distinct-user login, signup, or join steps/code. Do not reveal the expected result.",
        },
    },
    "required": ["role", "instruction"],
    "additionalProperties": False,
}


def make_delegate(health: EvaluatorHealth):
    async def delegate(runner, args: dict[str, Any]) -> dict[str, Any]:
        health.check()
        role = str(args.get("role", "")).strip()
        instruction = str(args.get("instruction", "")).strip()
        roles = [p for p in runner._active_participants if p.lower() != "creator"] or [
            "viewer_a"
        ]
        if role not in roles or not instruction:
            return {
                "success": False,
                "error": "Supply a declared viewer role and instruction.",
            }
        # Count calls separately from identities: a viewer can log in, act,
        # and inspect a later update without becoming a new anonymous user.
        if runner._viewer_count >= max(4, runner._cfg.max_viewers_per_claim * 3):
            return {
                "success": False,
                "error": "Viewer call budget exhausted for this claim.",
            }
        sessions = getattr(runner, "_ddb_viewers", {})
        runner._ddb_viewers = sessions
        key = (runner._active_query_id, role)
        session = sessions.get(key)
        raw_url = args.get("url")
        if raw_url is not None and (
            not isinstance(raw_url, str) or not raw_url.strip()
        ):
            return {"success": False, "error": "url must be a non-empty local URL."}
        target = (
            raw_url or runner._active_session_urls.get(role) or runner._active_app_url
        )
        try:
            target = local_url(target, runner._active_app_url)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        if session is None:
            session = await runner._env.spawn_isolated_session(target)
            sessions[key] = session
        elif raw_url is not None:
            await session.goto(target)
        # The parent PlaywrightEnv owns and closes these contexts on stop().
        # Keeping them open is essential for presence counts and later updates.
        page_info = await session.get_page_info()
        target = page_info.get("page_url") or target
        viewer = runner._new_viewer_runner(session, runner._make_viewer_config())
        runner._viewer_count += 1
        try:
            result = await viewer.verify_claim(
                claim_id=f"{runner._active_claim_id}_{role}_{runner._viewer_count}",
                checkpoint=f"[{role}] {instruction}",
                verification=(
                    f"You are {role}, a distinct participant, not the creator. "
                    "Your browser session is isolated and may need signup/login "
                    "and room joining through normal UI. Reuse your own existing "
                    "login when present. Complete the requested setup and action. "
                    f"Use marker {runner._active_marker!r} in any test content. "
                    "Report what identity/room you reached and what you observed. "
                    "Return supported only if you reached the intended screen "
                    "and completed the action/check. If login, joining, or access "
                    "failed, return insufficient_evidence and explain the failed "
                    "step; absence of data on an inaccessible screen proves nothing."
                ),
                claim_type="state_transition",
                app_url=target,
                query_id=runner._active_query_id,
                guideline="",
            )
        except Exception as exc:  # noqa: BLE001 - Preserve viewer SDK failures across tool handling.
            health.fail("viewer CUA", f"{type(exc).__name__}: {exc}")
        health.check_result("viewer CUA", result)
        observation = (
            result.get("cua_reasoning")
            or result.get("reason")
            or "No observation reported"
        )
        completed = result.get("verdict") == "supported"
        summary = f"{role}: completed={completed}; {observation}"
        if not completed:
            summary += " Setup/action was inconclusive; do not infer privacy or collaboration success."
        return {
            "success": True,
            "error": None,
            "text": summary,
            "image_b64": None,
            "clamped": False,
            "observation": observation,
            "marker": runner._active_marker,
            "viewer_completed": completed,
        }

    return delegate


async def call_anthropic_judge(
    messages,
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking: bool = False,
):
    """Stream native responses so the configured 32K output budget is supported."""
    import anthropic
    import usage_tracker
    from cua.native_adapter import _thinking_blocks_to_trace
    from verification import llm_judge

    url = base_url.rstrip("/").removesuffix("/v1")
    client_kwargs = {"api_key": api_key}
    if url:
        client_kwargs["base_url"] = url
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": llm_judge._openai_to_anthropic_messages(messages),
    }
    if thinking:
        request["thinking"] = {"type": "adaptive", "display": "summarized"}
        request["output_config"] = {"effort": "high"}
    else:
        request["temperature"] = temperature
    async with anthropic.AsyncAnthropic(**client_kwargs) as client:
        try:
            async with client.messages.stream(**request) as stream:
                response = await stream.get_final_message()
        except Exception as exc:
            # Anthropic SDK 2.x removed this keyword from messages.stream.
            # Older SDKs still accept it, but the model/gateway may reject it.
            unsupported_keyword = isinstance(exc, TypeError) and (
                "unexpected keyword argument 'temperature'" in str(exc)
            )
            if "temperature" not in request or not (
                unsupported_keyword or llm_judge._is_temperature_unsupported_error(exc)
            ):
                raise
            request.pop("temperature")
            async with client.messages.stream(**request) as stream:
                response = await stream.get_final_message()
    usage_tracker.record_response("llm_judge", response, model=model)
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    thinking_blocks = [
        b for b in response.content if b.type in {"thinking", "redacted_thinking"}
    ]
    trace = _thinking_blocks_to_trace(1, thinking_blocks) if thinking_blocks else []
    return text, trace


@contextmanager
def adapt_evaluator(*, judge_provider: str = "openai"):
    """Install scoped hooks before constructing any upstream runner/registry."""
    if judge_provider not in {"openai", "anthropic"}:
        raise ValueError(f"Unsupported judge provider: {judge_provider}")

    from cua.cua_agent import CUAAgent
    from cua.multi_session import MultiSessionMixin
    from cua.tools import schemas
    from verification import llm_judge
    from verification.claim_verifier import ClaimVerifier

    health = EvaluatorHealth()
    with ExitStack() as stack:

        def replace(obj, name, value):
            original = getattr(obj, name)
            setattr(obj, name, value)
            stack.callback(setattr, obj, name, original)

        # The snapshot stores LLMJudgeConfig.provider but never passes it to
        # judge_claim. Select its existing native transport explicitly instead
        # of relying on gateway hostname heuristics or an image-schema error.
        if judge_provider == "anthropic":
            replace(llm_judge, "_prefer_anthropic_route", lambda base_url, model: True)
            replace(llm_judge, "_call_anthropic_judge", call_anthropic_judge)

        replace(
            llm_judge,
            "judge_claim",
            health.guard("LLM judge", llm_judge.judge_claim, result=True),
        )
        replace(
            CUAAgent,
            "verify_claim",
            health.guard("CUA", CUAAgent.verify_claim, result=True),
        )
        # Upstream catches per-claim exceptions. Remember failures independently
        # and prevent subsequent claims from spending tokens or concealing them.
        replace(
            ClaimVerifier,
            "verify",
            health.guard("claim evaluator", ClaimVerifier.verify),
        )
        replace(
            MultiSessionMixin,
            "_delegate_to_viewer",
            health.guard("viewer", make_delegate(health)),
        )
        for name in (
            "multi_session",
            "runner",
            "native_adapter",
            "anthropic_fc_adapter",
            "openai_fc_adapter",
            "openai_native_adapter",
        ):
            module = importlib.import_module(f"cua.{name}")
            replace(module, "multi_session_block", collaboration_prompt)
        original = schemas._TOOL_DESCRIPTIONS[schemas.DELEGATE_TO_VIEWER_TOOL_NAME]
        schemas._TOOL_DESCRIPTIONS[schemas.DELEGATE_TO_VIEWER_TOOL_NAME] = (
            (
                "Ask a distinct local-app participant to log in/join and act. "
                "Use an observed invite URL when needed. Sessions persist by role; "
                "omit url to continue. No identities or rooms are pre-provisioned."
            ),
            DELEGATION_SCHEMA,
            "action",
        )
        stack.callback(
            schemas._TOOL_DESCRIPTIONS.__setitem__,
            schemas.DELEGATE_TO_VIEWER_TOOL_NAME,
            original,
        )
        yield health
