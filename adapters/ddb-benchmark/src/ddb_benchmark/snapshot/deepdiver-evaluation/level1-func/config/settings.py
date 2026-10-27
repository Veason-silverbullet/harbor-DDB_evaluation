"""
Central configuration: loads eval_config.yaml with environment variable overrides.
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)

# Allowed values for CUAConfig.agent_protocol.
#
# - ``function_calling``           : OpenAI-compatible Chat Completions
#   with our custom tool schema. Works for any LLM that supports
#   function/tool calling (GPT / Gemini / Claude-via-proxy).
# - ``native_computer_20251124``   : Anthropic native computer-use via
#   ``/v1/messages?beta=true`` with ``anthropic-beta:
#   computer-use-2025-11-24``. Claude-only.
# - ``native_computer_openai``     : OpenAI official CUA via Responses
#   API (``/v1/responses``) with the GA ``{"type": "computer"}`` tool.
#   GPT-5.4 / GPT-5.5 only (or any future GPT trained for the GA
#   computer tool). NOT compatible with Chat Completions gateways.
# - ``function_calling_anthropic`` : Anthropic SDK stable (non-beta)
#   ``/v1/messages`` + a hand-authored ``computer`` FC tool that mirrors
#   the beta computer tool's action vocabulary 1:1. Extended thinking is
#   sent as ``{"type":"adaptive","display":"summarized"}`` +
#   ``output_config={"effort":"high"}`` and the assistant's
#   ``response.content`` (including thinking blocks with signature) is
#   echoed back verbatim every turn so the CUA does NOT lose memory of
#   prior reasoning. Use this when the gateway 504s on the beta endpoint
#   or strips the beta header — provided you have first verified with
#   ``probe_anthropic_fc.py`` that the gateway round-trips
#   ``thinking.signature`` non-empty on the stable endpoint.
# - ``function_calling_openai``    : OpenAI Responses API
#   (``/v1/responses``) + our OWN custom function tools (NOT the GA
#   ``computer`` tool). Sidesteps relays that reject the GA computer
#   tool while still exposing GPT-5.5's encrypted reasoning. With
#   ``reasoning_passthrough: true`` it echoes reasoning items back
#   across turns (store=false + include=reasoning.encrypted_content).
#   Verify a relay supports it with ``probe_responses_minimal.py``.
AGENT_PROTOCOLS = (
    "function_calling",
    "native_computer_20251124",
    "native_computer_openai",
    "function_calling_anthropic",
    "function_calling_openai",
)
TOOL_MODES = ("multi_tool", "single_tool")

# Legacy provider name -> agent_protocol mapping (used when yaml only
# specifies the older `cua.provider` field).
#
# Note: ``provider=openai`` still maps to ``function_calling`` for
# backward compatibility. To opt into the new OpenAI native CUA path
# (Responses API + ``computer`` tool), set
# ``cua.agent_protocol: native_computer_openai`` explicitly.
_PROVIDER_TO_PROTOCOL: dict[str, str] = {
    "openai": "function_calling",
    "google": "function_calling",
    "gemini": "function_calling",
    "anthropic": "native_computer_20251124",
    "claude": "native_computer_20251124",
}


def _resolve_agent_protocol(cua_data: dict[str, Any]) -> str:
    """Resolve cua.agent_protocol with backward-compat for cua.provider.

    Priority: explicit ``agent_protocol`` > legacy ``provider`` > default.
    """
    if "agent_protocol" in cua_data:
        ap = str(cua_data["agent_protocol"])
        if ap in AGENT_PROTOCOLS:
            return ap
        # Silent fallback bit us once (cua.agent_protocol="native_computer_20250124"
        # got downgraded to function_calling because only the 20251124 routing
        # name is in AGENT_PROTOCOLS). Make this loud so anyone running the
        # Claude-native config notices a typo immediately instead of after
        # confusing eval results.
        logger.error(
            "cua.agent_protocol=%r is not in AGENT_PROTOCOLS=%r; "
            "FALLING BACK TO function_calling. If you meant to run the "
            "Claude native computer-use protocol, fix the config.",
            ap, AGENT_PROTOCOLS,
        )
        return "function_calling"

    if "provider" in cua_data:
        prov = str(cua_data["provider"]).lower()
        ap = _PROVIDER_TO_PROTOCOL.get(prov, "function_calling")
        logger.warning(
            "cua.provider=%r is deprecated; inferred agent_protocol=%r. "
            "Please switch to cua.agent_protocol in eval_config.yaml.",
            prov, ap,
        )
        return ap

    return "function_calling"

# Default iteration budgets per claim_type (empty -> fall back to max_iterations).
# Keyed on the two-axis verification methods (Axis 1). Legacy single-axis
# names are kept as aliases so older YAML configs / payloads keep tuning the
# right budget. Tightened in 2026-04: structural/perceptual need very few
# clicks; the stale-window early-stop usually triggers before these budgets
# anyway, so the savings are mostly on stuck loops where the model would
# otherwise burn the full budget producing the same wrong action repeatedly.
_DEFAULT_ITER_BUDGETS: dict[str, int] = {
    # canonical 7 verification methods
    "structural": 4,
    "navigation": 10,
    "state_transition": 12,
    "value_correctness": 10,
    "persistence": 12,
    "perceptual": 8,
    "runtime_integrity": 10,
    # legacy aliases (pre two-axis) — inert once claims carry method names,
    # but harmless and keep old configs meaningful.
    "presence": 4,
    "action_effect": 12,
    "computation": 10,
    "visual_semantics": 8,
}


@dataclass
class CUAConfig:
    # New-style fields.
    agent_protocol: str = "function_calling"   # or "native_computer_20251124"
    tool_mode: str = "multi_tool"              # or "single_tool"
    enable_readonly_tools: bool = False
    max_iterations_by_claim_type: dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_ITER_BUDGETS)
    )

    # Legacy / common fields.
    provider: str = ""   # kept for backward compat; mirrors agent_protocol if set
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str = ""
    max_iterations: int = 20
    screenshot_delay_ms: int = 500
    viewport: dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 900})
    headless: bool = True

    # Inject a controllable virtual-clock shim into every page so the CUA
    # can fast-forward long countdowns/timers via the `advance_time` tool
    # instead of waiting real wall-clock minutes. Inert until advanced, so
    # normal (non-timer) apps behave as on real time.
    enable_virtual_clock: bool = True

    # Extra HTTP headers to attach to every CUA-side LLM request. Primary
    # use-case is forcing `anthropic-beta: computer-use-2025-11-24` through
    # gateways that strip non-whitelisted headers (so the model actually
    # gets the `computer` tool registered instead of falling back to the
    # synthetic JSON-in-text salvage path). Other gateway-specific tags
    # (x-vendor, x-route, etc.) can also live here.
    extra_headers: dict[str, str] = field(default_factory=dict)

    # Per-request ``max_tokens`` ceiling used by the Anthropic native and
    # OpenAI native paths. Match this to the model's documented output
    # cap (Claude 4.7 Opus = 32k, GPT-5.5 = 128k) when you raise
    # thinking / reasoning budgets significantly; the default 4096 is
    # enough for typical 1-2 sentence VERDICT replies + a few tool_use
    # blocks. Anthropic requires ``thinking.budget_tokens <
    # max_tokens``, so bumping this is the right way to make room for
    # large thinking budgets. Ignored on the function_calling path
    # (which inherits the RunnerConfig default).
    max_output_tokens: int = 4096

    # Anthropic extended-thinking budget for the native computer-use path.
    # 0 disables extended thinking entirely (current behavior). Any positive
    # value enables it and is passed straight through as
    # ``thinking={"type": "enabled", "budget_tokens": <value>}`` on every
    # ``beta.messages.create`` request. Recommended starting points for
    # Claude 4.7 Opus computer-use:
    #   * 1024 - lightest, ~+1 reasoning step worth of headroom.
    #   * 4096 - balanced; matches Anthropic's quick-start CUA examples.
    #   * 8192 - heaviest; useful for multi-step action_effect /
    #            persistence chains where Claude tends to lose state.
    # Note: ``thinking.budget_tokens`` MUST be < ``max_output_tokens``;
    # the adapter will warn and clamp if you ask for too much. If you
    # want thinking_budget_tokens >= 4096, bump ``max_output_tokens``
    # accordingly (e.g. 8192 here pairs naturally with 4096 thinking
    # plus headroom for the visible reply).
    # Ignored on the function_calling path (Chat Completions has no
    # extended-thinking knob) and on the OpenAI native path (use
    # ``reasoning_effort`` instead).
    thinking_budget_tokens: int = 0

    # OpenAI reasoning controls for the native_computer_openai path. Only
    # consumed by the Responses-API adapter; the FC and Anthropic paths
    # ignore both fields.
    #   * reasoning_effort: "minimal" | "low" | "medium" | "high" - how
    #     much internal CoT the model is allowed to spend per turn. GPT-
    #     5.5 defaults to ``medium`` server-side; we mirror that here so
    #     users see the same behavior whether the field is set or not.
    #   * reasoning_summary: "auto" | "concise" | "detailed" | "" -
    #     controls whether the server returns a human-readable summary
    #     of each turn's reasoning in ``response.output[*]`` items of
    #     ``type=reasoning``. Empty string = no summaries (cheaper
    #     payload, but harder to debug). Default ``concise`` because it
    #     is cheap and dramatically improves trace explainability.
    # Both fields are passed straight through as
    # ``reasoning={"effort": ..., "summary": ...}``; unknown values are
    # forwarded so newer effort/summary tags work without a code change.
    reasoning_effort: str = "medium"
    reasoning_summary: str = "concise"

    # ── Multi-session collaboration (0702) ──────────────────────────
    # Master switch for the "main CUA + viewer subagent" protocol used to
    # test multi-person collaboration claims (realtime sync, shared read/
    # write, per-identity isolation). When False (default) the harness is
    # byte-for-byte the historical single-session behaviour: the
    # ``delegate_to_viewer`` tool is never registered and multi_session
    # claims run exactly like single-session ones. Only takes effect on the
    # ``function_calling`` (CUARunner) protocol.
    enable_multi_session: bool = False
    # Iteration budget for each viewer subagent loop (it performs ONE
    # delegated action then reports). Keep small to cap extra cost.
    viewer_max_iterations: int = 8
    # Model for the viewer subagent. Empty => reuse the main ``model``; point
    # at a cheaper model to reduce the cost of collaboration claims.
    viewer_model: str = ""
    # Cap on viewer subagents spawned per claim (default 2 = viewer_a/viewer_b).
    max_viewers_per_claim: int = 2

    # ── Transient-feedback capture (弹幕/toast) ──────────────────────
    # For state_transition claims, take ONE extra screenshot IMMEDIATELY
    # after the action (skipping the settle wait) so a fast-disappearing
    # element (danmaku/toast) is captured before it flies off screen. On by
    # default; the extra frame is added to the judge sampling pool and shown
    # to the CUA. Applies across all agent protocols.
    capture_transient_immediate: bool = True

    # ── Real-world data veracity check ──────────────────────────────
    # Master switch. When True, data-bearing claims (data_integration /
    # content / value_correctness) receive a ``search_web`` tool + a
    # ground-truth verification prompt so the CUA can confirm the app's
    # real-world data is actually correct/current (catches stale/mock data
    # such as a fixed "South Africa vs Mexico" fixture). Off by default.
    enable_veracity_check: bool = False
    # Pluggable search provider for ``search_web`` (parsed from ``cua.search``).
    # When no api_key/endpoint is configured the tool returns "unavailable"
    # and veracity degrades to insufficient_evidence — never a false fail.
    search_provider: str = "tavily"   # tavily | serper | custom
    search_api_key: str = ""
    search_endpoint: str = ""         # empty => provider default endpoint
    search_max_results: int = 5

    # OpenAI reasoning round-trip mode (native_computer_openai only).
    #
    # False (default): rely on server-side state via
    # ``previous_response_id``. The server keeps the model's reasoning
    # items between turns and we never re-upload them. Cheapest and
    # correct *as long as the endpoint actually persists responses* —
    # i.e. genuine ``api.openai.com`` or Azure OpenAI.
    #
    # True: run statelessly like the Anthropic native adapter. We set
    # ``store=False``, add ``include=["reasoning.encrypted_content"]``,
    # and echo each turn's ``reasoning`` items (with their opaque
    # ``encrypted_content`` blob — the GPT-5.5 analog of Anthropic's
    # ``redacted_thinking``) back into ``input`` ourselves. Use this
    # when the upstream is a relay that does NOT honor
    # ``previous_response_id`` (most third-party Responses proxies are
    # stateless, so server-side continuation silently loses context).
    # Trade-off: the reasoning blob + prior screenshots ride in the wire
    # payload, so per-turn upload cost grows with the step budget.
    # Ignored on every non-OpenAI protocol.
    reasoning_passthrough: bool = False


@dataclass
class CloudConfig:
    """DeepDiver Cloud generation API settings (used by run_e2e.py)."""
    api_url: str = "https://deepdiver.app/api/v1"
    # Cloud-side LLM model id, e.g. "claude-opus-4-6", "claude-sonnet-4-6",
    # "gemini-3-pro-preview", or "pangu_auto" (server default).
    model: str = "pangu_auto"


@dataclass
class ResilienceConfig:
    """Knobs for the page-load-failure -> reload-retry -> J1 cascade.

    The orchestrator wraps each claim with this protocol:

      1. Run CUA + judges; detect ``page_load_failed`` from runtime signals
         (visible_text length, requestfailed/4xx-5xx on the main document,
         blank screenshots, etc).
      2. If detected and we have budget left, ``env.reload()`` and retry
         the claim from scratch. ``claim_retry_on_page_failure`` controls
         how many reload-retries we attempt (0 disables the whole thing).
      3. If we still see a dead page after the last retry, fall through to
         the J1 branch: presence / visual_semantics / interaction_*
         claims are forced to ``contradicted`` (the page is observably
         broken, so the claim cannot be supported), while other claim
         types stay at ``insufficient_evidence`` (the page failed before
         we could exercise the feature). LLM judge is skipped in this
         branch to save quota.

    Tuning: 3 is the user-facing default — it matches roughly the
    timeout most flaky SPAs need on a fresh load while still failing
    fast on genuinely broken apps. Bump it when running against a slow
    sandbox; drop it to 0 to disable the entire mechanism.
    """

    claim_retry_on_page_failure: int = 3
    page_load_failure_visible_text_min: int = 20
    page_load_failure_main_resource_status_min: int = 400
    retry_backoff_ms: int = 800


@dataclass
class LLMJudgeConfig:
    provider: str = "openai"
    model: str = "claude-opus-4-6"
    api_key: str = ""
    base_url: str = "https://yibuapi.com/v1"
    temperature: float = 0.2
    max_tokens: int = 2048
    # Anthropic extended thinking for the judge (adaptive + summarized +
    # effort=high — the same validated shape the CUA FC adapter uses).
    # Only effective on the Anthropic-native transport; the OpenAI
    # chat.completions path has no thinking knob and ignores this. When
    # enabled, ``temperature`` is NOT sent (Anthropic rejects it alongside
    # thinking) and the thinking blocks (text + signature) are persisted
    # into each claim's ``judge_thinking_trace``.
    thinking: bool = False


@dataclass
class EvalConfig:
    eval_granularity: str = "atomic"
    seed_queries_path: str = ""
    # Asset roots for resolving relative fixture paths in seed JSONLs
    # (e.g. "S1-011/cat_a_avatar.png"). Relative entries are resolved
    # against the level1-func root. Searched in order, before the
    # built-in defaults in cua/playwright_env.py.
    fixtures_dirs: list[str] = field(default_factory=list)
    deepdiver_client_dir: str = ""
    deepdiver_server_url: str = "http://127.0.0.1:6274"
    mcp_server_url: str = "http://127.0.0.1:6274/mcp"
    cua: CUAConfig = field(default_factory=CUAConfig)
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    batch_size: int = 4
    max_concurrent_evals: int = 4
    output_dir: str = "./results"
    save_screenshots: bool = True
    save_traces: bool = True

    # Resolved absolute paths (populated by get_config)
    _root_dir: Path = field(default=Path("."), repr=False)


def _resolve_path(base: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _apply_env_overrides(cfg: EvalConfig) -> None:
    """Override config values from environment variables when set."""
    if v := os.environ.get("CUA_AGENT_PROTOCOL"):
        if v in AGENT_PROTOCOLS:
            cfg.cua.agent_protocol = v
    if v := os.environ.get("CUA_TOOL_MODE"):
        if v in TOOL_MODES:
            cfg.cua.tool_mode = v
    if v := os.environ.get("CUA_ENABLE_READONLY_TOOLS"):
        cfg.cua.enable_readonly_tools = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("CUA_ENABLE_VIRTUAL_CLOCK"):
        cfg.cua.enable_virtual_clock = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("CUA_PROVIDER"):
        cfg.cua.provider = v
        # Keep provider and agent_protocol in sync when provider is explicitly set
        # via env and protocol wasn't overridden.
        if not os.environ.get("CUA_AGENT_PROTOCOL"):
            cfg.cua.agent_protocol = _PROVIDER_TO_PROTOCOL.get(v.lower(), "function_calling")
    if v := os.environ.get("CUA_API_KEY"):
        cfg.cua.api_key = v
    elif v := os.environ.get("ANTHROPIC_API_KEY"):
        cfg.cua.api_key = v
    if v := os.environ.get("CUA_BASE_URL"):
        cfg.cua.base_url = v
    if v := os.environ.get("CUA_MODEL"):
        cfg.cua.model = v
    if v := os.environ.get("CUA_MAX_ITERATIONS"):
        cfg.cua.max_iterations = int(v)
    if v := os.environ.get("CUA_THINKING_BUDGET_TOKENS"):
        try:
            cfg.cua.thinking_budget_tokens = max(0, int(v))
        except ValueError:
            logger.warning("invalid CUA_THINKING_BUDGET_TOKENS=%r", v)
    if v := os.environ.get("CUA_REASONING_EFFORT"):
        cfg.cua.reasoning_effort = v
    if v := os.environ.get("CUA_REASONING_SUMMARY"):
        cfg.cua.reasoning_summary = v
    if v := os.environ.get("CUA_REASONING_PASSTHROUGH"):
        cfg.cua.reasoning_passthrough = v.strip().lower() in ("1", "true", "yes", "on")
    if v := os.environ.get("CUA_ENABLE_MULTI_SESSION"):
        cfg.cua.enable_multi_session = v.strip().lower() in ("1", "true", "yes", "on")
    if v := os.environ.get("CUA_VIEWER_MAX_ITERATIONS"):
        try:
            cfg.cua.viewer_max_iterations = max(1, int(v))
        except ValueError:
            logger.warning("invalid CUA_VIEWER_MAX_ITERATIONS=%r", v)
    if v := os.environ.get("CUA_VIEWER_MODEL"):
        cfg.cua.viewer_model = v
    if v := os.environ.get("CUA_MAX_VIEWERS_PER_CLAIM"):
        try:
            cfg.cua.max_viewers_per_claim = max(1, int(v))
        except ValueError:
            logger.warning("invalid CUA_MAX_VIEWERS_PER_CLAIM=%r", v)
    # CUA_EXTRA_HEADERS: JSON-encoded mapping merged on top of yaml values.
    # Example: CUA_EXTRA_HEADERS='{"anthropic-beta":"computer-use-2025-11-24"}'
    if v := os.environ.get("CUA_EXTRA_HEADERS"):
        try:
            import json as _json
            parsed = _json.loads(v)
            if isinstance(parsed, dict):
                merged = dict(cfg.cua.extra_headers)
                for hk, hv in parsed.items():
                    merged[str(hk)] = str(hv)
                cfg.cua.extra_headers = merged
            else:
                logger.warning(
                    "CUA_EXTRA_HEADERS must be a JSON object; got %r",
                    type(parsed).__name__,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to parse CUA_EXTRA_HEADERS=%r: %s", v, e)

    if v := os.environ.get("LLM_API_KEY"):
        cfg.llm_judge.api_key = v
    if v := os.environ.get("LLM_MODEL"):
        cfg.llm_judge.model = v
    if v := os.environ.get("LLM_BASE_URL"):
        cfg.llm_judge.base_url = v
    if v := os.environ.get("LLM_TEMPERATURE"):
        cfg.llm_judge.temperature = float(v)
    if v := os.environ.get("LLM_MAX_TOKENS"):
        cfg.llm_judge.max_tokens = int(v)

    if v := os.environ.get("CLOUD_MODEL"):
        cfg.cloud.model = v
    if v := os.environ.get("CLOUD_API_URL"):
        cfg.cloud.api_url = v
    elif v := os.environ.get("DEEPDIVER_API_URL"):
        cfg.cloud.api_url = v

    if v := os.environ.get("DEEPDIVER_SERVER_URL"):
        cfg.deepdiver_server_url = v
    if v := os.environ.get("MCP_SERVER_URL"):
        cfg.mcp_server_url = v
    if v := os.environ.get("EVAL_GRANULARITY"):
        cfg.eval_granularity = v
    if v := os.environ.get("EVAL_OUTPUT_DIR"):
        cfg.output_dir = v

    if v := os.environ.get("CLAIM_RETRY_ON_PAGE_FAILURE"):
        try:
            cfg.resilience.claim_retry_on_page_failure = max(0, int(v))
        except ValueError:
            logger.warning("invalid CLAIM_RETRY_ON_PAGE_FAILURE=%r", v)


def _load_yaml(yaml_path: Path) -> dict[str, Any]:
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_config(data: dict[str, Any], root_dir: Path) -> EvalConfig:
    cua_data = data.get("cua", {}) or {}

    agent_protocol = _resolve_agent_protocol(cua_data)
    tool_mode = str(cua_data.get("tool_mode", "multi_tool"))
    if tool_mode not in TOOL_MODES:
        logger.warning("cua.tool_mode=%r unknown; using multi_tool", tool_mode)
        tool_mode = "multi_tool"

    iter_budgets = dict(_DEFAULT_ITER_BUDGETS)
    raw_budgets = cua_data.get("max_iterations_by_claim_type") or {}
    if isinstance(raw_budgets, dict):
        for k, v in raw_budgets.items():
            try:
                iter_budgets[str(k)] = int(v)
            except (TypeError, ValueError):
                logger.warning("invalid max_iterations_by_claim_type[%s]=%r", k, v)

    raw_headers = cua_data.get("extra_headers") or {}
    extra_headers: dict[str, str] = {}
    if isinstance(raw_headers, dict):
        for k, v in raw_headers.items():
            try:
                extra_headers[str(k)] = str(v)
            except Exception:  # noqa: BLE001
                logger.warning("invalid cua.extra_headers[%s]=%r; skipping", k, v)
    else:
        logger.warning(
            "cua.extra_headers must be a mapping; got %r (ignored)",
            type(raw_headers).__name__,
        )

    # Pluggable web-search provider config for the veracity ``search_web`` tool.
    search_data = cua_data.get("search") or {}
    if not isinstance(search_data, dict):
        logger.warning(
            "cua.search must be a mapping; got %r (ignored)",
            type(search_data).__name__,
        )
        search_data = {}

    cua = CUAConfig(
        agent_protocol=agent_protocol,
        tool_mode=tool_mode,
        enable_readonly_tools=bool(cua_data.get("enable_readonly_tools", False)),
        max_iterations_by_claim_type=iter_budgets,
        provider=str(cua_data.get("provider", "")),
        model=cua_data.get("model", "claude-sonnet-4-20250514"),
        api_key=cua_data.get("api_key", ""),
        base_url=cua_data.get("base_url", ""),
        max_iterations=int(cua_data.get("max_iterations", 20)),
        screenshot_delay_ms=int(cua_data.get("screenshot_delay_ms", 500)),
        viewport=cua_data.get("viewport", {"width": 1280, "height": 900}),
        headless=bool(cua_data.get("headless", True)),
        enable_virtual_clock=bool(cua_data.get("enable_virtual_clock", True)),
        extra_headers=extra_headers,
        max_output_tokens=int(cua_data.get("max_output_tokens", 4096) or 4096),
        thinking_budget_tokens=int(cua_data.get("thinking_budget_tokens", 0) or 0),
        reasoning_effort=str(cua_data.get("reasoning_effort", "medium")),
        reasoning_summary=str(cua_data.get("reasoning_summary", "concise")),
        reasoning_passthrough=bool(cua_data.get("reasoning_passthrough", False)),
        enable_multi_session=bool(cua_data.get("enable_multi_session", False)),
        viewer_max_iterations=int(cua_data.get("viewer_max_iterations", 8) or 8),
        viewer_model=str(cua_data.get("viewer_model", "")),
        max_viewers_per_claim=int(cua_data.get("max_viewers_per_claim", 2) or 2),
        capture_transient_immediate=bool(
            cua_data.get("capture_transient_immediate", True)
        ),
        enable_veracity_check=bool(cua_data.get("enable_veracity_check", False)),
        search_provider=str(search_data.get("provider", "tavily") or "tavily"),
        search_api_key=str(search_data.get("api_key", "") or ""),
        search_endpoint=str(search_data.get("endpoint", "") or ""),
        search_max_results=int(search_data.get("max_results", 5) or 5),
    )

    llm_data = data.get("llm_judge", {}) or {}
    llm_judge = LLMJudgeConfig(
        provider=llm_data.get("provider", "openai"),
        model=llm_data.get("model", "claude-opus-4-6"),
        api_key=llm_data.get("api_key", ""),
        base_url=llm_data.get("base_url", "https://az.gptplus5.com/v1"),
        temperature=float(llm_data.get("temperature", 0.2)),
        max_tokens=int(llm_data.get("max_tokens", 2048)),
        thinking=bool(llm_data.get("thinking", False)),
    )

    cloud_data = data.get("cloud", {}) or {}
    cloud = CloudConfig(
        api_url=str(cloud_data.get("api_url", "https://deepdiver.app/api/v1")),
        model=str(cloud_data.get("model", "pangu_auto")),
    )

    res_data = data.get("resilience", {}) or {}
    resilience = ResilienceConfig(
        claim_retry_on_page_failure=int(
            res_data.get("claim_retry_on_page_failure", 3)
        ),
        page_load_failure_visible_text_min=int(
            res_data.get("page_load_failure_visible_text_min", 20)
        ),
        page_load_failure_main_resource_status_min=int(
            res_data.get("page_load_failure_main_resource_status_min", 400)
        ),
        retry_backoff_ms=int(res_data.get("retry_backoff_ms", 800)),
    )

    fixtures_dirs_raw = data.get("fixtures_dirs") or []
    if not isinstance(fixtures_dirs_raw, list):
        fixtures_dirs_raw = [str(fixtures_dirs_raw)]

    cfg = EvalConfig(
        eval_granularity=data.get("eval_granularity", "atomic"),
        seed_queries_path=data.get("seed_queries_path", ""),
        fixtures_dirs=[str(d) for d in fixtures_dirs_raw if str(d).strip()],
        deepdiver_client_dir=data.get("deepdiver_client_dir", ""),
        deepdiver_server_url=data.get("deepdiver_server_url", "http://127.0.0.1:6276"),
        mcp_server_url=data.get("mcp_server_url", "http://127.0.0.1:6276/mcp"),
        cua=cua,
        llm_judge=llm_judge,
        cloud=cloud,
        resilience=resilience,
        batch_size=int(data.get("batch_size", 4)),
        max_concurrent_evals=int(data.get("max_concurrent_evals", 4)),
        output_dir=data.get("output_dir", "./results"),
        save_screenshots=bool(data.get("save_screenshots", True)),
        save_traces=bool(data.get("save_traces", True)),
        _root_dir=root_dir,
    )

    _apply_env_overrides(cfg)
    return cfg


_cached_config: EvalConfig | None = None


def get_config(config_path: str | Path | None = None, force_reload: bool = False) -> EvalConfig:
    """
    Load and return the evaluation config (cached after first call).

    Resolution order: YAML file -> environment variable overrides.
    Default config path: level1-func/eval_config.yaml
    """
    global _cached_config
    if _cached_config is not None and not force_reload:
        return _cached_config

    root_dir = Path(__file__).resolve().parent.parent
    if config_path is None:
        config_path = root_dir / "eval_config.yaml"
    else:
        config_path = Path(config_path)

    data = _load_yaml(config_path)
    _cached_config = _build_config(data, root_dir)
    return _cached_config


def resolve_seed_path(cfg: EvalConfig) -> Path:
    """Resolve the seed queries JSONL path relative to the level1-func root."""
    return _resolve_path(cfg._root_dir, cfg.seed_queries_path)


def resolve_fixtures_dirs(cfg: EvalConfig) -> list[Path]:
    """Resolve the configured fixture asset roots (may be empty)."""
    return [_resolve_path(cfg._root_dir, d) for d in cfg.fixtures_dirs]


def resolve_output_dir(cfg: EvalConfig) -> Path:
    """Resolve the output directory path."""
    return _resolve_path(cfg._root_dir, cfg.output_dir)


def resolve_client_dir(cfg: EvalConfig) -> Path:
    """Resolve the deepdiver-client directory path."""
    return _resolve_path(cfg._root_dir, cfg.deepdiver_client_dir)
