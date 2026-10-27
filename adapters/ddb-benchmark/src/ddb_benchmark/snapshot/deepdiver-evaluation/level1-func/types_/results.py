"""
Result data structures for claim-level, query-level, and dataset-level metrics.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ClaimResult:
    """Full result for a single evaluated claim."""

    query_id: str
    claim_id: str
    source_fc_id: str
    checkpoint: str
    verification: str
    claim_type: str  # Axis 1: verification method (routing key)
    criticality: str
    verdict: str
    confidence: float
    reason: str
    # Axis 2: capability label (reporting). Optional for backward
    # compatibility with results saved before the two-axis taxonomy.
    semantic_type: str | None = None
    evidence_sufficient: bool | None = None
    used_validators: list[str] = field(default_factory=list)
    action_trace: list[dict] = field(default_factory=list)
    # Aggregated counters over ``action_trace`` (per-tool counts + failed
    # / clamped / synthetic / unknown counts + total_steps). Always
    # populated by ``cua.runner._compose_result`` so downstream metrics
    # / meta-review tooling can ground reasoning on the actual
    # interaction profile without re-iterating the raw trace.
    action_summary: dict = field(default_factory=dict)
    dom_signals: dict = field(default_factory=dict)
    screenshots: list[str] = field(default_factory=list)
    cua_reasoning: str = ""

    # Per-claim CUA native tool-use stats. Populated only when the
    # Anthropic native runner is in play. Carries the synthetic vs real
    # tool_use breakdown (mode / synthetic_ratio / step counts) so the
    # aggregator can report dataset-level "synthetic salvage" rates
    # without re-parsing action_trace.
    native_tool_use_stats: dict = field(default_factory=dict)

    # Verbatim extended-thinking blocks from the Anthropic CUA loop, in
    # chronological order. Each entry is
    #   {"step": int, "type": "thinking", "thinking": str, "signature": str}
    # or, for encrypted blocks,
    #   {"step": int, "type": "redacted_thinking", "data": str}.
    # ``signature`` is the server-issued token that MUST be echoed back
    # unchanged to continue a conversation, so persisting it (plus the
    # thinking text) is what makes a trajectory replayable/restorable.
    # Empty on non-Anthropic protocols and on runs saved before this field
    # existed.
    thinking_trace: list[dict] = field(default_factory=list)

    # LLM judge extended-thinking blocks (same shape as ``thinking_trace``:
    # text + server-issued ``signature``). Populated only when
    # ``llm_judge.thinking`` is enabled AND the judge ran on the
    # Anthropic-native transport; empty otherwise (incl. synthetic verdicts
    # from the J1 cascade / veracity gate and runs saved before this field).
    judge_thinking_trace: list[dict] = field(default_factory=list)

    # Wall-clock timing of this claim's FULL verification pipeline (CUA
    # loop + programmatic validators + LLM judge + page-load retries).
    # Shape: {"started_at": iso, "ended_at": iso, "duration_s": float}.
    # Combined with the per-step ``ts`` inside ``thinking_trace``, this is
    # what lets serial-vs-parallel latency be decomposed per claim.
    # Empty dict on runs saved before this field existed.
    timing: dict = field(default_factory=dict)

    # G2 + J1 cascade diagnostics. ``triggered`` indicates whether the
    # page_load_failure path fired for this claim, ``attempts`` is the
    # number of CUA+judge runs (1 = no retry, N = N-1 reload retries),
    # and ``history`` is the per-attempt reason list. Always present so
    # downstream tooling never KeyErrors, even when the cascade is
    # disabled (claim_retry_on_page_failure=0).
    page_load_failure: dict = field(default_factory=dict)

    # Per-claim runtime audit (harness-observed network truth). Built from
    # the Playwright ``runtime_signals`` snapshot in
    # ``ClaimVerifier._build_result`` and persisted here so it survives
    # ``dataclasses.asdict`` into ``<qid>.json``. Shape:
    #   {
    #     "backend_calls": [ {method,url,host,status,ok,pg_code,
    #                         pg_message,body_excerpt,ts_ms}, ... ],
    #     "backend_error_count": int,    # backend calls with status>=400
    #     "backend_ok_count": int,       # backend calls with status<400
    #     "pg_codes": ["42501", ...],    # distinct PostgreSQL/PostgREST codes
    #     "console_error_count": int,
    #     "failed_request_count": int,   # network-level failures (no response)
    #   }
    # Empty dict on older runs / when no backend traffic was observed, so
    # downstream tooling must treat absence as "no signal", not "healthy".
    runtime_audit: dict = field(default_factory=dict)

    # Empty-state observability marker (does NOT change the verdict; it is a
    # diagnostic flag for the next iteration). Set when a non-``supported``
    # verdict on a data-display claim (presence / computation / content)
    # coincides with empty-state placeholder language in the CUA reasoning
    # ("暂无", "还没有", "no records yet", ...). Lets us see, after a real
    # run, exactly which claims got stuck because the page had no user data
    # yet — so we can decide which ones deserve a create-then-verify setup.
    # Shape: {"suspected": bool, "markers": [str, ...], "verdict": str}.
    # Empty dict ({}) means "not suspected / not applicable".
    empty_state_signal: dict = field(default_factory=dict)

    # Attribution for non-``supported`` verdicts: WHY did this claim not
    # pass? The headline distinction we care about for cross-CUA comparison
    # is "the CUA never performed the required interaction" (a CUA-capability
    # problem — a weak agent can't drive create-then-verify flows) vs "the
    # CUA did act and the app genuinely failed / stayed inconclusive" (an
    # App problem). Diagnostic only — does NOT change the verdict or score.
    # Shape: {"category": str, "real_action_steps": int, "note": str}.
    # Categories:
    #   ""                          -> supported (no attribution)
    #   "cua_no_action"             -> insufficient + 0 real interaction steps
    #                                  (likely the CUA never tried — CUA-side)
    #   "inconclusive_after_action" -> insufficient despite real interaction
    #                                  (CUA tried; evidence still unclear)
    #   "contradicted_without_action" -> contradicted with 0 real steps
    #                                  (suspicious false-negative; CUA-side)
    #   "contradicted_after_action" -> contradicted after real interaction
    #                                  (more likely a genuine App defect)
    # Empty dict ({}) on supported claims / older runs.
    verdict_attribution: dict = field(default_factory=dict)


@dataclass
class QueryMetrics:
    """Aggregated metrics for a single query (v2 binary-gate rubric)."""

    weighted_func_score: float
    scorable_claim_count: int
    total_claim_count: int
    critical_claim_pass_rate: float | None
    insufficient_evidence_count: int
    insufficient_evidence_rate: float
    interaction_closure_rate: float | None = None
    correctness_rate: float | None = None
    persistence_rate: float | None = None
    contradiction_rate: float = 0.0
    claim_type_counts: dict[str, int] = field(default_factory=dict)
    # Axis-2 capability breakdown (counts per semantic_type).
    semantic_type_counts: dict[str, int] = field(default_factory=dict)
    verdict_counts: dict[str, int] = field(default_factory=dict)

    # ── v2 scoring diagnostics ───────────────────────────────────────
    # ``weighted_func_score`` above is the *post-gate* headline number
    # used in reports and ranking. The two fields below expose the
    # decomposition so reviewers can tell whether a low score came from
    # the binary verdicts themselves or from the critical-gate
    # multiplier kicking in.
    #
    # * ``weighted_func_score_pre_gate``: Σ(w·score)/Σ(w) over every
    #   non-N/A claim, before the critical gate is applied.
    # * ``critical_gate_factor``: ``min(1, crit_pass / 0.8)``. Always
    #   1.0 when the query has no critical claims (gate disabled).
    #
    # Defaults preserve backward compatibility with old reports that
    # were saved before v2 — ``weighted_func_score_pre_gate=None`` lets
    # the renderer fall back to the headline number when the
    # decomposition isn't available.
    weighted_func_score_pre_gate: float | None = None
    critical_gate_factor: float = 1.0

    # Native CUA synthetic-salvage rate. Computed as (claims whose CUA loop
    # had to fall back to text-action salvage at least once) /
    # (claims that ran the native runner at all). 0.0 when nobody used
    # native, 1.0 when every native claim was synthetic-only.
    native_synthetic_claim_rate: float = 0.0
    native_synthetic_only_claims: int = 0
    native_mixed_claims: int = 0
    native_native_only_claims: int = 0
    native_text_only_claims: int = 0

    # G2 + J1 resilience metrics.
    page_load_failure_claims: int = 0
    page_load_retry_total: int = 0
    page_load_cascade_triggered_claims: int = 0

    # Attribution breakdown for non-supported verdicts (see
    # ``ClaimResult.verdict_attribution``). Lets the report tell "the CUA
    # never drove the app" (cua_no_action / contradicted_without_action)
    # apart from "the app genuinely failed" (contradicted_after_action) when
    # comparing CUAs. Diagnostic only — does NOT affect the score.
    # Shape: {category: count}.
    verdict_attribution_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Complete evaluation result for one seed query."""

    query_id: str
    title: str
    category: str
    difficulty: int
    claim_results: list[ClaimResult]
    func_score: float
    metrics: QueryMetrics | None = None


@dataclass
class DatasetMetrics:
    """Aggregated metrics across all queries in a dataset."""

    total_queries: int
    total_claims: int
    overall_func_score: float
    grade: str
    mean_critical_claim_pass_rate: float | None
    mean_insufficient_evidence_rate: float
    mean_contradiction_rate: float
    score_by_difficulty: dict[str, float] = field(default_factory=dict)
    score_by_category: dict[str, float] = field(default_factory=dict)
    # Axis 1: score grouped by verification method (claim_type).
    score_by_claim_type: dict[str, float] = field(default_factory=dict)
    # Axis 2: capability profile — score grouped by semantic_type.
    score_by_semantic_type: dict[str, float] = field(default_factory=dict)
    # Per-claim_type validator reliability: for each verification method,
    # how often the programmatic validator decided on its own (coverage)
    # vs deferred to the LLM judge. Shape per method:
    #   {"count", "programmatic_decided", "llm_used", "programmatic_coverage"}.
    validator_reliability: dict[str, dict] = field(default_factory=dict)
    verdict_distribution: dict[str, int] = field(default_factory=dict)

    # Cross-query CUA synthetic-salvage health. ``mean_native_synthetic_claim_rate``
    # is the average synthetic_claim_rate across queries that had at least
    # one native claim; ``native_mode_distribution`` is a per-claim count of
    # the four mode buckets (native_only / synthetic_only / mixed /
    # text_only) so a single report cell can answer "did the gateway honor
    # the beta header today?".
    mean_native_synthetic_claim_rate: float = 0.0
    native_mode_distribution: dict[str, int] = field(default_factory=dict)

    # Cross-query G2/J1 resilience distribution.
    page_load_failure_claims: int = 0
    page_load_retry_total: int = 0
    page_load_cascade_triggered_claims: int = 0

    # ── v2 dataset-level scoring diagnostics ─────────────────────────
    # Average of per-query diagnostics. ``mean_weighted_func_score_pre_gate``
    # is the dataset's headline before any critical-gate dampening;
    # ``mean_critical_gate_factor`` shows how aggressively the gate is
    # firing across the run (1.0 = never; <1 = at least one query lost
    # points to the gate).
    mean_weighted_func_score_pre_gate: float | None = None
    mean_critical_gate_factor: float = 1.0
