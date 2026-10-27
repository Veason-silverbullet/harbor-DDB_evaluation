"""
Aggregation functions for query-level and dataset-level metrics.

v2 (binary + insufficient-in-denom + critical gate) rubric:

* Score is binary per claim — only ``supported`` earns 1; everything
  except ``not_applicable`` is 0 and *counts* toward the denominator.
  ``insufficient_evidence`` therefore now subtracts from the score
  instead of being silently excluded as it was under v1.5.
* The headline ``weighted_func_score`` is the v1.5 weighted ratio
  *multiplied* by ``min(1, critical_pass_rate / 0.8)``. Apps that lose
  more than 20% of their critical claims see their headline scaled
  proportionally; apps with no critical claims (or 100% critical pass)
  are unaffected.
* The pre-gate weighted ratio and the gate factor are surfaced as
  separate fields (``weighted_func_score_pre_gate`` and
  ``critical_gate_factor``) so reports can show the decomposition.
"""
from __future__ import annotations
from collections import defaultdict
from types_.results import ClaimResult, QueryMetrics, QueryResult, DatasetMetrics
from .scorer import (
    verdict_score,
    claim_weight,
    is_scorable,
    grade,
    critical_pass_rate,
    apply_critical_gate,
)


def aggregate_query_metrics(results: list[ClaimResult]) -> QueryMetrics:
    """Compute metrics for a single query from its claim results."""
    total = len(results)
    verdict_counts: dict[str, int] = defaultdict(int)
    for r in results:
        verdict_counts[r.verdict] += 1

    # v2: ``is_scorable`` now accepts everything except ``not_applicable``,
    # so ``insufficient_evidence`` claims contribute 0 to the numerator
    # and their full weight to the denominator.
    scorable = [r for r in results if is_scorable(r.verdict)]
    scorable_n = len(scorable)

    if scorable_n == 0:
        weighted_pre_gate = 0.0
    else:
        num = sum(claim_weight(r.criticality) * verdict_score(r.verdict) for r in scorable)
        den = sum(claim_weight(r.criticality) for r in scorable)
        weighted_pre_gate = num / den if den else 0.0

    # Critical claim pass rate. Distinct from the gate factor: the
    # rate is just the fraction of critical claims at ``supported``,
    # while ``critical_gate_factor`` is the multiplier the headline
    # actually pays. Both are reported.
    crit_rate = critical_pass_rate(results)

    final_score, gate_factor = apply_critical_gate(weighted_pre_gate, crit_rate)

    insuf = verdict_counts.get("insufficient_evidence", 0)
    insuf_rate = (insuf / total) if total else 0.0

    # Type-specific rates. Use the v2 ``is_scorable`` so type rates are
    # not artificially lifted by silently dropping ``insufficient_evidence``
    # claims out of the denominator the way v1.5 did.
    def _type_rate(claim_type: str) -> float | None:
        typed = [r for r in results if r.claim_type == claim_type]
        if not typed:
            return None
        scored = [r for r in typed if is_scorable(r.verdict)]
        if not scored:
            return 0.0
        return sum(verdict_score(r.verdict) for r in scored) / len(scored)

    # State changes (interaction closure) now route through the
    # ``state_transition`` method; ``navigation`` stays its own method.
    interaction_types = {"state_transition", "navigation"}
    interaction_claims = [r for r in results if r.claim_type in interaction_types]
    if interaction_claims:
        i_scored = [r for r in interaction_claims if is_scorable(r.verdict)]
        interaction_rate = (
            sum(verdict_score(r.verdict) for r in i_scored) / len(i_scored)
            if i_scored else 0.0
        )
    else:
        interaction_rate = None

    contradiction_count = verdict_counts.get("contradicted", 0)
    contradiction_rate = (contradiction_count / total) if total else 0.0

    type_counts: dict[str, int] = defaultdict(int)
    for r in results:
        type_counts[r.claim_type] += 1

    # Axis-2 capability breakdown.
    semantic_counts: dict[str, int] = defaultdict(int)
    for r in results:
        st = getattr(r, "semantic_type", None)
        if st:
            semantic_counts[st] += 1

    # ── Native CUA synthetic-salvage diagnostics ──────────────────────────
    # Each claim's ``native_tool_use_stats`` carries ``mode`` in
    # {native_only, synthetic_only, mixed, text_only}. We slice the count
    # of claims per mode and a single "anything synthetic in the loop"
    # rate so the report can flag "the gateway dropped the beta header
    # again" in one glance.
    native_mode_buckets: dict[str, int] = {
        "native_only": 0,
        "synthetic_only": 0,
        "mixed": 0,
        "text_only": 0,
    }
    native_claims_total = 0
    native_synth_claims = 0
    for r in results:
        stats = getattr(r, "native_tool_use_stats", None) or {}
        if not stats:
            continue
        mode = str(stats.get("mode") or "")
        if mode not in native_mode_buckets:
            continue
        native_mode_buckets[mode] += 1
        native_claims_total += 1
        if mode in ("synthetic_only", "mixed"):
            native_synth_claims += 1
    native_synth_rate = (
        native_synth_claims / native_claims_total if native_claims_total else 0.0
    )

    # ── G2 / J1 resilience diagnostics ────────────────────────────────────
    page_load_failed_claims = 0
    retry_total = 0
    cascade_triggered = 0
    for r in results:
        plf = getattr(r, "page_load_failure", None) or {}
        if not plf:
            continue
        attempts = int(plf.get("attempts") or 1)
        if attempts > 1:
            # attempts=N means N-1 reload retries happened.
            retry_total += attempts - 1
            page_load_failed_claims += 1
        if plf.get("triggered"):
            cascade_triggered += 1

    # Non-supported attribution breakdown (diagnostic only; does not affect
    # the score). Lets the report separate "CUA never drove the app" from
    # "the app genuinely failed" when comparing CUAs.
    attribution_counts: dict[str, int] = defaultdict(int)
    for r in results:
        attr = getattr(r, "verdict_attribution", None) or {}
        cat = attr.get("category") if isinstance(attr, dict) else None
        if cat:
            attribution_counts[cat] += 1

    return QueryMetrics(
        weighted_func_score=final_score,
        scorable_claim_count=scorable_n,
        total_claim_count=total,
        critical_claim_pass_rate=crit_rate,
        insufficient_evidence_count=insuf,
        insufficient_evidence_rate=insuf_rate,
        interaction_closure_rate=interaction_rate,
        correctness_rate=_type_rate("value_correctness"),
        persistence_rate=_type_rate("persistence"),
        contradiction_rate=contradiction_rate,
        claim_type_counts=dict(type_counts),
        semantic_type_counts=dict(semantic_counts),
        verdict_counts=dict(verdict_counts),
        weighted_func_score_pre_gate=weighted_pre_gate,
        critical_gate_factor=gate_factor,
        native_synthetic_claim_rate=round(native_synth_rate, 4),
        native_synthetic_only_claims=native_mode_buckets["synthetic_only"],
        native_mixed_claims=native_mode_buckets["mixed"],
        native_native_only_claims=native_mode_buckets["native_only"],
        native_text_only_claims=native_mode_buckets["text_only"],
        page_load_failure_claims=page_load_failed_claims,
        page_load_retry_total=retry_total,
        page_load_cascade_triggered_claims=cascade_triggered,
        verdict_attribution_counts=dict(attribution_counts),
    )


def aggregate_dataset_metrics(query_results: list[QueryResult]) -> DatasetMetrics:
    """Compute dataset-level aggregated metrics."""
    total_queries = len(query_results)
    total_claims = sum(len(r.claim_results) for r in query_results)

    if total_queries == 0:
        return DatasetMetrics(
            total_queries=0, total_claims=0, overall_func_score=0.0,
            grade="D", mean_critical_claim_pass_rate=None,
            mean_insufficient_evidence_rate=0.0, mean_contradiction_rate=0.0,
        )

    overall = sum(r.func_score for r in query_results) / total_queries

    # Verdict distribution
    verdict_dist: dict[str, int] = defaultdict(int)
    for r in query_results:
        for cr in r.claim_results:
            verdict_dist[cr.verdict] += 1

    # By difficulty
    by_diff: dict[str, list[float]] = defaultdict(list)
    for r in query_results:
        label = "★" * r.difficulty
        by_diff[label].append(r.func_score)
    score_by_difficulty = {d: sum(s) / len(s) for d, s in sorted(by_diff.items())}

    # By category
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in query_results:
        by_cat[r.category].append(r.func_score)
    score_by_category = {c: sum(s) / len(s) for c, s in sorted(by_cat.items())}

    # Axis 1: by claim type (verification method)
    type_scores: dict[str, list[float]] = defaultdict(list)
    for r in query_results:
        for cr in r.claim_results:
            if is_scorable(cr.verdict):
                type_scores[cr.claim_type].append(verdict_score(cr.verdict))
    score_by_claim_type = {
        t: sum(s) / len(s) for t, s in type_scores.items() if s
    }

    # Axis 2: capability profile — by semantic_type.
    sem_scores: dict[str, list[float]] = defaultdict(list)
    for r in query_results:
        for cr in r.claim_results:
            st = getattr(cr, "semantic_type", None)
            if st and is_scorable(cr.verdict):
                sem_scores[st].append(verdict_score(cr.verdict))
    score_by_semantic_type = {
        t: sum(s) / len(s) for t, s in sem_scores.items() if s
    }

    # Per-claim_type validator reliability: how often each verification
    # method's programmatic validator decided on its own vs deferred to the
    # LLM judge. ``llm_used`` = the claim's ``used_validators`` contains
    # ``llm_judge``; ``programmatic_decided`` = a programmatic validator ran
    # and the LLM judge did NOT. ``programmatic_coverage`` is the fraction
    # decided programmatically — a proxy for the evaluator's own reliability
    # on that method (high coverage = less reliance on the slower LLM path).
    _PROG_VALIDATOR_TAGS = (
        "structural_validator", "navigation_validator", "state_transition_validator",
        "value_correctness_validator", "runtime_integrity_validator",
        "persistence_validator", "geo_semantics_validator",
        # legacy tags from pre-refactor payloads
        "presence_validator", "action_effect_validator",
    )
    reliability_acc: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "programmatic_decided": 0, "llm_used": 0}
    )
    for r in query_results:
        for cr in r.claim_results:
            used = set(getattr(cr, "used_validators", []) or [])
            acc = reliability_acc[cr.claim_type]
            acc["count"] += 1
            llm_used = "llm_judge" in used
            prog_ran = any(tag in used for tag in _PROG_VALIDATOR_TAGS)
            if llm_used:
                acc["llm_used"] += 1
            if prog_ran and not llm_used:
                acc["programmatic_decided"] += 1
    validator_reliability = {
        ct: {
            **vals,
            "programmatic_coverage": round(
                vals["programmatic_decided"] / vals["count"], 4
            ) if vals["count"] else 0.0,
        }
        for ct, vals in reliability_acc.items()
    }

    # Means from query metrics
    crit_rates = [
        r.metrics.critical_claim_pass_rate
        for r in query_results
        if r.metrics and r.metrics.critical_claim_pass_rate is not None
    ]
    insuf_rates = [
        r.metrics.insufficient_evidence_rate
        for r in query_results if r.metrics
    ]
    contra_rates = [
        r.metrics.contradiction_rate
        for r in query_results if r.metrics
    ]

    # ── v2 scoring decomposition rollup ─────────────────────────────────
    # Average the pre-gate score and the gate factor across queries, so
    # the dataset-level report can show the same decomposition the
    # per-query view does. ``pre_gate`` is None when no query stored
    # the v2 diagnostics (e.g. payloads from older runs); we treat that
    # as "unavailable" and drop the field rather than mis-averaging.
    pre_gate_vals = [
        r.metrics.weighted_func_score_pre_gate
        for r in query_results
        if r.metrics and r.metrics.weighted_func_score_pre_gate is not None
    ]
    gate_factors = [
        r.metrics.critical_gate_factor
        for r in query_results
        if r.metrics and r.metrics.critical_gate_factor is not None
    ]
    mean_pre_gate = (
        round(sum(pre_gate_vals) / len(pre_gate_vals), 4)
        if pre_gate_vals else None
    )
    mean_gate_factor = (
        round(sum(gate_factors) / len(gate_factors), 4)
        if gate_factors else 1.0
    )

    # ── Native CUA synthetic-salvage rollup ─────────────────────────────
    synth_rates = [
        r.metrics.native_synthetic_claim_rate
        for r in query_results
        if r.metrics and (
            r.metrics.native_synthetic_only_claims
            + r.metrics.native_mixed_claims
            + r.metrics.native_native_only_claims
            + r.metrics.native_text_only_claims
        ) > 0
    ]
    mean_synth_rate = round(sum(synth_rates) / len(synth_rates), 4) if synth_rates else 0.0
    native_mode_dist: dict[str, int] = defaultdict(int)
    for r in query_results:
        if not r.metrics:
            continue
        native_mode_dist["native_only"] += r.metrics.native_native_only_claims
        native_mode_dist["synthetic_only"] += r.metrics.native_synthetic_only_claims
        native_mode_dist["mixed"] += r.metrics.native_mixed_claims
        native_mode_dist["text_only"] += r.metrics.native_text_only_claims

    # ── Resilience rollup ───────────────────────────────────────────────
    plf_claims = sum(r.metrics.page_load_failure_claims for r in query_results if r.metrics)
    plf_retry_total = sum(r.metrics.page_load_retry_total for r in query_results if r.metrics)
    plf_cascade = sum(r.metrics.page_load_cascade_triggered_claims for r in query_results if r.metrics)

    return DatasetMetrics(
        total_queries=total_queries,
        total_claims=total_claims,
        overall_func_score=round(overall, 4),
        grade=grade(overall),
        mean_critical_claim_pass_rate=(round(sum(crit_rates) / len(crit_rates), 4) if crit_rates else None),
        mean_insufficient_evidence_rate=(round(sum(insuf_rates) / len(insuf_rates), 4) if insuf_rates else 0.0),
        mean_contradiction_rate=(round(sum(contra_rates) / len(contra_rates), 4) if contra_rates else 0.0),
        score_by_difficulty=score_by_difficulty,
        score_by_category=score_by_category,
        score_by_claim_type=score_by_claim_type,
        score_by_semantic_type=score_by_semantic_type,
        validator_reliability=validator_reliability,
        verdict_distribution=dict(verdict_dist),
        mean_weighted_func_score_pre_gate=mean_pre_gate,
        mean_critical_gate_factor=mean_gate_factor,
        mean_native_synthetic_claim_rate=mean_synth_rate,
        native_mode_distribution=dict(native_mode_dist),
        page_load_failure_claims=plf_claims,
        page_load_retry_total=plf_retry_total,
        page_load_cascade_triggered_claims=plf_cascade,
    )
