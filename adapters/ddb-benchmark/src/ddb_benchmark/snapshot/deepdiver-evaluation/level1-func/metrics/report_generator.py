"""
Generate JSON and Markdown reports from evaluation results.
"""
from __future__ import annotations
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path
from types_.results import ClaimResult, QueryMetrics, QueryResult
from .aggregator import aggregate_dataset_metrics
from .scorer import grade

logger = logging.getLogger(__name__)


# ── Per-query persistence ──────────────────────────────────────
#
# Per-query JSON files live under ``<output_dir>/<stem>/<query_id>.json`` and
# act as the ground truth for evaluation results. The aggregated
# ``<stem>.json`` / ``<stem>.md`` are derived from them at the end of a run
# (or from disk on resume), so any crash/interrupt only loses the *currently
# running* query, not previously completed ones.


def save_query_result_json(
    result: QueryResult,
    path: Path | str,
    meta_review: dict | None = None,
) -> Path:
    """Persist a single :class:`QueryResult` to a JSON file (atomic-ish).

    If ``meta_review`` is a non-empty dict, it is appended as a top-level
    ``meta_review`` field of the payload (next to ``query_result``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "saved_at": datetime.now().isoformat(),
        "query_result": dataclasses.asdict(result),
    }
    if meta_review:
        payload["meta_review"] = meta_review
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)

    # Full-fidelity runtime audit log: one JSON line per backend call across
    # all claims, written next to <qid>.json for grep-able forensics. The
    # compact rollup already lives inside each ClaimResult.runtime_audit;
    # this file is the raw evidence channel. Best-effort: never let an audit
    # write failure break result persistence.
    try:
        _write_audit_log_jsonl(result, p.parent / "audit_log.jsonl")
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to write audit_log.jsonl for %s: %s", p, e)
    return p


def _write_audit_log_jsonl(result: QueryResult, path: Path) -> None:
    """Flatten every claim's ``runtime_audit.backend_calls`` into a JSONL file.

    Skips writing entirely when no claim observed any backend traffic, so
    we don't litter the run dir with empty files.
    """
    lines: list[str] = []
    for cr in result.claim_results:
        audit = getattr(cr, "runtime_audit", None) or {}
        for call in audit.get("backend_calls", []) or []:
            if not isinstance(call, dict):
                continue
            row = {"query_id": result.query_id, "claim_id": cr.claim_id}
            row.update(call)
            lines.append(json.dumps(row, ensure_ascii=False))
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_partial_query_result_json(result: QueryResult, path: Path | str) -> Path:
    """Persist an in-progress :class:`QueryResult` (a subset of its claims).

    Used for claim-level resume: each time a claim finishes we checkpoint the
    claims evaluated so far so an interrupt mid-query can continue from the
    next unevaluated claim instead of restarting the whole query (which would
    re-spend CUA tokens and discard already-captured screenshots).

    Unlike :func:`save_query_result_json` this writes no audit log and tags the
    payload with ``partial: True``. The file lives at
    ``<query_dir>/<qid>/<qid>.partial.json`` so it is *not* picked up by
    :func:`_iter_per_query_json_paths` (whose stem == parent-name filter only
    matches the canonical ``<qid>.json``), keeping query-level resume intact.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "saved_at": datetime.now().isoformat(),
        "partial": True,
        "query_result": dataclasses.asdict(result),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


def load_partial_claim_results(path: Path | str) -> list[ClaimResult]:
    """Load the claim results from a partial file written by
    :func:`save_partial_query_result_json`.

    Returns an empty list when the file is missing or unparseable, so callers
    can treat it as "no prior progress".
    """
    qr = load_query_result_json(path)
    return list(qr.claim_results) if qr is not None else []


def load_query_result_json(path: Path | str) -> QueryResult | None:
    """Load a :class:`QueryResult` previously saved by :func:`save_query_result_json`.

    Returns ``None`` if the file is missing or unparseable.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        qr = data.get("query_result", data)
        claim_results = [ClaimResult(**c) for c in qr.get("claim_results", [])]
        metrics = qr.get("metrics")
        metrics_obj = QueryMetrics(**metrics) if metrics else None
        return QueryResult(
            query_id=qr["query_id"],
            title=qr["title"],
            category=qr["category"],
            difficulty=qr["difficulty"],
            claim_results=claim_results,
            func_score=qr["func_score"],
            metrics=metrics_obj,
        )
    except Exception as e:
        logger.warning("Failed to load %s: %s", p, e)
        return None


def _iter_per_query_json_paths(directory: Path) -> list[Path]:
    """Find ``<qid>.json`` under both legacy flat and new per-query subdir layouts.

    Layouts in the wild:
      * Legacy: ``<directory>/<qid>.json``
      * Current: ``<directory>/<qid>/<qid>.json``

    We dedupe by ``stem`` so a run that gets partially migrated (e.g. an
    older flat file + a newly-written subdir file for the same query) only
    yields one path — preferring the new layout because that's where
    screenshots live.
    """
    if not directory.exists() or not directory.is_dir():
        return []
    by_stem: dict[str, Path] = {}
    for p in directory.glob("*.json"):
        if p.name.endswith(".tmp"):
            continue
        by_stem.setdefault(p.stem, p)
    for p in directory.glob("*/*.json"):
        if p.name.endswith(".tmp"):
            continue
        # Only treat <qid>/<qid>.json as a per-query result. Anything else
        # nested (e.g. <stem>/<qid>/some_other.json) is ignored to keep
        # the scan from accidentally picking up overview/meta files.
        if p.stem == p.parent.name:
            by_stem[p.stem] = p
    return [by_stem[k] for k in sorted(by_stem)]


def load_query_results_from_dir(directory: Path | str) -> list[QueryResult]:
    """Load every ``<query_id>.json`` from ``directory`` (sorted by query_id).

    Supports both legacy flat ``<dir>/<qid>.json`` and current per-query
    subdir ``<dir>/<qid>/<qid>.json`` layouts. Files that fail to parse
    are skipped with a warning.
    """
    out: list[QueryResult] = []
    for p in _iter_per_query_json_paths(Path(directory)):
        qr = load_query_result_json(p)
        if qr is not None:
            out.append(qr)
    return out


def list_completed_query_ids(directory: Path | str) -> set[str]:
    """Return the set of query_ids that already have a saved per-query JSON.

    See :func:`_iter_per_query_json_paths` for the layouts recognised.
    """
    return {p.stem for p in _iter_per_query_json_paths(Path(directory))}


def generate_json_report(
    results: list[QueryResult],
    output_path: Path | str | None = None,
) -> dict:
    """Generate a structured JSON report."""
    ds = aggregate_dataset_metrics(results)

    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "framework": "deepdiver-evaluation level1-func",
            "version": "v2",
            "scoring_rubric": (
                "binary supported/contradicted; insufficient_evidence "
                "counts as 0 in denominator (only not_applicable is "
                "excluded); final = weighted * min(1, crit_pass / 0.8)"
            ),
        },
        "summary": {
            "total_queries": ds.total_queries,
            "total_claims": ds.total_claims,
            "overall_func_score": ds.overall_func_score,
            "grade": ds.grade,
            "mean_weighted_func_score_pre_gate": ds.mean_weighted_func_score_pre_gate,
            "mean_critical_gate_factor": ds.mean_critical_gate_factor,
            "mean_critical_claim_pass_rate": ds.mean_critical_claim_pass_rate,
            "mean_insufficient_evidence_rate": ds.mean_insufficient_evidence_rate,
            "mean_contradiction_rate": ds.mean_contradiction_rate,
            "verdict_distribution": ds.verdict_distribution,
            "score_by_difficulty": ds.score_by_difficulty,
            "score_by_category": ds.score_by_category,
            "score_by_claim_type": ds.score_by_claim_type,
            "score_by_semantic_type": ds.score_by_semantic_type,
            "validator_reliability": ds.validator_reliability,
            "native_cua_health": {
                "mean_synthetic_claim_rate": ds.mean_native_synthetic_claim_rate,
                "mode_distribution": ds.native_mode_distribution,
            },
            "resilience": {
                "page_load_failure_claims": ds.page_load_failure_claims,
                "page_load_retry_total": ds.page_load_retry_total,
                "page_load_cascade_triggered_claims": ds.page_load_cascade_triggered_claims,
            },
        },
        "details": [],
    }

    for r in results:
        m = r.metrics
        detail = {
            "query_id": r.query_id,
            "title": r.title,
            "category": r.category,
            "difficulty": r.difficulty,
            "func_score": round(r.func_score, 4),
            "grade": grade(r.func_score),
            "metrics": (
                {
                    "weighted_func_score": round(m.weighted_func_score, 4),
                    "weighted_func_score_pre_gate": (
                        round(m.weighted_func_score_pre_gate, 4)
                        if m.weighted_func_score_pre_gate is not None
                        else None
                    ),
                    "critical_gate_factor": round(m.critical_gate_factor, 4),
                    "scorable_claim_count": m.scorable_claim_count,
                    "total_claim_count": m.total_claim_count,
                    "critical_claim_pass_rate": m.critical_claim_pass_rate,
                    "insufficient_evidence_count": m.insufficient_evidence_count,
                    "insufficient_evidence_rate": round(m.insufficient_evidence_rate, 4),
                    "interaction_closure_rate": m.interaction_closure_rate,
                    "correctness_rate": m.correctness_rate,
                    "persistence_rate": m.persistence_rate,
                    "contradiction_rate": round(m.contradiction_rate, 4),
                    "claim_type_counts": m.claim_type_counts,
                    "semantic_type_counts": m.semantic_type_counts,
                    "verdict_counts": m.verdict_counts,
                    "native_synthetic_claim_rate": m.native_synthetic_claim_rate,
                    "native_mode_counts": {
                        "native_only": m.native_native_only_claims,
                        "synthetic_only": m.native_synthetic_only_claims,
                        "mixed": m.native_mixed_claims,
                        "text_only": m.native_text_only_claims,
                    },
                    "page_load_failure_claims": m.page_load_failure_claims,
                    "page_load_retry_total": m.page_load_retry_total,
                    "page_load_cascade_triggered_claims": m.page_load_cascade_triggered_claims,
                }
                if m else None
            ),
            "claims": [
                {
                    "claim_id": cr.claim_id,
                    "source_fc_id": cr.source_fc_id,
                    "checkpoint": cr.checkpoint,
                    "verification": cr.verification,
                    "claim_type": cr.claim_type,
                    "semantic_type": getattr(cr, "semantic_type", None),
                    "criticality": cr.criticality,
                    "verdict": cr.verdict,
                    "confidence": cr.confidence,
                    "reason": cr.reason,
                    "evidence_sufficient": cr.evidence_sufficient,
                    "used_validators": cr.used_validators,
                    "action_trace": cr.action_trace,
                    "action_summary": cr.action_summary,
                    "screenshots": cr.screenshots,
                    "native_tool_use_stats": cr.native_tool_use_stats,
                    "page_load_failure": cr.page_load_failure,
                }
                for cr in r.claim_results
            ],
        }
        report["details"].append(detail)

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def generate_markdown_report(
    results: list[QueryResult],
    output_path: Path | str | None = None,
) -> str:
    """Generate a human-readable Markdown report."""
    ds = aggregate_dataset_metrics(results)
    lines: list[str] = []

    lines.append("# DeepDiver Level-1 Functionality Evaluation Report (v2)")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(
        "> **v2 scoring**: binary supported(=1)/everything-else(=0) at "
        "the claim level; only `not_applicable` is excluded from the "
        "denominator (so `insufficient_evidence` now subtracts from the "
        "score instead of being a free pass). The headline is "
        "`weighted_func_score = pre_gate × min(1, crit_pass / 0.8)`, "
        "i.e. losing more than 20% of *critical* claims scales the "
        "score down."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Weighted FuncScore (post-gate)** | **{ds.overall_func_score:.1%}** |")
    lines.append(f"| **Grade** | **{ds.grade}** |")
    if ds.mean_weighted_func_score_pre_gate is not None:
        lines.append(
            f"| Pre-gate weighted score | "
            f"{ds.mean_weighted_func_score_pre_gate:.1%} |"
        )
    lines.append(
        f"| Mean critical-gate factor | "
        f"{ds.mean_critical_gate_factor:.1%} |"
    )
    lines.append(f"| Queries evaluated | {ds.total_queries} |")
    lines.append(f"| Total claims | {ds.total_claims} |")
    if ds.mean_critical_claim_pass_rate is not None:
        lines.append(f"| Mean critical pass rate | {ds.mean_critical_claim_pass_rate:.1%} |")
    lines.append(f"| Mean insufficient evidence rate | {ds.mean_insufficient_evidence_rate:.1%} |")
    lines.append(f"| Mean contradiction rate | {ds.mean_contradiction_rate:.1%} |")
    lines.append("")

    lines.append("### Verdict Distribution")
    lines.append("")
    for v, c in sorted(ds.verdict_distribution.items(), key=lambda x: -x[1]):
        lines.append(f"- `{v}`: {c}")
    lines.append("")

    if any(ds.native_mode_distribution.values()):
        lines.append("### Native CUA Health (synthetic-salvage)")
        lines.append("")
        lines.append(
            f"- mean synthetic_claim_rate: "
            f"**{ds.mean_native_synthetic_claim_rate:.1%}**"
            f"  (close to 0 means the gateway honored the beta header;"
            f" close to 1 means almost every CUA action had to be salvaged"
            f" from text)"
        )
        for mode in ("native_only", "mixed", "synthetic_only", "text_only"):
            if ds.native_mode_distribution.get(mode):
                lines.append(
                    f"- `{mode}`: {ds.native_mode_distribution[mode]} claims"
                )
        lines.append("")

    if (
        ds.page_load_failure_claims
        or ds.page_load_retry_total
        or ds.page_load_cascade_triggered_claims
    ):
        lines.append("### Resilience (G2 + J1)")
        lines.append("")
        lines.append(
            f"- claims that needed a reload-retry: "
            f"{ds.page_load_failure_claims}"
        )
        lines.append(
            f"- total reload-retries fired: {ds.page_load_retry_total}"
        )
        lines.append(
            f"- claims that exhausted retries and fell to J1 cascade: "
            f"{ds.page_load_cascade_triggered_claims}"
        )
        lines.append("")

    if ds.score_by_difficulty:
        lines.append("## Score by Difficulty")
        lines.append("")
        lines.append("| Difficulty | FuncScore | Grade |")
        lines.append("|------------|-----------|-------|")
        for d, s in ds.score_by_difficulty.items():
            lines.append(f"| {d} | {s:.1%} | {grade(s)} |")
        lines.append("")

    if ds.score_by_category:
        lines.append("## Score by Category")
        lines.append("")
        lines.append("| Category | FuncScore | Grade |")
        lines.append("|----------|-----------|-------|")
        for cat, s in sorted(ds.score_by_category.items(), key=lambda x: -x[1]):
            lines.append(f"| {cat} | {s:.1%} | {grade(s)} |")
        lines.append("")

    # Axis 2 — capability profile (model ability breakdown).
    if ds.score_by_semantic_type:
        lines.append("## Capability Profile (by semantic_type)")
        lines.append("")
        lines.append("| Capability (semantic_type) | FuncScore | Grade |")
        lines.append("|----------------------------|-----------|-------|")
        for st, s in sorted(ds.score_by_semantic_type.items(), key=lambda x: -x[1]):
            lines.append(f"| {st} | {s:.1%} | {grade(s)} |")
        lines.append("")

    # Axis 1 — verification-method score + evaluator reliability (coverage).
    if ds.score_by_claim_type or ds.validator_reliability:
        lines.append("## Evaluator Reliability (by claim_type / verification method)")
        lines.append("")
        lines.append(
            "| Method (claim_type) | FuncScore | Claims | Programmatic coverage | LLM used |"
        )
        lines.append(
            "|---------------------|-----------|--------|-----------------------|----------|"
        )
        methods = set(ds.score_by_claim_type) | set(ds.validator_reliability)
        for ct in sorted(methods):
            score = ds.score_by_claim_type.get(ct)
            rel = ds.validator_reliability.get(ct, {})
            count = rel.get("count", 0)
            cov = rel.get("programmatic_coverage", 0.0)
            llm = rel.get("llm_used", 0)
            score_s = f"{score:.1%}" if score is not None else "-"
            lines.append(
                f"| {ct} | {score_s} | {count} | {cov:.1%} | {llm} |"
            )
        lines.append("")

    emoji_map = {
        "supported": "OK",
        "contradicted": "X",
        "insufficient_evidence": "?",
        "not_applicable": "-",
    }

    lines.append("## Per-Query Details")
    lines.append("")
    for r in results:
        lines.append(f"### {r.query_id} -- {r.title} ({grade(r.func_score)} {r.func_score:.1%})")
        if r.metrics:
            if (
                r.metrics.weighted_func_score_pre_gate is not None
                and r.metrics.critical_gate_factor < 0.999
            ):
                # Show the decomposition only when the gate actually
                # cut into the score, otherwise the line is noise.
                lines.append(
                    f"- Score decomposition: pre_gate="
                    f"{r.metrics.weighted_func_score_pre_gate:.1%} × "
                    f"crit_gate={r.metrics.critical_gate_factor:.2f} = "
                    f"{r.func_score:.1%}"
                )
            if r.metrics.critical_claim_pass_rate is not None:
                lines.append(f"- Critical pass rate: {r.metrics.critical_claim_pass_rate:.1%}")
            lines.append(
                f"- Insufficient evidence: {r.metrics.insufficient_evidence_count}"
                f" / {r.metrics.total_claim_count}"
            )
            native_claims_total = (
                r.metrics.native_native_only_claims
                + r.metrics.native_synthetic_only_claims
                + r.metrics.native_mixed_claims
                + r.metrics.native_text_only_claims
            )
            if native_claims_total > 0:
                lines.append(
                    f"- CUA native synthetic rate: "
                    f"{r.metrics.native_synthetic_claim_rate:.1%}"
                    f"  (native_only={r.metrics.native_native_only_claims},"
                    f" mixed={r.metrics.native_mixed_claims},"
                    f" synthetic_only={r.metrics.native_synthetic_only_claims},"
                    f" text_only={r.metrics.native_text_only_claims})"
                )
            if (
                r.metrics.page_load_failure_claims
                or r.metrics.page_load_cascade_triggered_claims
            ):
                lines.append(
                    f"- Page-load resilience: "
                    f"failures={r.metrics.page_load_failure_claims},"
                    f" retries={r.metrics.page_load_retry_total},"
                    f" cascade_to_J1={r.metrics.page_load_cascade_triggered_claims}"
                )
        lines.append("")
        lines.append("| Claim | Method | Capability | Crit | Verdict | Conf | Validators | Reason |")
        lines.append("|-------|--------|-----------|------|---------|------|------------|--------|")
        for cr in r.claim_results:
            emoji = emoji_map.get(cr.verdict, "?")
            reason_short = (cr.reason or "").replace("|", " ").replace("\n", " ")[:120]
            validators = ",".join(cr.used_validators) if cr.used_validators else "-"
            sem = getattr(cr, "semantic_type", None) or "-"
            lines.append(
                f"| {cr.claim_id} | {cr.claim_type} | {sem} | {cr.criticality} "
                f"| {emoji} {cr.verdict} | {cr.confidence:.0%} "
                f"| {validators} | {reason_short} |"
            )
        lines.append("")

    md_text = "\n".join(lines)

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md_text, encoding="utf-8")

    return md_text
