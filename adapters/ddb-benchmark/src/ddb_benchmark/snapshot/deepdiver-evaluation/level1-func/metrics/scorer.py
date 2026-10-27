"""
Verdict-to-score mapping, claim weight, and critical-gate helpers.

Semantics:

* Score is strictly binary at the claim level — ``supported`` is 1,
  everything that is not ``not_applicable`` is 0.
* ``is_scorable`` returns True for every verdict except
  ``not_applicable``. ``insufficient_evidence`` *does* count toward the
  denominator now; previously it was excluded which let models inflate
  scores by giving up.
* :func:`apply_critical_gate` multiplies the weighted score by
  ``min(1, crit_pass / threshold)``. The default threshold is 0.8: if a
  query's critical-claim pass rate is below 80%, the headline score is
  scaled down proportionally — losing core functionality can no longer
  hide behind a wall of supported minor claims.
"""
from __future__ import annotations
from typing import Iterable
from types_.verdicts import (
    VERDICT_SCORES,
    NON_APPLICABLE_VERDICTS,
    CRITICALITY_WEIGHTS,
)


def verdict_score(verdict: str) -> float:
    return float(VERDICT_SCORES.get(verdict, 0.0))


def claim_weight(criticality: str) -> float:
    return float(CRITICALITY_WEIGHTS.get(criticality, CRITICALITY_WEIGHTS.get("major", 1.0)))


def is_scorable(verdict: str) -> bool:
    """v2: a claim contributes to the denominator unless it's N/A.

    Note: ``WEIGHTED_VERDICTS`` is retained for backward compatibility
    but the canonical check is "verdict is not non-applicable", because
    the set of non-applicable verdicts is much smaller than the set of
    weighted ones and the rule is easier to reason about.
    """
    return verdict not in NON_APPLICABLE_VERDICTS


def grade(score: float) -> str:
    if score >= 0.9:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= 0.6:
        return "C"
    return "D"


# ── v2 critical-gate helpers ─────────────────────────────────────────

# Default threshold for the critical-claim gate. Reads as "an app is
# expected to land 80% of its critical claims; below that we start
# scaling the headline score down proportionally". Picked at 0.8 to
# leave a 1-claim slack on a typical 5-7 critical-claim query — one
# unlucky CUA flake shouldn't cliff-edge the score.
CRITICAL_GATE_THRESHOLD = 0.8


def critical_pass_rate(results: Iterable) -> float | None:
    """Fraction of *critical* claims that landed at ``supported``.

    Returns ``None`` when the query has no critical claims, so callers
    can skip the gate (or treat the factor as 1.0).
    """
    crit = [r for r in results if getattr(r, "criticality", None) == "critical"]
    if not crit:
        return None
    crit_supp = sum(1 for r in crit if getattr(r, "verdict", "") == "supported")
    return crit_supp / len(crit)


def critical_gate_factor(
    crit_pass: float | None,
    threshold: float = CRITICAL_GATE_THRESHOLD,
) -> float:
    """Multiplier applied to the weighted score.

    * No critical claims → no gate, factor 1.0.
    * Negative / zero threshold → gate disabled, factor 1.0.
    * Otherwise → ``min(1, crit_pass / threshold)``.
    """
    if crit_pass is None:
        return 1.0
    if threshold <= 0:
        return 1.0
    return min(1.0, max(0.0, crit_pass) / threshold)


def apply_critical_gate(
    weighted: float,
    crit_pass: float | None,
    threshold: float = CRITICAL_GATE_THRESHOLD,
) -> tuple[float, float]:
    """Apply the gate and return ``(final_score, gate_factor)``.

    Pure function so aggregator callers can unit-test the gate in
    isolation and so report renderers can display the factor.
    """
    factor = critical_gate_factor(crit_pass, threshold)
    return weighted * factor, factor
