"""
Verdict taxonomy and scoring constants (binary rubric).

* A claim is scored binary: ``supported`` is 1, ``contradicted`` and
  ``insufficient_evidence`` are 0. There is no half-credit verdict.
* ``insufficient_evidence`` is part of the weighted denominator (still
  0 in the numerator), so a model cannot inflate its score by giving
  up — "I couldn't tell" is the score-0 outcome it really is.
* Only ``not_applicable`` stays excluded from the denominator; that is
  the legitimate "this claim does not apply in the current context"
  exit hatch (e.g. visual_semantics on a non-rendered map). The set
  ``NON_APPLICABLE_VERDICTS`` is the single source of truth for what
  scoring excludes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"


VERDICT_SCORES: dict[str, float] = {
    "supported": 1.0,
    "contradicted": 0.0,
    "insufficient_evidence": 0.0,
    "not_applicable": 0.0,
}

# Verdicts that are *excluded* from both numerator and denominator of
# the weighted score. Exactly one verdict — ``not_applicable`` — qualifies
# under v2; everything else (including ``insufficient_evidence``) is now
# scored as 0 inside the denominator.
NON_APPLICABLE_VERDICTS = frozenset({"not_applicable"})

# Verdicts that contribute to the weighted score's denominator. Under
# v2 this is "everything except not_applicable". The constant is kept
# for backward compatibility with callers that still consult it.
WEIGHTED_VERDICTS = frozenset({
    "supported",
    "contradicted",
    "insufficient_evidence",
})

CRITICALITY_WEIGHTS: dict[str, float] = {
    "critical": 3.0,
    "major": 2.0,
    "minor": 1.0,
}


@dataclass
class ClaimVerdict:
    """Final verdict for a single claim."""

    claim_id: str
    verdict: str
    confidence: float = 0.0
    reason: str = ""
    evidence_sufficient: bool = False
    used_validators: list[str] = field(default_factory=list)
    contradiction_flags: list[str] = field(default_factory=list)
