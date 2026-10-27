"""
Data types for evaluation claims and seed queries.

Aligned with functionality_v1.5.md claim taxonomy.

Two-axis (factored) taxonomy
============================
The claim taxonomy is factored into two orthogonal axes, each sized by its
own criterion (see functionality_v1.5.md 5.3):

- ``ClaimType`` — the *verification method* / routing key. One value <=> one
  validator branch with a well-defined ⟨evidence, procedure, decision⟩
  contract. Small and stable (7 values). This is what the verifier routes on.
- ``SemanticType`` — the *capability* being asserted. Richer (12 values),
  used for reporting / diagnostic breakdowns and as criticality anchors. It
  never participates in routing directly.

``SemanticType`` maps many-to-one onto ``ClaimType`` via
``CAPABILITY_TO_METHOD`` (a total, CI-checkable function). Legacy single-axis
values (the historical 12-type vocabulary, plus ``action_effect``) are mapped
back to ``(claim_type, semantic_type)`` via ``LEGACY_TYPE_ALIAS`` so older
seed JSONLs keep working.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class ClaimType(str, Enum):
    """Axis 1: verification method (routing key). One value per validator."""

    STRUCTURAL = "structural"           # static existence / count / enumeration
    NAVIGATION = "navigation"           # view / route change
    STATE_TRANSITION = "state_transition"  # action -> observable state change
    VALUE_CORRECTNESS = "value_correctness"  # displayed value / sort / stat / date legality
    PERSISTENCE = "persistence"         # survives reload / revisit
    PERCEPTUAL = "perceptual"           # visual semantics / layout / color / animation / geo
    RUNTIME_INTEGRITY = "runtime_integrity"  # real API call / console health / hit-test


class SemanticType(str, Enum):
    """Axis 2: capability asserted (reporting / criticality anchor)."""

    PRESENCE = "presence"
    NAVIGATION = "navigation"
    INTERACTION = "interaction"
    FUNCTIONALITY = "functionality"
    CONTENT = "content"
    COMPUTATION = "computation"
    PERSISTENCE = "persistence"
    VISUAL_SEMANTICS = "visual_semantics"
    VISUAL_DESIGN = "visual_design"
    DATA_INTEGRATION = "data_integration"
    ACCESSIBILITY = "accessibility"
    RUNTIME_HEALTH = "runtime_health"


# Total function semantic_type -> claim_type (CI-checkable).
CAPABILITY_TO_METHOD: dict[str, str] = {
    SemanticType.PRESENCE.value: ClaimType.STRUCTURAL.value,
    SemanticType.NAVIGATION.value: ClaimType.NAVIGATION.value,
    SemanticType.INTERACTION.value: ClaimType.STATE_TRANSITION.value,
    SemanticType.FUNCTIONALITY.value: ClaimType.STATE_TRANSITION.value,
    SemanticType.CONTENT.value: ClaimType.VALUE_CORRECTNESS.value,
    SemanticType.COMPUTATION.value: ClaimType.VALUE_CORRECTNESS.value,
    SemanticType.PERSISTENCE.value: ClaimType.PERSISTENCE.value,
    SemanticType.VISUAL_SEMANTICS.value: ClaimType.PERCEPTUAL.value,
    SemanticType.VISUAL_DESIGN.value: ClaimType.PERCEPTUAL.value,
    SemanticType.DATA_INTEGRATION.value: ClaimType.RUNTIME_INTEGRITY.value,
    SemanticType.ACCESSIBILITY.value: ClaimType.RUNTIME_INTEGRITY.value,
    SemanticType.RUNTIME_HEALTH.value: ClaimType.RUNTIME_INTEGRITY.value,
}

CLAIM_TYPES: frozenset[str] = frozenset(ct.value for ct in ClaimType)
SEMANTIC_TYPES: frozenset[str] = frozenset(st.value for st in SemanticType)

# Legacy single-axis claim_type values -> (claim_type, semantic_type).
# Covers the historical 12-type prompt vocabulary plus ``action_effect``
# (the previous canonical name for state_transition).
LEGACY_TYPE_ALIAS: dict[str, tuple[str, str]] = {
    "presence": (ClaimType.STRUCTURAL.value, SemanticType.PRESENCE.value),
    "navigation": (ClaimType.NAVIGATION.value, SemanticType.NAVIGATION.value),
    "interaction": (ClaimType.STATE_TRANSITION.value, SemanticType.INTERACTION.value),
    "functionality": (ClaimType.STATE_TRANSITION.value, SemanticType.FUNCTIONALITY.value),
    "action_effect": (ClaimType.STATE_TRANSITION.value, SemanticType.INTERACTION.value),
    "content": (ClaimType.VALUE_CORRECTNESS.value, SemanticType.CONTENT.value),
    "computation": (ClaimType.VALUE_CORRECTNESS.value, SemanticType.COMPUTATION.value),
    "persistence": (ClaimType.PERSISTENCE.value, SemanticType.PERSISTENCE.value),
    "visual_semantics": (ClaimType.PERCEPTUAL.value, SemanticType.VISUAL_SEMANTICS.value),
    "visual_design": (ClaimType.PERCEPTUAL.value, SemanticType.VISUAL_DESIGN.value),
    "data_integration": (ClaimType.RUNTIME_INTEGRITY.value, SemanticType.DATA_INTEGRATION.value),
    "interaction_accessibility": (ClaimType.RUNTIME_INTEGRITY.value, SemanticType.ACCESSIBILITY.value),
    "accessibility": (ClaimType.RUNTIME_INTEGRITY.value, SemanticType.ACCESSIBILITY.value),
    "runtime_health": (ClaimType.RUNTIME_INTEGRITY.value, SemanticType.RUNTIME_HEALTH.value),
}


def resolve_types(
    claim_type: str | None,
    semantic_type: str | None = None,
) -> tuple[str, str]:
    """Resolve a raw ``(claim_type, semantic_type)`` pair to canonical values.

    Resolution order (robust to legacy single-axis data):

    1. If ``semantic_type`` is a known capability, it wins: the canonical
       ``claim_type`` is derived from ``CAPABILITY_TO_METHOD`` (any provided
       ``claim_type`` that disagrees is overridden by the mapping).
    2. Else if the raw ``claim_type`` is a known verification method, keep it
       and leave ``semantic_type`` unknown (caller may default it).
    3. Else treat ``claim_type`` as a legacy single-axis value and expand via
       ``LEGACY_TYPE_ALIAS``.
    4. Fallback: ``(structural, presence)``.
    """
    st = (semantic_type or "").strip()
    ct = (claim_type or "").strip()

    if st in SEMANTIC_TYPES:
        return CAPABILITY_TO_METHOD[st], st
    if ct in CLAIM_TYPES:
        # Canonical method given without (valid) capability; mirror it where
        # a capability shares the same name (navigation / persistence), else
        # leave the capability as the method name for downstream defaulting.
        return ct, st if st in SEMANTIC_TYPES else ct
    if ct in LEGACY_TYPE_ALIAS:
        return LEGACY_TYPE_ALIAS[ct]
    return ClaimType.STRUCTURAL.value, SemanticType.PRESENCE.value


class Criticality(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class EvalClaim:
    """One atomic evaluation unit, from atomic_functional_checkpoints or coarse FC."""

    id: str
    source_fc_id: str
    checkpoint: str
    verification: str
    claim_type: str  # Axis 1 routing key (canonical verification method)
    criticality: str
    semantic_type: str | None = None  # Axis 2 capability label (reporting)
    # Per-query verification guideline (Stage-3, model-agnostic). Copied from
    # the owning SeedQuery onto every claim so the verifier can hand it to the
    # CUA without threading query objects through the pipeline. Empty when the
    # query has no guideline. MUST NOT reveal the expected verdict.
    verification_guideline: str = ""
    # ── Multi-session collaboration (0702) ──────────────────────────
    # When ``True`` this checkpoint asserts a cross-identity property that a
    # single anonymous session cannot verify honestly: realtime sync
    # (A writes / B reads), shared read-write, or per-identity isolation
    # (rls_private, A writes / B must NOT see). The verifier then generates a
    # one-off unique marker and exposes the ``delegate_to_viewer`` tool so the
    # main CUA can spawn a second, isolated viewer session. Empty / False
    # keeps the historical single-session behaviour byte-for-byte.
    multi_session: bool = False
    # Named collaborators for this checkpoint, e.g.
    # ``["viewer_a", "viewer_b"]``. Purely descriptive: routed into the goal
    # so the CUA knows how many distinct identities the protocol needs.
    participants: list[str] = field(default_factory=list)


@dataclass
class SeedQuery:
    """A seed query loaded from JSONL, with its associated claims."""

    id: str
    category: str
    difficulty: int
    difficulty_label: str
    title: str
    query: str
    eval_claims: list[EvalClaim] = field(default_factory=list)
    # Per-query verification guideline (Stage-3, model-agnostic). Loaded from
    # the JSONL ``verification_guideline`` field; propagated to each EvalClaim.
    verification_guideline: str = ""
    # File-upload fixtures available to the harness for this query. Each entry
    # is ``{"name": str, "path": str}`` (path absolute or relative to the run
    # CWD). The orchestrator registers these on the env so the CUA's
    # ``upload_file`` tool / any native file picker can attach them.
    fixtures: list[dict] = field(default_factory=list)
    # Query-level roll-up of ``EvalClaim.multi_session``: ``True`` when ANY
    # claim in this query needs the multi-session (viewer subagent) protocol.
    # Handy for routing / reporting; the per-claim flag is still the source of
    # truth the verifier acts on. Defaults False so single-session queries are
    # unaffected.
    multi_session: bool = False
    # Pre-minted per-identity preview links for multi-session collaboration,
    # keyed by participant role, e.g.
    # ``{"creator": "https://.../?token=...", "viewer_a": "https://.../?viewer=..."}``.
    # The main CUA session opens the app under the ``creator`` identity (from
    # the URL registry); every OTHER participant a claim needs (``viewer_a``,
    # ``viewer_b``, ...) is handed the matching ``?viewer=<token>`` link here so
    # the viewer subagent joins the SAME room as a REAL distinct server-side
    # identity — instead of reading a share link out of the app UI at runtime.
    # These links are minted by the DeepDiver platform (Share dialog / Platform
    # DB ``viewer_links``); the harness only consumes them. Empty when the query
    # is single-session or the links have not been provided yet.
    session_urls: dict[str, str] = field(default_factory=dict)

    def required_viewer_roles(self) -> list[str]:
        """Distinct non-creator participant roles this query needs links for.

        Union of every multi-session claim's ``participants`` minus the
        ``creator`` identity (which the main CUA session already embodies via
        the app URL). Order-preserving so ``viewer_a`` sorts before
        ``viewer_b``. Empty for single-session queries.
        """
        seen: list[str] = []
        for c in self.eval_claims:
            if not getattr(c, "multi_session", False):
                continue
            for p in getattr(c, "participants", []) or []:
                role = str(p).strip()
                if not role or role.lower() == "creator":
                    continue
                if role not in seen:
                    seen.append(role)
        return seen

    def missing_viewer_roles(self) -> list[str]:
        """Required viewer roles that lack a non-empty pre-minted link."""
        return [
            role
            for role in self.required_viewer_roles()
            if not (self.session_urls.get(role) or "").strip()
        ]
