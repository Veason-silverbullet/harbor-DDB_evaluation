"""
Evidence data structures collected by the CUA during verification.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionTrace:
    """Single action performed by the CUA."""

    step: int
    action_type: str  # click, type, scroll, key, drag, screenshot, wait
    params: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    screenshot_path: str | None = None


@dataclass
class DOMSignals:
    """Programmatic signals extracted from the page DOM."""

    page_url: str = ""
    page_title: str = ""
    visible_text: str = ""
    visible_text_raw_len: int = 0
    element_counts: dict[str, int] = field(default_factory=dict)
    url_changed: bool = False
    title_changed: bool = False
    text_changed: bool = False


@dataclass
class RuntimeSignals:
    """Runtime browser signals captured during a single claim verification.

    These are populated by ``PlaywrightEnv``'s console / pageerror /
    requestfailed / response listeners and are critical for claims of
    type ``data_integration`` (was the real API actually called?),
    ``runtime_health`` (any console / 4xx / 5xx errors?) and
    ``interaction_accessibility`` (did the click actually hit the
    expected element?).
    """

    console_errors: list[dict[str, Any]] = field(default_factory=list)
    failed_requests: list[dict[str, Any]] = field(default_factory=list)
    external_hosts_called: list[dict[str, Any]] = field(default_factory=list)
    hit_test_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvidenceBundle:
    """All evidence collected for a single claim verification."""

    claim_id: str
    before_state: DOMSignals | None = None
    after_state: DOMSignals | None = None
    action_trace: list[ActionTrace] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)  # paths to screenshot files
    cua_reasoning: str = ""  # CUA's textual analysis
    evidence_sufficient: bool = False
    raw_cua_response: str = ""  # full CUA output for debugging
    runtime_signals: RuntimeSignals = field(default_factory=RuntimeSignals)
