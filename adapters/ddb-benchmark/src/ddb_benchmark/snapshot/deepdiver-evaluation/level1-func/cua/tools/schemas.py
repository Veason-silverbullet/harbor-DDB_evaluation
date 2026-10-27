"""
Function-calling schemas for the custom CUA tool set.

Design rules (aligned with computer-use-2025-11-24 capabilities, names kept local):
- one function = one tool (multi-tool function calling);
- coordinates are integers within the viewport;
- modifiers accepted on click / scroll as ["shift", "ctrl", "alt", "meta"];
- `report_verdict` is a sentinel tool the model calls exactly once to end the loop.

Converters produce the concrete wire format for OpenAI Chat Completions and
Anthropic Messages API (without the computer-use beta).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class Tool:
    """A single function-calling tool definition with its handler."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (object)
    handler: ToolHandler | None = None
    kind: str = "action"  # "action" | "readonly" | "control"


# ── Constants ───────────────────────────────────────────────────

REPORT_VERDICT_TOOL_NAME = "report_verdict"

# Multi-session collaboration tool (0702). Injected ONLY for claims flagged
# ``multi_session: true`` (and only when the feature is enabled in config), so
# single-session claims never see it and their cost/behaviour is unchanged.
DELEGATE_TO_VIEWER_TOOL_NAME = "delegate_to_viewer"
MULTI_SESSION_TOOL_NAMES: tuple[str, ...] = (DELEGATE_TO_VIEWER_TOOL_NAME,)

# Veracity / ground-truth tool. Injected ONLY for data-bearing claims (those
# that display real-world data) when ``enable_veracity_check`` is on. Lets the
# CUA look up the current real-world truth and compare it against what the app
# shows, so a stale/mock value (e.g. a fixed "South Africa vs Mexico" fixture)
# is caught instead of being scored as merely present.
SEARCH_TOOL_NAME = "search_web"
VERACITY_TOOL_NAMES: tuple[str, ...] = (SEARCH_TOOL_NAME,)

CORE_TOOL_NAMES: tuple[str, ...] = (
    "click",
    "mouse_down",
    "mouse_up",
    "move",
    "drag",
    "scroll",
    "type_text",
    "press_key",
    "hold_key",
    "wait",
    "advance_time",
    "zoom",
    "reload",
    "upload_file",
)

READONLY_TOOL_NAMES: tuple[str, ...] = (
    "get_page_info",
    "get_accessibility_tree",
)


# ── Parameter schemas ───────────────────────────────────────────

_COORD_X = {"type": "integer", "description": "x pixel, 0..viewport_width-1"}
_COORD_Y = {"type": "integer", "description": "y pixel, 0..viewport_height-1"}
_BUTTON = {
    "type": "string",
    "enum": ["left", "right", "middle"],
    "default": "left",
}
_MODIFIERS = {
    "type": "array",
    "items": {"type": "string", "enum": ["shift", "ctrl", "alt", "meta"]},
    "description": "Optional modifier keys held while performing the action.",
    "default": [],
}

_CLICK_PARAMS = {
    "type": "object",
    "properties": {
        "x": _COORD_X,
        "y": _COORD_Y,
        "button": _BUTTON,
        "clicks": {
            "type": "integer",
            "minimum": 1,
            "maximum": 3,
            "default": 1,
            "description": "1=single, 2=double, 3=triple click.",
        },
        "modifiers": _MODIFIERS,
    },
    "required": ["x", "y"],
    "additionalProperties": False,
}

_MOUSE_UP_DOWN_PARAMS = {
    "type": "object",
    "properties": {"x": _COORD_X, "y": _COORD_Y, "button": _BUTTON},
    "required": ["x", "y"],
    "additionalProperties": False,
}

_MOVE_PARAMS = {
    "type": "object",
    "properties": {"x": _COORD_X, "y": _COORD_Y},
    "required": ["x", "y"],
    "additionalProperties": False,
}

_DRAG_PARAMS = {
    "type": "object",
    "properties": {
        "start_x": _COORD_X,
        "start_y": _COORD_Y,
        "end_x": _COORD_X,
        "end_y": _COORD_Y,
        "button": _BUTTON,
    },
    "required": ["start_x", "start_y", "end_x", "end_y"],
    "additionalProperties": False,
}

_SCROLL_PARAMS = {
    "type": "object",
    "properties": {
        "x": _COORD_X,
        "y": _COORD_Y,
        "direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right"],
        },
        "amount": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "default": 3,
            "description": "Number of scroll clicks. ~100 pixels each.",
        },
        "modifiers": _MODIFIERS,
    },
    "required": ["x", "y", "direction"],
    "additionalProperties": False,
}

_TYPE_TEXT_PARAMS = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}

_PRESS_KEY_PARAMS = {
    "type": "object",
    "properties": {
        "key": {
            "type": "string",
            "description": "Playwright key name, supports combos like 'Ctrl+S'.",
        }
    },
    "required": ["key"],
    "additionalProperties": False,
}

_HOLD_KEY_PARAMS = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "duration_ms": {"type": "integer", "minimum": 50, "maximum": 10000},
    },
    "required": ["key", "duration_ms"],
    "additionalProperties": False,
}

_WAIT_PARAMS = {
    "type": "object",
    "properties": {
        "duration_ms": {
            "type": "integer",
            "minimum": 50,
            "maximum": 10000,
            "default": 500,
        }
    },
    "required": ["duration_ms"],
    "additionalProperties": False,
}

_ADVANCE_TIME_PARAMS = {
    "type": "object",
    "properties": {
        "minutes": {
            "type": "number",
            "minimum": 0,
            "maximum": 1440,
            "description": (
                "Virtual minutes to fast-forward. Use this to skip a long "
                "countdown/timer/cooldown to its end without waiting real "
                "time (e.g. minutes=25 to complete a 25-minute focus timer). "
                "Either `minutes` or `seconds` (or both) may be given."
            ),
        },
        "seconds": {
            "type": "number",
            "minimum": 0,
            "maximum": 86400,
            "description": "Additional virtual seconds to fast-forward.",
        },
    },
    "additionalProperties": False,
}

_ZOOM_PARAMS = {
    "type": "object",
    "properties": {
        "region": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x1, y1, x2, y2] top-left to bottom-right of the region to inspect.",
        }
    },
    "required": ["region"],
    "additionalProperties": False,
}

_RELOAD_PARAMS = {
    "type": "object",
    "properties": {
        "retries": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
            "default": 1,
            "description": (
                "How many additional attempts after the first one if the "
                "first reload throws (transient network errors). The "
                "orchestrator runs its own outer retry loop on top of this."
            ),
        }
    },
    "additionalProperties": False,
}

_UPLOAD_FILE_PARAMS = {
    "type": "object",
    "properties": {
        "x": _COORD_X,
        "y": _COORD_Y,
        "fixture": {
            "type": "string",
            "description": (
                "Optional name of the registered fixture file to attach. "
                "Omit to attach all available fixtures."
            ),
        },
    },
    "required": ["x", "y"],
    "additionalProperties": False,
}

_GET_PAGE_INFO_PARAMS = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_GET_A11Y_PARAMS = {
    "type": "object",
    "properties": {
        "max_nodes": {
            "type": "integer",
            "minimum": 10,
            "maximum": 500,
            "default": 150,
        }
    },
    "additionalProperties": False,
}

_DELEGATE_TO_VIEWER_PARAMS = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": (
                "What the SECOND participant (viewer B, a different person) "
                "should do in the app, in one imperative sentence. Examples: "
                "'send a chat comment that contains the marker', 'open your "
                "own settings page and report whether you can see the marker', "
                "'cast one vote for either team'. Do NOT tell them the "
                "expected result. You do NOT provide any link — the harness "
                "opens viewer B on their own pre-provisioned identity link "
                "automatically."
            ),
        },
    },
    "required": ["instruction"],
    "additionalProperties": False,
}

_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "A concise web-search query for the CURRENT real-world fact "
                "you need to check the page against, e.g. \"2026 World Cup "
                "fixtures today\", \"current USD to CNY exchange rate\", "
                "\"weather in Tokyo now\". Search the ground truth, do NOT "
                "search for the app itself."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

_REPORT_VERDICT_PARAMS = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [
                "supported",
                "contradicted",
                "insufficient_evidence",
                "not_applicable",
            ],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "evidence_sufficient": {"type": "boolean", "default": False},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reasoning"],
    "additionalProperties": False,
}


# ── Tool descriptions ───────────────────────────────────────────

_TOOL_DESCRIPTIONS: dict[str, tuple[str, dict[str, Any], str]] = {
    "click": (
        "Click at (x, y). Supports left/right/middle button, single/double/triple clicks, "
        "and modifier keys like shift/ctrl/alt/meta.",
        _CLICK_PARAMS,
        "action",
    ),
    "mouse_down": (
        "Press a mouse button down at (x, y) without releasing. Pair with mouse_up.",
        _MOUSE_UP_DOWN_PARAMS,
        "action",
    ),
    "mouse_up": (
        "Release a previously pressed mouse button at (x, y).",
        _MOUSE_UP_DOWN_PARAMS,
        "action",
    ),
    "move": (
        "Move the cursor to (x, y) without clicking.",
        _MOVE_PARAMS,
        "action",
    ),
    "drag": (
        "Press at (start_x, start_y), move to (end_x, end_y), then release.",
        _DRAG_PARAMS,
        "action",
    ),
    "scroll": (
        "Scroll at (x, y) by `amount` clicks in the given direction. "
        "Optionally hold modifier keys (e.g. shift for horizontal).",
        _SCROLL_PARAMS,
        "action",
    ),
    "type_text": (
        "Type `text` into the currently focused element.",
        _TYPE_TEXT_PARAMS,
        "action",
    ),
    "press_key": (
        "Press a key or key combo, e.g. 'Enter', 'Ctrl+S', 'ArrowDown'.",
        _PRESS_KEY_PARAMS,
        "action",
    ),
    "hold_key": (
        "Hold a key down for `duration_ms` milliseconds.",
        _HOLD_KEY_PARAMS,
        "action",
    ),
    "wait": (
        "Wait for `duration_ms` milliseconds. Use when the UI needs time to settle.",
        _WAIT_PARAMS,
        "action",
    ),
    "advance_time": (
        "Fast-forward the page's virtual clock by `minutes`/`seconds` so a "
        "long countdown, timer, or cooldown reaches its end INSTANTLY "
        "without waiting real time. All in-page timers (setInterval / "
        "setTimeout / requestAnimationFrame) and Date/now reads jump "
        "forward together, so the app renders its real end-state (e.g. "
        "00:00, completion popup, round counter +1). ONLY use after you have "
        "STARTED the timer; prefer this over stacking many `wait` calls for "
        "anything longer than ~15 seconds.",
        _ADVANCE_TIME_PARAMS,
        "action",
    ),
    "zoom": (
        "Return a cropped, full-resolution sub-image of the current screenshot for "
        "the given region [x1,y1,x2,y2]. Useful to inspect small UI details.",
        _ZOOM_PARAMS,
        "action",
    ),
    "reload": (
        "Reload the current page. Use ONLY when the page looks blank, the "
        "viewport is white/empty, the last screenshot has not changed for "
        "several steps, or you suspect a transient network failure prevented "
        "the SPA from booting. Do NOT call this as a general 'restart' — it "
        "discards any in-memory state on the page.",
        _RELOAD_PARAMS,
        "action",
    ),
    "upload_file": (
        "Attach a file to a file-upload control. Use ONLY when the task "
        "requires uploading/importing a file AND a visible upload button / "
        "drop zone / file input exists. Provide the (x, y) of that upload "
        "control; the harness supplies the file content from a pre-registered "
        "fixture (you cannot pick arbitrary local files). Optionally name a "
        "specific `fixture`. If no fixture is registered the call fails and "
        "you should treat the upload as a blocked prerequisite.",
        _UPLOAD_FILE_PARAMS,
        "action",
    ),
    "get_page_info": (
        "Readonly. Return current URL, title, and a truncated visible_text snapshot.",
        _GET_PAGE_INFO_PARAMS,
        "readonly",
    ),
    "get_accessibility_tree": (
        "Readonly. Return a flattened accessibility tree (role/name/bbox/enabled). "
        "Use only when pixels are not enough to identify a target.",
        _GET_A11Y_PARAMS,
        "readonly",
    ),
    DELEGATE_TO_VIEWER_TOOL_NAME: (
        "Spawn a SECOND, independent participant (viewer B) in a brand-new "
        "isolated session (no shared login/cookies with you) to perform ONE "
        "action, then get their observation back. Use this — instead of "
        "pretending to be two people yourself — for cross-identity claims: "
        "realtime sync (let B write content bearing the unique marker, then "
        "YOU watch your own session for it), shared read/write, online-viewer "
        "counts, or per-identity isolation (you write private marked data, "
        "then ask B whether THEY can see your marker). The viewer runs, is "
        "closed automatically afterwards, and returns {success, summary, "
        "observation, marker}. You still decide the final verdict from what "
        "YOUR own session shows.",
        _DELEGATE_TO_VIEWER_PARAMS,
        "action",
    ),
    SEARCH_TOOL_NAME: (
        "Look up the CURRENT real-world truth on the web to check whether the "
        "real-world data the app displays (sports fixtures/scores, exchange "
        "rates, weather, news, prices, schedules, etc.) is actually correct "
        "and up to date — not stale, fabricated, or mock. Returns the top "
        "search results as text. Use it when the claim is about real-world "
        "data: search the ground truth, then compare it to what the page "
        "shows. If they clearly disagree the claim is contradicted; if search "
        "is unavailable or inconclusive, do NOT fail the claim on that basis.",
        _SEARCH_PARAMS,
        "action",
    ),
    REPORT_VERDICT_TOOL_NAME: (
        "Call exactly once when the claim investigation is complete. "
        "Provides the final verdict, confidence (0..1), and reasoning.",
        _REPORT_VERDICT_PARAMS,
        "control",
    ),
}


def anthropic_tool_schema(name: str) -> dict[str, Any]:
    """Single tool in Anthropic Messages ``tools=[...]`` shape
    (``{name, description, input_schema}``). Used by the native/FC Anthropic
    adapters to hang ``delegate_to_viewer`` / ``search_web`` alongside their
    hand-authored ``computer`` tool.
    """
    desc, params, _kind = _TOOL_DESCRIPTIONS[name]
    return {"name": name, "description": desc, "input_schema": params}


def responses_function_tool_schema(name: str) -> dict[str, Any]:
    """Single tool in OpenAI **Responses** flat function shape. Used by the
    native OpenAI adapter to hang ``delegate_to_viewer`` / ``search_web``
    alongside the GA ``computer`` tool.
    """
    desc, params, _kind = _TOOL_DESCRIPTIONS[name]
    return {
        "type": "function",
        "name": name,
        "description": desc,
        "parameters": params,
        "strict": False,
    }


# ── Registry ────────────────────────────────────────────────────

@dataclass
class ToolRegistry:
    """A collection of Tools keyed by name, with format converters."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def names(self) -> list[str]:
        return list(self.tools.keys())

    def visible(self, include_readonly: bool = False) -> list[Tool]:
        return [
            t for t in self.tools.values()
            if t.kind != "readonly" or include_readonly
        ]

    def to_openai_format(self, include_readonly: bool = False) -> list[dict[str, Any]]:
        """Return [{"type": "function", "function": {name, description, parameters}}, ...]."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.visible(include_readonly=include_readonly)
        ]

    def to_anthropic_format(self, include_readonly: bool = False) -> list[dict[str, Any]]:
        """Return [{name, description, input_schema}, ...] for Anthropic tools=[...]."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in self.visible(include_readonly=include_readonly)
        ]


def build_default_registry(
    include_multi_session: bool = False,
    include_veracity: bool = False,
) -> ToolRegistry:
    """Build a registry with all schema metadata but no handlers yet.

    Handlers are bound in CUARunner at construction time (so they can capture
    the concrete ActionExecutor / PageInspector instances).

    ``include_multi_session`` (default False) gates the collaboration tool
    (``delegate_to_viewer``): it is only registered when the feature is
    enabled, and per-claim visibility is further narrowed by the runner so
    single-session claims never receive it.

    ``include_veracity`` (default False) gates the ground-truth tool
    (``search_web``): registered only when veracity checking is enabled, and
    the runner further restricts it to data-bearing claims per-claim.
    """
    reg = ToolRegistry()
    for name, (desc, params, kind) in _TOOL_DESCRIPTIONS.items():
        if name in MULTI_SESSION_TOOL_NAMES and not include_multi_session:
            continue
        if name in VERACITY_TOOL_NAMES and not include_veracity:
            continue
        reg.register(Tool(name=name, description=desc, parameters=params, kind=kind))
    return reg
