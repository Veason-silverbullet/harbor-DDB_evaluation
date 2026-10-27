"""
Tool layer for the custom CUA.

Contains:
- schemas.py   : function-calling schemas + OpenAI/Anthropic format converters
- executor.py  : core browser actions (click/scroll/type/zoom/...)
- inspector.py : readonly helpers (page info, a11y tree, DOM query)
- native_adapter.py : optional adapter for Anthropic computer_20251124
"""
from .schemas import (
    Tool,
    ToolRegistry,
    build_default_registry,
    CORE_TOOL_NAMES,
    READONLY_TOOL_NAMES,
    REPORT_VERDICT_TOOL_NAME,
)
from .executor import ActionExecutor
from .inspector import PageInspector

__all__ = [
    "Tool",
    "ToolRegistry",
    "build_default_registry",
    "CORE_TOOL_NAMES",
    "READONLY_TOOL_NAMES",
    "REPORT_VERDICT_TOOL_NAME",
    "ActionExecutor",
    "PageInspector",
]
