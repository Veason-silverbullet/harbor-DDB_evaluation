"""
Backward-compat shim.

The real tool-oriented ActionExecutor now lives in ``cua.tools.executor``.
This module keeps the old import path working for legacy call sites that
use ``ActionExecutor.execute({"action": "click", "coordinate": [...]})``.

Prefer importing from ``cua.tools`` in new code.
"""
from __future__ import annotations
from .tools.executor import ActionExecutor  # re-export

__all__ = ["ActionExecutor"]
