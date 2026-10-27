"""
Readonly helpers exposed to the LLM as optional tools.

These are opt-in via ``cua.enable_readonly_tools``. When disabled the
CUARunner falls back to pure-pixel CUA.
"""
from __future__ import annotations
import logging
from typing import Any
from ..playwright_env import PlaywrightEnv

logger = logging.getLogger(__name__)


# Roles we consider actionable / informative enough to keep in the flat list.
_INTERESTING_ROLES = {
    "button", "link", "textbox", "searchbox", "combobox", "checkbox",
    "radio", "switch", "tab", "tablist", "menuitem", "menu", "listbox",
    "option", "slider", "spinbutton", "dialog", "heading", "image",
    "list", "listitem", "navigation", "form", "main", "banner",
    "contentinfo", "region",
}


class PageInspector:
    """Readonly tools: get_page_info / get_accessibility_tree."""

    def __init__(self, env: PlaywrightEnv) -> None:
        self._env = env

    async def get_page_info(self, args: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            info = await self._env.get_page_info()
            return {
                "success": True,
                "error": None,
                "text": (
                    f"url: {info.get('page_url', '')}\n"
                    f"title: {info.get('page_title', '')}\n"
                    f"visible_text_len: {info.get('visible_text_raw_len', 0)}\n"
                    f"visible_text (truncated):\n{info.get('visible_text', '')}"
                ),
                "data": info,
                "image_b64": None,
                "clamped": False,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "text": None,
                "data": None,
                "image_b64": None,
                "clamped": False,
            }

    async def get_accessibility_tree(self, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        max_nodes = int(args.get("max_nodes", 150) or 150)
        snap = await self._env.accessibility_snapshot(interesting_only=True)
        if snap is None:
            return {
                "success": False,
                "error": "accessibility snapshot unavailable",
                "text": None,
                "data": None,
                "image_b64": None,
                "clamped": False,
            }
        flat = _flatten_a11y(snap, max_nodes=max_nodes)
        lines = [
            f"[{i}] role={n.get('role')} name={n.get('name')!r}"
            + (f" value={n.get('value')!r}" if n.get("value") else "")
            + (" disabled" if n.get("disabled") else "")
            for i, n in enumerate(flat)
        ]
        return {
            "success": True,
            "error": None,
            "text": "\n".join(lines) if lines else "(no interesting a11y nodes)",
            "data": flat,
            "image_b64": None,
            "clamped": False,
        }


def _flatten_a11y(node: dict[str, Any], max_nodes: int) -> list[dict[str, Any]]:
    """Breadth-first flatten of the a11y tree to at most max_nodes interesting nodes."""
    out: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = [node]
    while queue and len(out) < max_nodes:
        cur = queue.pop(0)
        role = str(cur.get("role", ""))
        name = cur.get("name", "")
        if role in _INTERESTING_ROLES or name:
            entry: dict[str, Any] = {"role": role, "name": name}
            val = cur.get("value")
            if val:
                entry["value"] = val
            if cur.get("disabled"):
                entry["disabled"] = True
            if cur.get("checked") is not None:
                entry["checked"] = cur.get("checked")
            out.append(entry)
        for child in cur.get("children", []) or []:
            queue.append(child)
    return out
