"""
Programmatic validators that check claims without needing LLM.

These run fast, deterministic checks on DOM signals, text content,
URL changes, etc. They provide a first-pass verdict that may be
sufficient for simple claims or serve as evidence for the LLM judge.
"""
from __future__ import annotations
import math
import re as _re
from typing import Any


# Keywords that indicate a claim is about map geometry (route polyline,
# check-in markers, etc.). Used to opt-into ``validate_geo_semantics``
# from the visual_semantics path.
_MAP_KEYWORDS = (
    "地图", "map ", "map.", " map", "leaflet", "mapbox", "amap", "高德",
)
_MAP_ROUTE_KEYWORDS = (
    "路线", "连线", "polyline", "路径", "经纬度", "经度", "纬度",
    "lat", "lng", "lat/lng", "实际地理", "大圆", "弧线",
)
_MAP_MARKER_KEYWORDS = (
    "打卡", "标点", "marker", "图钉", "地点", "城市坐标", "poi",
)


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _tokens_from_claim(checkpoint: str, verification: str) -> list[str]:
    """Extract key tokens from claim text for matching."""
    combined = f"{checkpoint} {verification}"
    tokens = []
    for seg in combined.replace("，", " ").replace("、", " ").replace("；", " ").split():
        seg = seg.strip()
        if len(seg) >= 2:
            tokens.append(seg)
    return tokens


def _token_coverage(tokens: list[str], text: str) -> tuple[float, list[str]]:
    """Check how many tokens appear in the text."""
    if not tokens:
        return 0.0, []
    hay = _norm(text)
    matched = [t for t in tokens if _norm(t) in hay]
    return len(matched) / len(tokens), matched


def validate_structural(
    page_info: dict[str, Any],
    checkpoint: str,
    verification: str,
) -> dict[str, Any]:
    """Check if claimed UI elements / text / count exist on the page.

    Axis-1 method ``structural`` (formerly ``presence``).
    """
    tokens = _tokens_from_claim(checkpoint, verification)
    visible = page_info.get("visible_text", "")
    title = page_info.get("page_title", "")
    url = page_info.get("page_url", "")
    combined = f"{title} {url} {visible}"

    coverage, matched = _token_coverage(tokens, combined)

    if coverage >= 0.5:
        return {
            "verdict": "supported",
            "confidence": min(0.9, 0.55 + coverage * 0.35),
            "reason": f"Token match {len(matched)}/{len(tokens)}: {matched[:5]}",
            "evidence_sufficient": True,
            "needs_llm": False,
        }
    elif coverage == 0.0 and tokens:
        return {
            "verdict": "contradicted",
            "confidence": 0.6,
            "reason": f"No key tokens found (0/{len(tokens)})",
            "evidence_sufficient": True,
            "needs_llm": True,
        }
    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.35,
        "reason": f"Partial token match {len(matched)}/{len(tokens)}",
        "evidence_sufficient": False,
        "needs_llm": True,
    }


# Back-compat alias: legacy callers / tests may still import ``validate_presence``.
validate_presence = validate_structural


def validate_navigation(
    before_info: dict[str, Any],
    after_info: dict[str, Any],
    action_trace: list[dict],
) -> dict[str, Any]:
    """Check if navigation occurred (URL/title change after actions)."""
    url_changed = _norm(before_info.get("page_url")) != _norm(after_info.get("page_url"))
    title_changed = _norm(before_info.get("page_title")) != _norm(after_info.get("page_title"))
    ok_actions = sum(1 for a in action_trace if a.get("success", False))

    if url_changed or title_changed:
        return {
            "verdict": "supported",
            "confidence": 0.78,
            "reason": f"Page changed (url_changed={url_changed}, title_changed={title_changed})",
            "evidence_sufficient": True,
            "needs_llm": False,
        }
    elif ok_actions > 0:
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.45,
            "reason": "Actions executed but no URL/title change detected",
            "evidence_sufficient": False,
            "needs_llm": True,
        }
    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.35,
        "reason": "No successful actions and no page change",
        "evidence_sufficient": False,
        "needs_llm": True,
    }


# v2: action_effect 短路阈值。旧实现只要 `text_changed=True` 或者
# checkpoint+verification 的 token 在 after_text 里的 coverage ≥ 0.4
# 就敢直接发 ``supported``。问题是「点击爱心 → 心愿清单空状态」这种
# 场景里，点击成功 + 跳转新页 → text 必然变 → 走捷径打 supported，
# 完全无视 verification 期望的具体效果。
#
# v2 改为：要发 supported，必须满足
#   (a) verification 自身的关键 token，在 after_text 中的 coverage ≥
#       _ACTION_EFFECT_VERIFICATION_COVERAGE_MIN (0.5)，且
#   (b) 至少一个真实交互动作成功。
# 否则一律标 ``needs_llm=True`` 退回 LLM judge，让它配合截图做实判。
_ACTION_EFFECT_VERIFICATION_COVERAGE_MIN = 0.5


def _verification_tokens(verification: str) -> list[str]:
    """Tokens specifically from the *verification* text only.

    The previous helper concatenated ``checkpoint`` and ``verification``,
    which caused trivial subject tokens (e.g. "搜索框", "按钮") to dominate
    the coverage match — they appear on almost every post-action page.
    Restricting to verification keeps the focus on the claimed *outcome*.
    """
    out: list[str] = []
    for seg in (verification or "").replace("，", " ").replace("、", " ").replace("；", " ").split():
        seg = seg.strip()
        if len(seg) >= 2:
            out.append(seg)
    return out


def validate_state_transition(
    before_info: dict[str, Any],
    after_info: dict[str, Any],
    action_trace: list[dict],
    checkpoint: str,
    verification: str,
) -> dict[str, Any]:
    """Check if an action produced observable state change.

    Axis-1 method ``state_transition`` (formerly ``action_effect``).

    v2 (binary-rubric) tightening: a supported short-circuit now requires
    *outcome-specific* token coverage in the post-action page text, not
    merely "the page changed somehow". Anything weaker hands off to the
    LLM judge instead of fabricating a half-confidence supported result.
    """
    text_changed = _norm(before_info.get("visible_text")) != _norm(after_info.get("visible_text"))
    ok_actions = sum(1 for a in action_trace if a.get("success", False))

    after_text = after_info.get("visible_text", "")

    # Verification-only coverage drives the supported short-circuit.
    v_tokens = _verification_tokens(verification)
    v_coverage, v_matched = _token_coverage(v_tokens, after_text)

    # Combined coverage stays around for the partial / failure branches
    # so we can describe what *did* land in the reason.
    combined_tokens = _tokens_from_claim(checkpoint, verification)
    combined_coverage, _ = _token_coverage(combined_tokens, after_text)

    if (
        ok_actions > 0
        and v_tokens
        and v_coverage >= _ACTION_EFFECT_VERIFICATION_COVERAGE_MIN
    ):
        return {
            "verdict": "supported",
            "confidence": 0.72,
            "reason": (
                f"Action caused state change matching verification "
                f"({len(v_matched)}/{len(v_tokens)} verification tokens "
                f"present in post-action page)"
            ),
            "evidence_sufficient": True,
            "needs_llm": False,
        }

    if ok_actions > 0 and (text_changed or combined_coverage >= 0.4):
        # Page changed but the verification-specific signature didn't
        # fully appear. Don't shortcut — let the LLM judge inspect the
        # screenshots to decide between supported / contradicted /
        # insufficient_evidence (no more partial in v2).
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.45,
            "reason": (
                "Action executed and page changed, but verification "
                f"tokens only matched {int(v_coverage * 100)}% in post-action "
                "text; defer to LLM judge"
            ),
            "evidence_sufficient": False,
            "needs_llm": True,
        }

    if ok_actions > 0:
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.35,
            "reason": "Action executed but produced no observable state change",
            "evidence_sufficient": False,
            "needs_llm": True,
        }

    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.35,
        "reason": "Could not execute required actions",
        "evidence_sufficient": False,
        "needs_llm": True,
    }


# Back-compat alias: legacy callers / tests may still import ``validate_action_effect``.
validate_action_effect = validate_state_transition


# ── Geo / map semantics ────────────────────────────────────────────


def is_map_geo_claim(checkpoint: str, verification: str) -> bool:
    """True if a visual_semantics claim is about map polyline / markers."""
    text = f"{checkpoint} {verification}".lower()
    has_map = any(kw in text for kw in _MAP_KEYWORDS)
    has_route = any(kw in text for kw in _MAP_ROUTE_KEYWORDS)
    has_marker = any(kw in text for kw in _MAP_MARKER_KEYWORDS)
    return has_map and (has_route or has_marker)


def _geo_intent(checkpoint: str, verification: str) -> str:
    """Whether the claim is route-shaped or marker-shaped (or both)."""
    text = f"{checkpoint} {verification}".lower()
    is_route = any(kw in text for kw in _MAP_ROUTE_KEYWORDS)
    is_marker = any(kw in text for kw in _MAP_MARKER_KEYWORDS)
    if is_route and is_marker:
        return "both"
    if is_route:
        return "route"
    if is_marker:
        return "marker"
    return "unknown"


def _is_straight_polyline(poly: dict[str, Any]) -> bool:
    """Detect if an SVG polyline is essentially a straight line.

    A polyline is "straight" if (a) it has only 2 vertices, or (b) all
    vertices are nearly collinear (max perpendicular deviation from the
    chord is below a small threshold relative to the chord length).
    Real geographic routes between cities almost always produce dozens
    of intermediate vertices and a non-trivial deviation.
    """
    pts = poly.get("screen_points") or []
    vc = int(poly.get("vertex_count", len(pts)) or 0)
    if vc <= 2:
        return True
    if len(pts) < 3:
        return vc <= 2
    x0 = float(pts[0]["x"])
    y0 = float(pts[0]["y"])
    x1 = float(pts[-1]["x"])
    y1 = float(pts[-1]["y"])
    dx = x1 - x0
    dy = y1 - y0
    chord = math.hypot(dx, dy)
    if chord < 1.0:
        return True
    max_dev = 0.0
    for pt in pts[1:-1]:
        px = float(pt["x"])
        py = float(pt["y"])
        dev = abs(dy * px - dx * py + x1 * y0 - y1 * x0) / chord
        if dev > max_dev:
            max_dev = dev
    return (max_dev / chord) < 0.01


def validate_geo_semantics(
    geometry: dict[str, Any] | None,
    checkpoint: str,
    verification: str,
) -> dict[str, Any]:
    """Programmatic check for map polyline / marker semantics.

    Returns the standard ``{verdict, confidence, reason,
    evidence_sufficient, needs_llm}`` shape. ``needs_llm=True`` means
    the LLM judge should still take another pass (e.g. to confirm
    map content matches geographic semantics beyond pure geometry).
    """
    geo = geometry or {}
    polylines: list[dict[str, Any]] = list(geo.get("polylines") or [])
    markers: list[dict[str, Any]] = list(geo.get("markers") or [])
    container_count = int(geo.get("map_containers", 0) or 0)

    intent = _geo_intent(checkpoint, verification)

    if container_count == 0 and not polylines and not markers:
        return {
            "verdict": "contradicted",
            "confidence": 0.7,
            "reason": (
                "No map container / polyline / marker found in DOM; "
                "claim references map but page has none."
            ),
            "evidence_sufficient": True,
            "needs_llm": True,
            "geometry_summary": {
                "container_count": container_count,
                "polyline_count": 0,
                "marker_count": 0,
            },
        }

    if intent in ("route", "both") and polylines:
        sig_polys = [
            p for p in polylines
            if max(float(p.get("span_x", 0) or 0), float(p.get("span_y", 0) or 0)) >= 30.0
        ]
        candidates = sig_polys or polylines
        straight_count = sum(1 for p in candidates if _is_straight_polyline(p))
        all_straight = bool(candidates) and straight_count == len(candidates)
        max_vertices = max((int(p.get("vertex_count", 0) or 0) for p in candidates), default=0)

        if all_straight and max_vertices <= 2:
            return {
                "verdict": "contradicted",
                "confidence": 0.85,
                "reason": (
                    f"All {len(candidates)} significant polylines on the map have "
                    f"<=2 vertices (max={max_vertices}); they are pure straight "
                    "segments rather than lat/lng-following routes."
                ),
                "evidence_sufficient": True,
                "needs_llm": False,
                "geometry_summary": {
                    "container_count": container_count,
                    "polyline_count": len(polylines),
                    "marker_count": len(markers),
                    "max_vertices": max_vertices,
                    "straight_polyline_count": straight_count,
                },
            }
        if all_straight and max_vertices >= 3:
            return {
                "verdict": "insufficient_evidence",
                "confidence": 0.45,
                "reason": (
                    f"Polylines exist with up to {max_vertices} vertices but they "
                    "are nearly collinear; route may still be a straight line."
                ),
                "evidence_sufficient": False,
                "needs_llm": True,
                "geometry_summary": {
                    "container_count": container_count,
                    "polyline_count": len(polylines),
                    "marker_count": len(markers),
                    "max_vertices": max_vertices,
                    "straight_polyline_count": straight_count,
                },
            }
        if max_vertices >= 8 and not all_straight:
            return {
                "verdict": "supported",
                "confidence": 0.72,
                "reason": (
                    f"Polyline has {max_vertices} vertices with non-trivial "
                    "curvature, consistent with a real geographic route."
                ),
                "evidence_sufficient": True,
                "needs_llm": True,
                "geometry_summary": {
                    "container_count": container_count,
                    "polyline_count": len(polylines),
                    "marker_count": len(markers),
                    "max_vertices": max_vertices,
                    "straight_polyline_count": straight_count,
                },
            }

    if intent in ("marker", "both") and markers:
        xs = [float(m.get("x", 0) or 0) for m in markers]
        ys = [float(m.get("y", 0) or 0) for m in markers]
        span_x = (max(xs) - min(xs)) if xs else 0.0
        span_y = (max(ys) - min(ys)) if ys else 0.0
        if len(markers) >= 2 and (span_x >= 50.0 or span_y >= 50.0):
            return {
                "verdict": "supported",
                "confidence": 0.65,
                "reason": (
                    f"Found {len(markers)} markers spread over "
                    f"{int(span_x)}x{int(span_y)} px, consistent with multiple "
                    "geographic locations."
                ),
                "evidence_sufficient": True,
                "needs_llm": True,
                "geometry_summary": {
                    "container_count": container_count,
                    "polyline_count": len(polylines),
                    "marker_count": len(markers),
                    "marker_span_x": int(span_x),
                    "marker_span_y": int(span_y),
                },
            }
        if len(markers) >= 2 and span_x < 5.0 and span_y < 5.0:
            return {
                "verdict": "contradicted",
                "confidence": 0.75,
                "reason": (
                    f"All {len(markers)} markers cluster within a few pixels "
                    "(span<5px); positions do not reflect real coordinates."
                ),
                "evidence_sufficient": True,
                "needs_llm": True,
                "geometry_summary": {
                    "container_count": container_count,
                    "polyline_count": len(polylines),
                    "marker_count": len(markers),
                    "marker_span_x": int(span_x),
                    "marker_span_y": int(span_y),
                },
            }

    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.4,
        "reason": (
            f"Map geometry collected (containers={container_count}, "
            f"polylines={len(polylines)}, markers={len(markers)}) but did not "
            "trigger any decisive geo rule; defer to LLM judge."
        ),
        "evidence_sufficient": False,
        "needs_llm": True,
        "geometry_summary": {
            "container_count": container_count,
            "polyline_count": len(polylines),
            "marker_count": len(markers),
        },
    }


# ── value_correctness ──────────────────────────────────────────────

# Number with optional thousands separators / decimals / sign / percent.
_NUMBER_RE = _re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
# ISO-ish dates (YYYY-MM-DD / YYYY/MM/DD) and Chinese 年月日.
_ISO_DATE_RE = _re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_CN_DATE_RE = _re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")

_DAYS_IN_MONTH = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

_DATE_CLAIM_KEYWORDS = ("日期", "date", "年月日", "月份", "deadline", "截止", "到期")
_INVALID_HINT_KEYWORDS = (
    "非法", "无效", "不合法", "invalid", "错误", "拒绝", "被拒", "不存在", "规范化",
    "归一", "normalize", "rejected", "error", "提示", "校验",
)
_SORT_KEYWORDS = ("排序", "从低到高", "从高到低", "升序", "降序", "sort", "ascending", "descending")
_COUNT_KEYWORDS = ("至少", "条", "个", "数量", "count", "最多", "共", "总计")


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _is_legal_date(year: int, month: int, day: int) -> bool:
    if not (1 <= month <= 12):
        return False
    max_day = _DAYS_IN_MONTH[month - 1]
    if month == 2 and not _is_leap(year):
        max_day = 28
    return 1 <= day <= max_day


def _parse_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUMBER_RE.findall(text or ""):
        raw = m.replace(",", "")
        pct = raw.endswith("%")
        if pct:
            raw = raw[:-1]
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _find_dates(text: str) -> list[tuple[int, int, int, str]]:
    """Return (year, month, day, raw) tuples found in ``text``."""
    found: list[tuple[int, int, int, str]] = []
    for m in _ISO_DATE_RE.finditer(text or ""):
        found.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0)))
    for m in _CN_DATE_RE.finditer(text or ""):
        found.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0)))
    return found


def _validate_date_legality(
    after_text: str, checkpoint: str, verification: str
) -> dict[str, Any] | None:
    """Date-legality sub-check for ``value_correctness``.

    Only engages when the claim text is about date legality / validation
    (keyword-gated). It asserts that:
      * any date string surfaced in the post-action page is a *legal*
        calendar date (rejects 2026-02-30, month 13, etc.), and
      * when the claim expects illegal input to be rejected / normalized,
        the page shows a rejection / normalization hint OR no illegal date
        survived.
    Returns the standard verdict dict, or ``None`` if not a date claim
    (so the caller can fall through to numeric/sort handling).
    """
    blob = f"{checkpoint} {verification}".lower()
    if not any(k.lower() in blob for k in _DATE_CLAIM_KEYWORDS):
        return None

    dates = _find_dates(after_text)
    illegal = [d for d in dates if not _is_legal_date(d[0], d[1], d[2])]
    expects_rejection = any(k.lower() in blob for k in _INVALID_HINT_KEYWORDS)
    has_reject_hint = any(k.lower() in (after_text or "").lower() for k in _INVALID_HINT_KEYWORDS)

    if illegal:
        # An illegal date is still displayed as-is -> the app accepted it.
        return {
            "verdict": "contradicted",
            "confidence": 0.8,
            "reason": (
                f"Illegal calendar date(s) present in page: "
                f"{[d[3] for d in illegal][:5]}; date legality not enforced."
            ),
            "evidence_sufficient": True,
            "needs_llm": False,
        }

    if expects_rejection:
        if has_reject_hint or dates:
            return {
                "verdict": "supported",
                "confidence": 0.7,
                "reason": (
                    "No illegal dates survived and/or a rejection/normalization "
                    "hint is present, consistent with date validation."
                ),
                "evidence_sufficient": True,
                "needs_llm": False,
            }
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.4,
            "reason": (
                "Claim is about date legality but no date / rejection hint was "
                "observed in the page; defer to LLM judge."
            ),
            "evidence_sufficient": False,
            "needs_llm": True,
        }

    if dates:
        return {
            "verdict": "supported",
            "confidence": 0.65,
            "reason": f"All {len(dates)} date(s) on the page are legal calendar dates.",
            "evidence_sufficient": True,
            "needs_llm": False,
        }
    return None


def validate_value_correctness(
    page_info: dict[str, Any],
    checkpoint: str,
    verification: str,
    *,
    semantic_type: str | None = None,
) -> dict[str, Any]:
    """Programmatic check for displayed-value / sort / count / date claims.

    Axis-1 method ``value_correctness`` (covers semantic_type content /
    computation). Only the *programmatically decidable* slices are handled
    here — monotonic sort order, count thresholds, and date legality. Any
    claim that needs semantic understanding of text returns ``needs_llm``.
    """
    after_text = page_info.get("visible_text", "") or ""
    blob = f"{checkpoint} {verification}"

    # 1) Date legality (含日期合法化判定).
    date_result = _validate_date_legality(after_text, checkpoint, verification)
    if date_result is not None:
        return date_result

    # 2) Sort / ordering monotonicity.
    if any(k.lower() in blob.lower() for k in _SORT_KEYWORDS):
        nums = _parse_numbers(after_text)
        if len(nums) >= 3:
            asc = all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))
            desc = all(nums[i] >= nums[i + 1] for i in range(len(nums) - 1))
            wants_desc = any(k in blob for k in ("从高到低", "降序", "descending"))
            wants_asc = any(k in blob for k in ("从低到高", "升序", "ascending"))
            if (wants_asc and asc) or (wants_desc and desc) or (not wants_asc and not wants_desc and (asc or desc)):
                return {
                    "verdict": "supported",
                    "confidence": 0.68,
                    "reason": f"Displayed numeric sequence is monotonic ({'asc' if asc else 'desc'}).",
                    "evidence_sufficient": True,
                    "needs_llm": False,
                }
            if (wants_asc and not asc) or (wants_desc and not desc):
                return {
                    "verdict": "contradicted",
                    "confidence": 0.62,
                    "reason": "Displayed numeric sequence is not monotonic in the claimed order.",
                    "evidence_sufficient": True,
                    "needs_llm": True,
                }

    # 3) Count / enumeration threshold ("至少 N 条").
    if any(k.lower() in blob.lower() for k in _COUNT_KEYWORDS):
        m = _re.search(r"(?:至少|最少|>=|≥)\s*(\d+)", blob)
        if m:
            threshold = int(m.group(1))
            # Heuristic item count: list-ish separators in the visible text.
            approx_items = max(
                after_text.count("\n"),
                len(_NUMBER_RE.findall(after_text)) // 2,
            )
            if approx_items >= threshold:
                return {
                    "verdict": "supported",
                    "confidence": 0.55,
                    "reason": f"Observed ~{approx_items} items >= claimed threshold {threshold}.",
                    "evidence_sufficient": True,
                    "needs_llm": True,
                }

    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.4,
        "reason": (
            "value_correctness claim not programmatically decidable from text "
            "(no parseable sort/count/date signature); defer to LLM judge."
        ),
        "evidence_sufficient": False,
        "needs_llm": True,
    }


# ── runtime_integrity ──────────────────────────────────────────────


def validate_runtime_integrity(
    runtime_signals: dict[str, Any],
    action_trace: list[dict],
    checkpoint: str,
    verification: str,
    *,
    semantic_type: str | None = None,
) -> dict[str, Any]:
    """Validate runtime signals, sub-dispatched by ``semantic_type``.

    Axis-1 method ``runtime_integrity`` consumes the already-collected
    ``RuntimeSignals``:
      * data_integration -> a claimed external host was actually called;
      * runtime_health   -> no console errors / no key 4xx-5xx requests;
      * accessibility    -> the relevant hit-test landed on the element.
    """
    rs = runtime_signals or {}
    st = (semantic_type or "").strip().lower()

    if st == "data_integration":
        hosts = rs.get("external_hosts_called") or []
        ext_hits = [h for h in hosts if bool(h.get("is_external")) and int(h.get("count", 0) or 0) >= 1]
        blob = f"{checkpoint} {verification}".lower()
        named = [h for h in ext_hits if str(h.get("host", "")).lower() in blob]
        if named:
            return {
                "verdict": "supported",
                "confidence": 0.8,
                "reason": f"Claimed external host(s) actually called: {[h.get('host') for h in named][:3]}.",
                "evidence_sufficient": True,
                "needs_llm": False,
            }
        if ext_hits:
            return {
                "verdict": "insufficient_evidence",
                "confidence": 0.45,
                "reason": (
                    f"External hosts were called {[h.get('host') for h in ext_hits][:3]} but none "
                    "matched the host named in the claim; defer to LLM judge."
                ),
                "evidence_sufficient": False,
                "needs_llm": True,
            }
        return {
            "verdict": "contradicted",
            "confidence": 0.62,
            "reason": "No external host was called; data likely mocked / hard-coded.",
            "evidence_sufficient": True,
            "needs_llm": True,
        }

    if st == "runtime_health":
        console_errors = rs.get("console_errors") or []
        failed = rs.get("failed_requests") or []
        key_failures = [
            f for f in failed
            if str(f.get("status", "")).startswith(("4", "5"))
            or f.get("kind") in ("requestfailed", "failed")
        ]
        if console_errors or key_failures:
            return {
                "verdict": "contradicted",
                "confidence": 0.75,
                "reason": (
                    f"Runtime errors detected: {len(console_errors)} console error(s), "
                    f"{len(key_failures)} failed/4xx-5xx request(s)."
                ),
                "evidence_sufficient": True,
                "needs_llm": False,
            }
        return {
            "verdict": "supported",
            "confidence": 0.7,
            "reason": "No console errors and no key 4xx/5xx requests during the main path.",
            "evidence_sufficient": True,
            "needs_llm": False,
        }

    if st == "accessibility":
        hits = rs.get("hit_test_results") or []
        if not hits:
            return {
                "verdict": "insufficient_evidence",
                "confidence": 0.35,
                "reason": "No hit-test performed; cannot assert reachability.",
                "evidence_sufficient": False,
                "needs_llm": True,
            }
        landed = [h for h in hits if bool(h.get("hit"))]
        blocked = [h for h in hits if not bool(h.get("hit"))]
        if landed and not blocked:
            return {
                "verdict": "supported",
                "confidence": 0.78,
                "reason": f"{len(landed)} hit-test(s) landed on the target element.",
                "evidence_sufficient": True,
                "needs_llm": False,
            }
        if blocked:
            return {
                "verdict": "contradicted",
                "confidence": 0.7,
                "reason": f"{len(blocked)} hit-test(s) missed the target (occluded / pointer-events:none).",
                "evidence_sufficient": True,
                "needs_llm": False,
            }

    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.4,
        "reason": (
            f"runtime_integrity claim with semantic_type={semantic_type!r} could not be "
            "decided from runtime signals; defer to LLM judge."
        ),
        "evidence_sufficient": False,
        "needs_llm": True,
    }


# ── persistence ────────────────────────────────────────────────────


def validate_persistence(
    persistence_states: dict[str, Any] | None,
    checkpoint: str,
    verification: str,
) -> dict[str, Any]:
    """Validate that state survives a reload / revisit.

    Consumes the ``persistence_states`` evidence produced by the
    orchestrator's reload protocol, expected shape::

        {
          "performed": bool,             # did the orchestrator reload?
          "before_reload": {"visible_text": str, ...},
          "after_reload":  {"visible_text": str, ...},
        }

    Decision: the tokens written before the reload (verification-specific)
    must still be present after the reload. When the reload protocol did
    not run (no evidence), we cannot decide programmatically -> needs_llm.
    """
    states = persistence_states or {}
    if not states.get("performed"):
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.35,
            "reason": "No reload/revisit evidence captured; cannot verify persistence programmatically.",
            "evidence_sufficient": False,
            "needs_llm": True,
        }

    before = states.get("before_reload") or {}
    after = states.get("after_reload") or {}
    before_text = before.get("visible_text", "") or ""
    after_text = after.get("visible_text", "") or ""

    v_tokens = _verification_tokens(verification)
    # Prefer tokens that were actually present before the reload (the data
    # the user created), then check they survived.
    present_before = [t for t in v_tokens if _norm(t) in _norm(before_text)]
    anchor_tokens = present_before or v_tokens
    if not anchor_tokens:
        # Fall back to a coarse "did the content survive" comparison.
        if after_text and before_text and _norm(after_text) == _norm(before_text):
            return {
                "verdict": "supported",
                "confidence": 0.6,
                "reason": "Page content identical before and after reload.",
                "evidence_sufficient": True,
                "needs_llm": True,
            }
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0.4,
            "reason": "No anchor tokens to track across reload; defer to LLM judge.",
            "evidence_sufficient": False,
            "needs_llm": True,
        }

    coverage_after, matched_after = _token_coverage(anchor_tokens, after_text)
    if coverage_after >= 0.5:
        return {
            "verdict": "supported",
            "confidence": min(0.85, 0.6 + coverage_after * 0.25),
            "reason": (
                f"State survived reload: {len(matched_after)}/{len(anchor_tokens)} "
                "tracked tokens still present."
            ),
            "evidence_sufficient": True,
            "needs_llm": False,
        }
    if present_before and coverage_after == 0.0:
        return {
            "verdict": "contradicted",
            "confidence": 0.75,
            "reason": "Content present before reload disappeared afterwards; state not persisted.",
            "evidence_sufficient": True,
            "needs_llm": False,
        }
    return {
        "verdict": "insufficient_evidence",
        "confidence": 0.45,
        "reason": (
            f"Only {int(coverage_after * 100)}% of tracked tokens survived reload; "
            "defer to LLM judge."
        ),
        "evidence_sufficient": False,
        "needs_llm": True,
    }
