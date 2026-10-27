"""
Build EvalClaim lists from seed JSONL data.

Supports two granularities:
- "atomic": uses atomic_functional_checkpoints (fine-grained, typed claims)
- "fc": uses coarse functional_checkpoints with heuristic type/criticality inference
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from types_.claims import EvalClaim, SeedQuery, resolve_types


def _infer_semantic_type(checkpoint: str, verification: str) -> str:
    """Heuristic semantic_type (capability) inference for coarse FCs."""
    blob = f"{checkpoint}\n{verification}"
    if any(k in blob for k in ("导航", "切换", "进入", "跳转", "Tab", "页面间", "路由")):
        return "navigation"
    if any(k in blob for k in ("刷新", "持久", "localStorage", "重进", "保存后")):
        return "persistence"
    if any(k in blob for k in ("排序", "换算", "统计", "计算", "正确", "百分比", "概率", "日期", "合法")):
        return "computation"
    if any(k in blob for k in ("点击", "输入", "提交", "滑动", "拖拽", "动画", "播放")):
        return "interaction"
    if any(k in blob for k in ("瀑布流", "主题", "配色", "渐变", "布局", "高亮", "样式")):
        return "visual_semantics"
    return "presence"


def _infer_criticality(checkpoint: str, verification: str) -> str:
    """Heuristic criticality inference for coarse FCs."""
    blob = f"{checkpoint}\n{verification}"
    if any(k in blob for k in ("核心", "必须", "主流程", "关键")):
        return "critical"
    if any(k in blob for k in ("次要", "装饰", "可选", "配色细节")):
        return "minor"
    return "major"


def _claims_from_atomic(raw_list: list[dict]) -> list[EvalClaim]:
    """Build claims from atomic_functional_checkpoints (pre-typed).

    Resolves the two-axis types via ``resolve_types`` so both the new
    ``(claim_type, semantic_type)`` schema and legacy single-axis JSONLs
    (the historical 12-type / ``action_effect`` vocabulary) work.
    """
    out: list[EvalClaim] = []
    for ac in raw_list:
        claim_type, semantic_type = resolve_types(
            ac.get("claim_type"), ac.get("semantic_type")
        )
        # Multi-session fields default to the single-session behaviour when
        # absent, so legacy JSONLs (no ``multi_session`` key) are unchanged.
        participants = ac.get("participants") or []
        if not isinstance(participants, list):
            participants = []
        out.append(
            EvalClaim(
                id=ac["id"],
                source_fc_id=ac.get("source_fc_id", ac["id"]),
                checkpoint=ac["checkpoint"],
                verification=ac["verification"],
                claim_type=claim_type,
                criticality=ac.get("criticality", "major"),
                semantic_type=semantic_type,
                multi_session=bool(ac.get("multi_session", False)),
                participants=[str(p) for p in participants],
            )
        )
    return out


def _claims_from_coarse(functional_checkpoints: list[dict]) -> list[EvalClaim]:
    """Build claims from coarse FCs with inferred type and criticality."""
    out: list[EvalClaim] = []
    for fc in functional_checkpoints:
        cid = fc["id"]
        ck = fc["checkpoint"]
        ver = fc["verification"]
        claim_type, semantic_type = resolve_types(None, _infer_semantic_type(ck, ver))
        out.append(
            EvalClaim(
                id=cid,
                source_fc_id=cid,
                checkpoint=ck,
                verification=ver,
                claim_type=claim_type,
                criticality=_infer_criticality(ck, ver),
                semantic_type=semantic_type,
            )
        )
    return out


def build_eval_claims(raw: dict, granularity: str = "atomic") -> list[EvalClaim]:
    """
    Build EvalClaim list from a single JSONL record.

    Args:
        raw: parsed JSON object from seed JSONL
        granularity: "atomic" or "fc"
    """
    if granularity == "atomic":
        atomic = raw.get("atomic_functional_checkpoints") or []
        if isinstance(atomic, list) and len(atomic) > 0:
            return _claims_from_atomic(atomic)
    return _claims_from_coarse(raw.get("functional_checkpoints") or [])


def load_seed_queries(jsonl_path: str | Path, granularity: str = "atomic") -> list[SeedQuery]:
    """Load all seed queries from a JSONL file."""
    queries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            claims = build_eval_claims(raw, granularity=granularity)
            guideline = (raw.get("verification_guideline") or "").strip()
            if guideline:
                # Propagate the per-query guideline onto every claim so the
                # verifier can pass it to the CUA without threading the query.
                for c in claims:
                    c.verification_guideline = guideline
            fixtures = raw.get("fixtures") or []
            if not isinstance(fixtures, list):
                fixtures = []
            # Pre-minted per-identity viewer links for multi-session
            # collaboration (role -> preview URL). Absent / malformed => empty
            # dict, which the pre-flight gate treats as "no links provided".
            session_urls_raw = raw.get("session_urls") or {}
            if isinstance(session_urls_raw, dict):
                session_urls = {
                    str(k): str(v or "") for k, v in session_urls_raw.items()
                }
            else:
                session_urls = {}
            # Query-level roll-up: an explicit top-level ``multi_session``
            # wins; otherwise infer it from the claims so reporting / routing
            # can cheaply tell collaboration queries apart.
            query_multi_session = bool(raw.get("multi_session", False)) or any(
                c.multi_session for c in claims
            )
            queries.append(
                SeedQuery(
                    id=raw["id"],
                    category=raw["category"],
                    difficulty=raw["difficulty"],
                    difficulty_label=raw["difficulty_label"],
                    title=raw["title"],
                    query=raw["query"],
                    eval_claims=claims,
                    verification_guideline=guideline,
                    fixtures=fixtures,
                    multi_session=query_multi_session,
                    session_urls=session_urls,
                )
            )
    return queries


def filter_queries(
    queries: list[SeedQuery],
    ids: list[str] | None = None,
    categories: list[str] | None = None,
    max_difficulty: int | None = None,
) -> list[SeedQuery]:
    """Filter queries by ID, category, or difficulty."""
    result = queries
    if ids is not None:
        id_set = set(ids)
        result = [q for q in result if q.id in id_set]
    if categories is not None:
        cat_set = set(categories)
        result = [q for q in result if q.category in cat_set]
    if max_difficulty is not None:
        result = [q for q in result if q.difficulty <= max_difficulty]
    return result


def filter_claims(
    queries: list[SeedQuery],
    claim_ids: list[str],
) -> list[SeedQuery]:
    """Keep only the claims whose id is in ``claim_ids`` (e.g. ["AC-4"]).

    Mutates each query's ``eval_claims`` in place, recomputes the query-level
    ``multi_session`` roll-up from the surviving claims (so a single-session
    subset is not blocked by the multi-session pre-flight gate), and drops
    queries left with no claims. Claim ids that match no query are reported
    with a WARNING so typos don't silently evaluate nothing.
    """
    wanted = {str(c).strip() for c in claim_ids if str(c).strip()}
    matched: set[str] = set()
    result: list[SeedQuery] = []
    for q in queries:
        kept = [c for c in q.eval_claims if c.id in wanted]
        matched.update(c.id for c in kept)
        if not kept:
            continue
        q.eval_claims = kept
        q.multi_session = any(c.multi_session for c in kept)
        result.append(q)
    unmatched = sorted(wanted - matched)
    if unmatched:
        logging.getLogger(__name__).warning(
            "--claims ids matched no claim in the selected queries: %s",
            ", ".join(unmatched),
        )
    return result
