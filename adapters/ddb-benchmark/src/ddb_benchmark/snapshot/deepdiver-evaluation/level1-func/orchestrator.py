"""
Evaluation orchestrator: wires together data loading, CUA, verification, and metrics.

Supports single-query and batch-query evaluation.
Detects quota exhaustion and saves partial results for resumption.
"""
from __future__ import annotations
import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any, Callable
from config.settings import EvalConfig, resolve_fixtures_dirs, resolve_output_dir
from types_.claims import SeedQuery
from types_.results import ClaimResult, QueryResult
from types_.errors import QuotaExhaustedError
from cua.playwright_env import PlaywrightEnv
from cua.cua_agent import CUAAgent
from verification.claim_verifier import ClaimVerifier
from metrics.aggregator import aggregate_query_metrics
from metrics.report_generator import save_partial_query_result_json, load_partial_claim_results

logger = logging.getLogger(__name__)


# Callback fired right after each query result is produced. Used to persist
# results incrementally so that interrupts only lose the in-flight query.
# Both sync and async callables are accepted.
QueryDoneCallback = Callable[[QueryResult], Any]


async def _maybe_await(value: Any) -> None:
    """Await ``value`` if it is awaitable; otherwise no-op."""
    if inspect.isawaitable(value):
        await value


class EvalOrchestrator:
    """
    Top-level orchestrator for functionality evaluation.

    Manages browser lifecycle, CUA agent, and claim verification pipeline.
    """

    def __init__(
        self,
        config: EvalConfig,
        run_stem: str | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: full evaluation config.
            run_stem: optional run identifier (e.g. ``opus_opus_url_opus_20260509_160010``).
                When set, the runner is pointed at ``<output_dir>/<run_stem>``
                as its screenshot root; the runner itself appends
                ``<query_id>/screenshots/<claim>_stepN.png`` so each query's
                screenshots land next to its result JSON. When ``None``,
                the runner falls back to ``<output_dir>`` and still creates
                per-query subdirs underneath.
        """
        self._config = config
        self._run_stem = run_stem
        self._env: PlaywrightEnv | None = None
        self._cua: CUAAgent | None = None
        self._verifier: ClaimVerifier | None = None
        self.partial_results: list[QueryResult] = []
        self.interrupted_at: dict[str, Any] | None = None

    async def start(self) -> None:
        """Initialize browser and CUA agent."""
        cua_cfg = self._config.cua
        output_dir = resolve_output_dir(self._config)
        # Pass the *run root* (or output_dir when no stem). The runner
        # appends <query_id>/screenshots/<claim>_stepN.png itself so
        # each query owns a self-contained subdir holding both the
        # result JSON and its screenshots.
        if self._run_stem:
            ss_dir = output_dir / self._run_stem
        else:
            ss_dir = output_dir

        self._env = PlaywrightEnv(
            viewport=cua_cfg.viewport,
            headless=cua_cfg.headless,
            enable_virtual_clock=cua_cfg.enable_virtual_clock,
            fixture_roots=resolve_fixtures_dirs(self._config),
        )
        await self._env.start()

        self._cua = CUAAgent(
            env=self._env,
            api_key=cua_cfg.api_key,
            model=cua_cfg.model,
            agent_protocol=cua_cfg.agent_protocol,
            tool_mode=cua_cfg.tool_mode,
            enable_readonly_tools=cua_cfg.enable_readonly_tools,
            base_url=cua_cfg.base_url,
            max_iterations=cua_cfg.max_iterations,
            max_iterations_by_claim_type=cua_cfg.max_iterations_by_claim_type,
            screenshot_delay_ms=cua_cfg.screenshot_delay_ms,
            save_screenshots=self._config.save_screenshots,
            screenshot_dir=ss_dir,
            extra_headers=cua_cfg.extra_headers,
            max_output_tokens=cua_cfg.max_output_tokens,
            thinking_budget_tokens=cua_cfg.thinking_budget_tokens,
            reasoning_effort=cua_cfg.reasoning_effort,
            reasoning_summary=cua_cfg.reasoning_summary,
            reasoning_passthrough=cua_cfg.reasoning_passthrough,
            enable_multi_session=cua_cfg.enable_multi_session,
            viewer_max_iterations=cua_cfg.viewer_max_iterations,
            viewer_model=cua_cfg.viewer_model,
            max_viewers_per_claim=cua_cfg.max_viewers_per_claim,
            capture_transient_immediate=cua_cfg.capture_transient_immediate,
            enable_veracity_check=cua_cfg.enable_veracity_check,
            search_provider=cua_cfg.search_provider,
            search_api_key=cua_cfg.search_api_key,
            search_endpoint=cua_cfg.search_endpoint,
            search_max_results=cua_cfg.search_max_results,
        )

        self._verifier = ClaimVerifier(
            cua_agent=self._cua,
            env=self._env,
            config=self._config,
        )

    async def stop(self) -> None:
        """Shutdown browser."""
        if self._env:
            await self._env.stop()

    def _partial_path(self, query_id: str) -> Path | None:
        """Location of the claim-level checkpoint file for ``query_id``.

        Mirrors the per-query result layout
        (``<output_dir>/<run_stem>/<qid>/<qid>.partial.json``). Returns
        ``None`` when no run stem is set, in which case claim-level resume is
        disabled (there is no stable directory to key the checkpoint on).
        """
        if not self._run_stem:
            return None
        output_dir = resolve_output_dir(self._config)
        return output_dir / self._run_stem / query_id / f"{query_id}.partial.json"

    def _build_query_result(
        self,
        query: SeedQuery,
        claim_results: list[ClaimResult],
    ) -> QueryResult:
        """Assemble a (possibly partial) :class:`QueryResult` from claim results."""
        metrics = aggregate_query_metrics(claim_results)
        return QueryResult(
            query_id=query.id,
            title=query.title,
            category=query.category,
            difficulty=query.difficulty,
            claim_results=claim_results,
            func_score=metrics.weighted_func_score,
            metrics=metrics,
        )

    async def evaluate_single(
        self,
        query: SeedQuery,
        app_url: str,
    ) -> QueryResult:
        """
        Evaluate all claims for a single query.

        Args:
            query: The seed query with claims
            app_url: URL of the generated web app to evaluate

        Raises:
            QuotaExhaustedError: re-raised so the batch runner can save partial results.
        """
        if not self._env or not self._verifier:
            raise RuntimeError("Orchestrator not started. Call start() first.")

        logger.info("Evaluating %s (%s) at %s", query.id, query.title, app_url)

        # Per-query navigation resets the SPA state for isolation. We use
        # `domcontentloaded` (set as default in PlaywrightEnv.goto) instead
        # of `networkidle` and a short warm-up wait, which is plenty for
        # most generated apps and saves ~1s+ per claim.
        await self._env.goto(app_url)
        await self._env.wait(800)

        # Register this query's file-upload fixtures (if any) so the CUA's
        # ``upload_file`` tool / any native file picker can attach them. Safe
        # no-op when the query declares none.
        try:
            self._env.set_fixtures(getattr(query, "fixtures", None))
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to register fixtures for %s: %s", query.id, e)

        # ── claim-level resume ────────────────────────────────────
        # Reload any claims already evaluated for this query in a prior
        # (interrupted) run. Those claims are skipped below so we don't
        # re-spend CUA tokens or re-capture screenshots; their cached
        # results (incl. screenshot paths) are carried straight through.
        claim_results: list[ClaimResult] = []
        completed_ids: set[str] = set()
        partial_path = self._partial_path(query.id)
        if partial_path is not None and partial_path.exists():
            prior = load_partial_claim_results(partial_path)
            if prior:
                claim_results.extend(prior)
                completed_ids = {cr.claim_id for cr in prior}
                logger.info(
                    "Resume: reusing %d cached claim result(s) for %s; "
                    "skipping %s",
                    len(prior), query.id, sorted(completed_ids),
                )

        for i, claim in enumerate(query.eval_claims):
            if claim.id in completed_ids:
                logger.info(
                    "  Claim %d/%d: %s [cached — reused, 0 tokens]",
                    i + 1, len(query.eval_claims), claim.id,
                )
                continue
            logger.info(
                "  Claim %d/%d: %s [%s] %s",
                i + 1, len(query.eval_claims),
                claim.id, claim.claim_type, claim.checkpoint[:60],
            )
            try:
                result = await self._verifier.verify(
                    claim=claim,
                    app_url=app_url,
                    query_id=query.id,
                    session_urls=getattr(query, "session_urls", None) or {},
                )
                claim_results.append(result)
                logger.info("    -> %s (%.0f%%)", result.verdict, result.confidence * 100)
            except QuotaExhaustedError:
                logger.critical(
                    "QUOTA EXHAUSTED during claim %d/%d (%s) of query %s",
                    i + 1, len(query.eval_claims), claim.id, query.id,
                )
                raise
            except Exception as e:
                logger.error("    -> ERROR: %s", e)
                claim_results.append(
                    ClaimResult(
                        query_id=query.id,
                        claim_id=claim.id,
                        source_fc_id=claim.source_fc_id,
                        checkpoint=claim.checkpoint,
                        verification=claim.verification,
                        claim_type=claim.claim_type,
                        criticality=claim.criticality,
                        verdict="insufficient_evidence",
                        confidence=0.0,
                        reason=f"Evaluation error: {type(e).__name__}: {e}",
                        evidence_sufficient=False,
                        used_validators=["error"],
                        native_tool_use_stats={},
                        page_load_failure={},
                    )
                )

            # Checkpoint progress after every claim (success or handled error)
            # so a quota exhaustion / crash on the *next* claim resumes here.
            if partial_path is not None:
                try:
                    save_partial_query_result_json(
                        self._build_query_result(query, claim_results),
                        partial_path,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to checkpoint partial result for %s: %s",
                        query.id, e,
                    )

        # Keep claim order aligned with the seed (resume prepends cached
        # claims; this guards against any out-of-order checkpoint file).
        order = {c.id: idx for idx, c in enumerate(query.eval_claims)}
        claim_results.sort(key=lambda cr: order.get(cr.claim_id, len(order)))

        return self._build_query_result(query, claim_results)

    async def evaluate_batch(
        self,
        queries: list[SeedQuery],
        url_map: dict[str, str],
        default_url: str = "",
        on_query_done: QueryDoneCallback | None = None,
    ) -> list[QueryResult]:
        """
        Evaluate multiple queries sequentially.

        Each query navigates to its own app URL from the url_map,
        falling back to default_url if not found.

        On QuotaExhaustedError the already-collected results are stored in
        ``self.partial_results`` and the exception is re-raised so the
        caller (run_eval.py) can save them to disk.

        Args:
            on_query_done: optional callback invoked with each completed
                ``QueryResult`` immediately after it is produced. Used by
                callers to persist results incrementally; may be sync or
                async. Exceptions from the callback are logged and ignored.
        """
        results: list[QueryResult] = []
        for i, query in enumerate(queries):
            url = url_map.get(query.id, default_url)
            if not url:
                logger.warning("No URL for %s, skipping", query.id)
                continue
            logger.info(
                "[%d/%d] %s -- %s",
                i + 1, len(queries), query.id, query.title,
            )
            try:
                result = await self.evaluate_single(query, url)
                results.append(result)
                if on_query_done is not None:
                    try:
                        await _maybe_await(on_query_done(result))
                    except Exception as cb_err:  # noqa: BLE001
                        logger.error(
                            "on_query_done callback failed for %s: %s",
                            query.id, cb_err,
                        )
            except QuotaExhaustedError as qe:
                self.partial_results = results
                self.interrupted_at = {
                    "query_index": i,
                    "query_id": query.id,
                    "query_title": query.title,
                    "total_queries": len(queries),
                    "completed_queries": len(results),
                    "source": qe.source,
                    "detail": qe.detail,
                }
                logger.critical(
                    "QUOTA EXHAUSTED at query %d/%d (%s). "
                    "%d query results saved. Re-raising.",
                    i + 1, len(queries), query.id, len(results),
                )
                raise
            except Exception as e:
                logger.error("Query %s failed: %s", query.id, e)
        return results

    async def evaluate_batch_concurrent(
        self,
        queries: list[SeedQuery],
        url_map: dict[str, str],
        default_url: str = "",
        max_concurrency: int = 1,
        on_query_done: QueryDoneCallback | None = None,
    ) -> list[QueryResult]:
        """
        Evaluate multiple queries with limited concurrency.

        Each concurrent worker spins up its own ``EvalOrchestrator`` (and
        therefore its own Playwright browser + CUA agent), so workers are
        fully isolated.

        On :class:`QuotaExhaustedError` from any worker, in-flight workers
        are cancelled, completed results are stored in
        ``self.partial_results``, ``self.interrupted_at`` is populated, and
        the exception is re-raised so the caller can save partial reports.

        Args:
            on_query_done: optional callback invoked with each completed
                ``QueryResult`` immediately after it is produced. Used by
                callers to persist results incrementally; may be sync or
                async. Per-query callbacks across workers run sequentially
                because they fire from ``asyncio.as_completed`` below, so
                callers do NOT need to lock when writing distinct files.
        """
        if max_concurrency <= 1:
            return await self.evaluate_batch(
                queries, url_map, default_url, on_query_done=on_query_done,
            )

        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[QueryResult | None] = [None] * len(queries)
        first_quota: dict[str, Any] = {}

        async def _eval_one(idx: int, query: SeedQuery) -> None:
            url = url_map.get(query.id, default_url)
            if not url:
                return
            async with semaphore:
                sub = EvalOrchestrator(self._config, run_stem=self._run_stem)
                await sub.start()
                try:
                    qr = await sub.evaluate_single(query, url)
                    results[idx] = qr
                    if on_query_done is not None:
                        try:
                            await _maybe_await(on_query_done(qr))
                        except Exception as cb_err:  # noqa: BLE001
                            logger.error(
                                "on_query_done callback failed for %s: %s",
                                query.id, cb_err,
                            )
                except QuotaExhaustedError as qe:
                    if not first_quota:
                        first_quota.update({
                            "query_index": idx,
                            "query_id": query.id,
                            "query_title": query.title,
                            "source": qe.source,
                            "detail": qe.detail,
                        })
                    raise
                finally:
                    try:
                        await sub.stop()
                    except Exception:
                        pass

        tasks = [
            asyncio.create_task(_eval_one(i, q))
            for i, q in enumerate(queries)
        ]

        try:
            for fut in asyncio.as_completed(tasks):
                try:
                    await fut
                except QuotaExhaustedError:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    completed = [r for r in results if r is not None]
                    self.partial_results = completed
                    if first_quota:
                        first_quota["total_queries"] = len(queries)
                        first_quota["completed_queries"] = len(completed)
                        self.interrupted_at = first_quota
                    logger.critical(
                        "QUOTA EXHAUSTED in concurrent batch (worker for %s). "
                        "Cancelling siblings; %d query results saved.",
                        first_quota.get("query_id"), len(completed),
                    )
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                except Exception as e:
                    logger.error("Concurrent worker failed: %s", e)
        finally:
            pending = [t for t in tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        return [r for r in results if r is not None]
