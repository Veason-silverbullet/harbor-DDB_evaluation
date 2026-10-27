"""Start the submitted app and run DDB's original claim evaluator and scorer."""

from __future__ import annotations
import asyncio
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4
from ddb_runtime import adapt_evaluator, local_url

LOGGER = logging.getLogger(__name__)
TESTS = Path(__file__).resolve().parent
APP = Path("/app")
LOGS = Path("/logs/verifier")
APP_URL = "http://127.0.0.1:5173/"


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before running evaluation")
    return value


def make_config():
    from config.settings import AGENT_PROTOCOLS, CUAConfig, EvalConfig, LLMJudgeConfig

    protocol = os.environ.get("DDB_CUA_AGENT_PROTOCOL", "function_calling")
    if protocol not in AGENT_PROTOCOLS:
        raise ValueError(f"Unsupported DDB_CUA_AGENT_PROTOCOL: {protocol}")
    provider = os.environ.get("DDB_JUDGE_PROVIDER", "openai")
    if provider not in {"openai", "anthropic"}:
        raise ValueError(f"Unsupported DDB_JUDGE_PROVIDER: {provider}")
    cua = CUAConfig(
        agent_protocol=protocol,
        api_key=required_env("DDB_CUA_API_KEY"),
        model=required_env("DDB_CUA_MODEL"),
        base_url=required_env("DDB_CUA_BASE_URL"),
        max_iterations=int(os.environ.get("DDB_CUA_MAX_ITERATIONS", "20")),
        enable_multi_session=True,
        viewer_max_iterations=20,
        enable_veracity_check=True,
        search_api_key=os.environ.get("DDB_SEARCH_API_KEY", ""),
        search_provider=os.environ.get("DDB_SEARCH_PROVIDER", "tavily"),
        search_endpoint=os.environ.get("DDB_SEARCH_ENDPOINT", ""),
    )
    judge = LLMJudgeConfig(
        provider=provider,
        api_key=required_env("DDB_JUDGE_API_KEY"),
        model=required_env("DDB_JUDGE_MODEL"),
        base_url=required_env("DDB_JUDGE_BASE_URL"),
        max_tokens=int(os.environ.get("DDB_JUDGE_MAX_TOKENS", "32000")),
    )
    return EvalConfig(
        cua=cua,
        llm_judge=judge,
        fixtures_dirs=[str(TESTS / "fixtures")],
        output_dir=str(LOGS / "ddb"),
        _root_dir=TESTS,
    )


def validate_fixtures(query) -> None:
    missing = []
    for fixture in query.fixtures:
        if raw := fixture.get("path"):
            root = (TESTS / "fixtures").resolve()
            path = (root / raw).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                missing.append(raw)
    if missing:
        raise FileNotFoundError(f"Missing upstream fixtures for {query.id}: {missing}")


def configure_sessions(query) -> None:
    """Load explicit entry hints; identity and room setup happen through the UI."""
    mapping_path = APP / "session_urls.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    if not isinstance(mapping, dict):
        raise TypeError("/app/session_urls.json must be a participant-to-URL object")
    roles = {
        role
        for claim in query.eval_claims
        for role in claim.participants
        if role.lower() != "creator"
    }
    # Never use stale DeepDiver deployment links embedded in a seed.
    query.session_urls = {}
    for role in roles:
        if role in mapping:
            raw = mapping[role]
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Participant {role} must have a non-empty entry URL")
            query.session_urls[role] = local_url(raw, APP_URL)


def app_readiness_error(url: str) -> str | None:
    try:
        with urlopen(url, timeout=2) as response:
            if response.status < 400:
                return None
            return f"{url} returned HTTP {response.status}"
    except (URLError, TimeoutError, ConnectionError) as exc:
        return f"{url}: {exc}"


def stop_app(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def start_app() -> tuple[subprocess.Popen[bytes] | None, str | None]:
    error = app_readiness_error(APP_URL)
    if error is None:
        return None, None
    if not (APP / "start.sh").is_file():
        return None, f"{error}; /app/start.sh is missing"
    # Judge credentials are supplied only to the verifier, never to the app server.
    app_env = {k: v for k, v in os.environ.items() if not k.startswith("DDB_")}
    with (LOGS / "app.log").open("w") as log:
        process = subprocess.Popen(
            ["bash", str(APP / "start.sh")],
            cwd=APP,
            env=app_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        error = app_readiness_error(APP_URL)
        if error is None:
            return process, None
        if (code := process.poll()) is not None:
            return process, f"{error}; start.sh exited with code {code}; see app.log"
        time.sleep(0.5)
    return process, f"{error}; app startup timed out after 120 seconds; see app.log"


def write_reward(score: float) -> None:
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid DDB score: {score}")
    (LOGS / "reward.txt").write_text(f"{score:.10f}\n")


async def evaluate(query, cfg) -> None:
    from metrics.report_generator import save_query_result_json
    from orchestrator import EvalOrchestrator

    with adapt_evaluator(judge_provider=cfg.llm_judge.provider) as health:
        # A retry must not reuse upstream partial verdicts from a failed service
        # call, including old verdicts produced before error propagation was fixed.
        orchestrator = EvalOrchestrator(cfg, run_stem=f"evaluation-{uuid4().hex}")
        try:
            await orchestrator.start()
            result = await orchestrator.evaluate_single(query, APP_URL)
            save_query_result_json(result, LOGS / "ddb" / "result.json")
            health.check()
            if len(result.claim_results) != len(query.eval_claims):
                raise RuntimeError("DDB returned an incomplete evaluation")
            errors = [r.claim_id for r in result.claim_results if "error" in r.used_validators]
            if errors:
                raise RuntimeError(f"DDB evaluation errors in claims: {errors}")
            write_reward(result.func_score)
        finally:
            await orchestrator.stop()


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.txt").unlink(missing_ok=True)
    sys.path.insert(0, str(TESTS / "upstream"))
    logging.basicConfig(level=logging.INFO)
    process = None
    try:
        from claim_builder import load_seed_queries

        cfg = make_config()
        queries = load_seed_queries(TESTS / "query.jsonl", granularity="atomic")
        if len(queries) != 1 or not queries[0].eval_claims:
            raise ValueError("Expected exactly one query with atomic claims")
        query = queries[0]
        cfg.cua.enable_readonly_tools = query.multi_session
        validate_fixtures(query)
        configure_sessions(query)
        process, error = start_app()
        if error is not None:
            LOGGER.error("App readiness failed: %s", error)
            (LOGS / "app_failure.json").write_text(
                json.dumps(
                    {
                        "query_id": query.id,
                        "reason": error,
                    },
                    indent=4,
                )
                + "\n"
            )
            write_reward(0.0)
            return
        asyncio.run(evaluate(query, cfg))
    except Exception as exc:
        (LOGS / "error.json").write_text(
            json.dumps(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=4,
            )
            + "\n"
        )
        raise
    finally:
        stop_app(process)


if __name__ == "__main__":
    main()
