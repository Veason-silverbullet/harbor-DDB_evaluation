"""Package DDB seeds and its upstream evaluator as Harbor tasks."""

from __future__ import annotations
import ast
import hashlib
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from harbor.models.task.config import TaskConfig

LOGGER = logging.getLogger(__name__)
TEMPLATE = Path(__file__).parent / "task-template"
BUNDLED_SOURCE = Path(__file__).parent / "snapshot"
DEFAULT_DATA = "data/seed-queries/CN_EN_50_50_v4_atomic_stage1-3_0911.jsonl"
RUNTIME_DIRS = ("config", "cua", "verification", "metrics", "types_")
RUNTIME_FILES = ("claim_builder.py", "orchestrator.py", "usage_tracker.py")
# Standalone analysis/reporting is not part of a Harbor task's verifier.
RUNTIME_EXCLUDES = {
    "metrics/meta_review_aggregator.py",
    "metrics/overview_llm.py",
    "metrics/overview_report.py",
    "metrics/passk.py",
    "verification/meta_review.py",
    "verification/prompts/cua_goal.txt",
    "verification/prompts/meta_review.txt",
    "verification/prompts/overview_synthesis.txt",
}


def remove_preview_credential(source: bytes) -> bytes:
    """Remove DDB's embedded hosted-preview bypass token from the local harness.

    Keep all evaluator logic and the environment-variable override intact. AST
    positions identify only the string fallback, avoiding a whole-file rewrite.
    """
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Attribute) and target.attr == "_preview_secret"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.BoolOp) or not isinstance(node.value.op, ast.Or):
            continue
        fallback = node.value.values[-1]
        if not isinstance(fallback, ast.Constant) or not isinstance(fallback.value, str):
            continue
        lines = source.splitlines(keepends=True)
        if fallback.end_lineno is None or fallback.end_col_offset is None:
            raise ValueError("Unable to locate upstream preview credential")
        start = sum(map(len, lines[: fallback.lineno - 1])) + fallback.col_offset
        end = sum(map(len, lines[: fallback.end_lineno - 1])) + fallback.end_col_offset
        return source[:start] + b'""' + source[end:]
    return source


def safe_relative_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a relative path without traversal: {raw!r}")
    return path


class SeedValidationError(ValueError):
    """The serialized seed dataset cannot be used to generate evaluable tasks."""


def _required_text(record: dict[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SeedValidationError(f"{context}: {field} must be a non-empty string")
    return value


def load_records(data_path: Path) -> dict[str, dict[str, Any]]:
    """Validate seed inputs before generating tasks or replacing a snapshot."""
    records: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for line_number, line in enumerate(data_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        context = f"{data_path.name}:{line_number}"
        if not isinstance(record, dict):
            raise SeedValidationError(f"{context}: expected a seed object")
        task_id = _required_text(record, "id", context)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task_id):
            raise SeedValidationError(f"Unsafe task ID: {task_id!r}")
        if task_id.lower() in seen_ids:
            raise SeedValidationError(f"Duplicate task ID: {task_id}")
        for field in ("query", "category", "difficulty_label", "title"):
            _required_text(record, field, task_id)
        if type(record.get("difficulty")) is not int:
            raise SeedValidationError(f"{task_id}: difficulty must be an integer")
        guideline = record.get("verification_guideline")
        if guideline is not None and not isinstance(guideline, str):
            raise SeedValidationError(f"{task_id}: verification_guideline must be a string")
        checkpoints = record.get("atomic_functional_checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise SeedValidationError(f"{task_id}: atomic checkpoints must be a non-empty list")
        claim_ids: set[str] = set()
        for index, checkpoint in enumerate(checkpoints, 1):
            context = f"{task_id} checkpoint {index}"
            if not isinstance(checkpoint, dict):
                raise SeedValidationError(f"{context}: expected a checkpoint object")
            claim_id = _required_text(checkpoint, "id", context)
            if claim_id in claim_ids:
                raise SeedValidationError(f"{task_id}: duplicate checkpoint ID: {claim_id}")
            for field in ("checkpoint", "verification"):
                _required_text(checkpoint, field, context)
            claim_ids.add(claim_id)
        fixtures = record.get("fixtures", [])
        if not isinstance(fixtures, list):
            raise SeedValidationError(f"{task_id}: fixtures must be a list")
        for fixture in fixtures:
            if not isinstance(fixture, dict):
                raise SeedValidationError(f"{task_id}: expected a fixture object")
            path = fixture.get("path")
            if path is not None and path != "":
                if not isinstance(path, str):
                    raise SeedValidationError(f"{task_id}: fixture path must be a string")
                safe_relative_path(path)
        seen_ids.add(task_id.lower())
        records[task_id] = record
    if not records:
        raise SeedValidationError("Seed dataset must contain at least one task")
    return records


def copy_runtime(runtime: Path, destination: Path) -> None:
    """Copy only evaluator code and prompts, removing the hosted-preview token."""
    destination.mkdir(parents=True, exist_ok=True)
    for directory in RUNTIME_DIRS:
        if not (runtime / directory).is_dir():
            raise FileNotFoundError(f"Missing evaluator directory: {runtime / directory}")
        for source in (runtime / directory).rglob("*"):
            if source.is_file() and source.suffix in {".py", ".txt"}:
                relative = source.relative_to(runtime)
                if relative.as_posix() in RUNTIME_EXCLUDES:
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if relative.as_posix() == "metrics/__init__.py":
                    # The upstream initializer eagerly imports standalone reports.
                    target.write_text('"""DDB task scoring and result persistence."""\n')
                elif relative.as_posix() == "cua/playwright_env.py":
                    target.write_bytes(remove_preview_credential(source.read_bytes()))
                else:
                    shutil.copy2(source, target)
    for filename in RUNTIME_FILES:
        shutil.copy2(runtime / filename, destination / filename)


class DDBBenchmarkAdapter:
    def __init__(
        self,
        source_dir: Path | None,
        output_dir: Path,
        *,
        data_path: Path | None = None,
        fixtures_dir: Path | None = None,
    ) -> None:
        source_dir = (source_dir or BUNDLED_SOURCE).resolve()
        self.output_dir = output_dir.resolve()
        self.data_path = data_path or source_dir / DEFAULT_DATA
        self.fixtures_dir = fixtures_dir or source_dir / "data/CN_EN_fixtures_0911"
        self.runtime = source_dir / "deepdiver-evaluation/level1-func"

    def prepare_tasks(
        self,
        task_ids: list[str] | None = None,
        limit: int | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        records = load_records(self.data_path)
        selected = list(dict.fromkeys(task_ids)) if task_ids is not None else list(records)
        unknown = set(selected) - records.keys()
        if unknown:
            raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
        selected = selected[:limit]
        if not selected:
            raise ValueError("No tasks selected")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return [self._generate(records[task_id], overwrite) for task_id in selected]

    def _generate(self, record: dict[str, Any], overwrite: bool) -> Path:
        task_name = f"ddb-benchmark-{record['id'].lower()}"
        destination = self.output_dir / task_name
        if destination.exists() and not overwrite:
            return destination
        # Assemble before replacing an existing task, so a bad seed cannot destroy it.
        with tempfile.TemporaryDirectory(prefix=".ddb-", dir=self.output_dir) as tmp:
            task = Path(tmp) / task_name
            shutil.copytree(TEMPLATE, task)
            tests = task / "tests"
            upstream = tests / "upstream"
            copy_runtime(self.runtime, upstream)
            missing = []
            for fixture in record.get("fixtures", []):
                if not fixture.get("path"):
                    continue
                relative = safe_relative_path(fixture["path"])
                source = (self.fixtures_dir / relative).resolve()
                if not source.is_relative_to(self.fixtures_dir.resolve()):
                    raise ValueError(f"Fixture escapes fixture directory: {relative}")
                if source.is_file():
                    target = tests / "fixtures" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                else:
                    missing.append(str(relative))
            if missing:
                LOGGER.warning("%s: missing upstream fixtures: %s", record["id"], missing)
            (tests / "query.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
            instruction = (task / "instruction.md").read_text()
            (task / "instruction.md").write_text(instruction.replace("{{QUERY}}", record["query"]))
            config = TaskConfig.model_validate_toml((task / "task.toml").read_text())
            if config.task is None:
                raise ValueError("Task template must contain a [task] section")
            config.task.name = f"ddb-benchmark/{record['id'].lower()}"
            config.metadata.update(
                ddb_id=record["id"],
                category=record["category"],
                difficulty=record["difficulty"],
                num_claims=len(record["atomic_functional_checkpoints"]),
                missing_fixtures=missing,
            )
            (task / "task.toml").write_text(config.model_dump_toml())
            # Hash all inputs, including fixture bytes; never copy source YAML/API keys.
            hashes = {
                str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(task.rglob("*"))
                if p.is_file()
            }
            (tests / "provenance.json").write_text(
                json.dumps(
                    {
                        "dataset": self.data_path.name,
                        "missing_fixtures": missing,
                        "sha256": hashes,
                    },
                    indent=4,
                )
                + "\n"
            )
            for script in (tests / "test.sh", task / "solution/solve.sh"):
                script.chmod(0o755)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(task), destination)
        return destination
