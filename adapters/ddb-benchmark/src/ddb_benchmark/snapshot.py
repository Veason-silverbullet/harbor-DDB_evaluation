"""Import the inputs needed to generate DDB tasks without a source checkout."""

from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from ddb_benchmark.adapter import (
    BUNDLED_SOURCE,
    DEFAULT_DATA,
    DEFAULT_FIXTURES,
    RUNTIME_EXCLUDES,
    copy_runtime,
    load_dataset_provenance,
    load_records,
    safe_relative_path,
)


def get_source_revision(source: Path) -> str | None:
    """Preserve bundled provenance without requiring Git or an upstream checkout."""
    manifest = source / "manifest.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected a snapshot manifest object: {manifest}")
        revision = metadata.get("source_revision")
        if revision is not None and not isinstance(revision, str):
            raise ValueError(f"Invalid source revision in {manifest}")
        return revision
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def create_snapshot(
    source: Path | None,
    destination: Path,
    *,
    data_path: Path | None = None,
    fixtures_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    source = (source or BUNDLED_SOURCE).resolve()
    destination = destination.resolve()
    if source.is_relative_to(destination):
        raise ValueError("Snapshot destination must not contain the source checkout")
    source_data = (data_path or source / DEFAULT_DATA).resolve()
    fixtures = (fixtures_dir or source / DEFAULT_FIXTURES).resolve()
    if source_data.is_relative_to(destination) or fixtures.is_relative_to(destination):
        raise ValueError(
            "Snapshot destination must not contain dataset or fixture inputs"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Snapshot already exists: {destination}; use --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ddb-snapshot-", dir=destination.parent) as tmp:
        snapshot = Path(tmp) / "snapshot"
        data = snapshot / DEFAULT_DATA
        data.parent.mkdir(parents=True)
        shutil.copy2(source_data, data)
        records = load_records(data)
        generation = load_dataset_provenance(source_data)
        if generation is not None:
            shutil.copy2(
                source_data.with_suffix(".provenance.json"),
                data.with_suffix(".provenance.json"),
            )
        copy_runtime(
            source / "deepdiver-evaluation/level1-func",
            snapshot / "deepdiver-evaluation/level1-func",
        )
        missing: dict[str, list[str]] = {}
        for record in records.values():
            for fixture in record.get("fixtures", []):
                if not fixture.get("path"):
                    continue
                relative = safe_relative_path(fixture["path"])
                original = (fixtures / relative).resolve()
                if not original.is_relative_to(fixtures.resolve()):
                    raise ValueError(f"Fixture escapes fixture directory: {relative}")
                if not original.is_file():
                    missing.setdefault(record["id"], []).append(str(relative))
                    continue
                target = snapshot / DEFAULT_FIXTURES / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
        manifest = {
            "source": "DDB_evaluation",
            "source_revision": get_source_revision(source),
            "task_count": len(records),
            "dataset": source_data.name,
            "missing_fixtures": missing,
            "runtime_exclusions": sorted(RUNTIME_EXCLUDES),
            "sanitization": "Removed the embedded hosted-preview token; omitted source YAML configs.",
            "sha256": {
                str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(snapshot.rglob("*"))
                if p.is_file()
            },
        }
        (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=4) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(snapshot), destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Override the bundled evaluator with another snapshot or source checkout")
    parser.add_argument("--output-dir", type=Path, default=BUNDLED_SOURCE)
    parser.add_argument("--data", type=Path, help="Override the source seed JSONL")
    parser.add_argument("--fixtures-dir", type=Path, help="Root for seed fixture paths")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        create_snapshot(
            args.source_dir,
            args.output_dir,
            data_path=args.data,
            fixtures_dir=args.fixtures_dir,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Prepared DDB snapshot in {args.output_dir}")


if __name__ == "__main__":
    main()
