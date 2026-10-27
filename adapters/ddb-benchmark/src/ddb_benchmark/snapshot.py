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
    RUNTIME_EXCLUDES,
    copy_runtime,
    load_records,
    safe_relative_path,
)


def create_snapshot(
    source: Path, destination: Path, *, overwrite: bool = False
) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source.is_relative_to(destination):
        raise ValueError("Snapshot destination must not contain the source checkout")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Snapshot already exists: {destination}; use --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ddb-snapshot-", dir=destination.parent) as tmp:
        snapshot = Path(tmp) / "snapshot"
        data = snapshot / DEFAULT_DATA
        data.parent.mkdir(parents=True)
        shutil.copy2(source / DEFAULT_DATA, data)
        records = load_records(data)
        copy_runtime(
            source / "deepdiver-evaluation/level1-func",
            snapshot / "deepdiver-evaluation/level1-func",
        )
        fixtures = source / "data/CN_EN_fixtures_0911"
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
                target = snapshot / "data/CN_EN_fixtures_0911" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        manifest = {
            "source": "DDB_evaluation",
            "source_revision": revision.stdout.strip()
            if revision.returncode == 0
            else None,
            "task_count": len(records),
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
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=BUNDLED_SOURCE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        create_snapshot(args.source_dir, args.output_dir, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Prepared DDB snapshot in {args.output_dir}")


if __name__ == "__main__":
    main()
