"""Collect upstream Stage 1–3 outputs and their fixtures into a Harbor dataset."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from ddb_benchmark.adapter import load_records, safe_relative_path

VERSIONS = ("long", "lite")
RECORD_FILES = (
    "query.json",
    "stage2_raw.json",
    "stage3_guideline_raw.json",
    "stage3_raw.json",
)


def select_input(input_dir: Path, version: str | None) -> tuple[Path, str | None]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if version is not None and version not in VERSIONS:
        raise ValueError(f"Unknown prompt version: {version}")
    if version and (input_dir / version).is_dir():
        return input_dir / version, version
    if input_dir.name in VERSIONS:
        if version and version != input_dir.name:
            raise ValueError("Requested version does not match the input directory")
        return input_dir, input_dir.name
    if (input_dir / "query.json").is_file():
        inferred = input_dir.parent.name if input_dir.parent.name in VERSIONS else None
        if version and inferred and version != inferred:
            raise ValueError("Requested version does not match the query directory")
        return input_dir, version or inferred
    available = [name for name in VERSIONS if (input_dir / name).is_dir()]
    if available:
        if version:
            raise ValueError(f"No {version} outputs in {input_dir}")
        if len(available) > 1:
            raise ValueError("Both long and lite outputs exist; select --version")
        return input_dir / available[0], available[0]
    return input_dir, version


def extract_json(text: str) -> Any:
    """Match upstream's JSON extraction from raw model responses."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def read_object(
    path: Path, root: Path, hashes: dict[str, str], *, nested: bool = False
) -> dict[str, Any]:
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Input escapes selected directory: {path}")
    content = path.read_bytes()
    hashes[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
    try:
        payload = json.loads(content)
        if (
            nested
            and isinstance(payload, dict)
            and isinstance(payload.get("text"), str)
        ):
            payload = extract_json(payload["text"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def default_stage3_log(root: Path, version: str | None) -> Path | None:
    """Locate the standard sibling run log for a version or single-query input."""
    if version is None:
        return None
    version_dir = root if root.name == version else root.parent
    if version_dir.name != version:
        return None
    candidate = version_dir.parent / f"{version}_stage3.jsonl"
    return candidate if candidate.exists() else None


def read_stage3_log(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    content = path.read_bytes()
    records = {}
    for number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Stage 3 run log: {path}:{number}") from exc
        if (
            not isinstance(record, dict)
            or record.get("stage") != 3
            or not isinstance(record.get("query_id"), str)
            or not record["query_id"]
        ):
            raise ValueError(f"Invalid Stage 3 run log record: {path}:{number}")
        # A later failed retry must supersede an earlier successful entry.
        records[record["query_id"]] = record
    return records, hashlib.sha256(content).hexdigest()


def generated_contents(final: dict[str, Any], directory: Path) -> dict[Path, bytes]:
    """Decode final Stage 3 payloads using upstream's Base64 normalization."""
    files = final.get("generated_files")
    if not isinstance(files, list):
        raise ValueError(
            f"Missing generated_files in final Stage 3 artifact: {directory}"
        )
    contents = {}
    for item in files:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("path", "name", "content_base64")
        ):
            raise ValueError(
                f"Incomplete generated file in final Stage 3 artifact: {directory}"
            )
        path = safe_relative_path(item["path"].strip())
        if path in contents:
            raise ValueError(f"Duplicate generated file path in {directory}: {path}")
        encoded = item["content_base64"].strip()
        if encoded.startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        encoded = encoded.replace("```base64", "").replace("```", "")
        encoded = "".join(encoded.split())
        encoded += "=" * (-len(encoded) % 4)
        try:
            contents[path] = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(
                f"Invalid generated file Base64 in {directory}: {path}"
            ) from exc
    return contents


def import_outputs(
    input_dir: Path,
    output_dir: Path,
    *,
    version: str | None = None,
    generation_model: str | None = None,
    source_dir: Path | None = None,
    source_revision: str | None = None,
    stage3_log: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate and stage the entire import before replacing an existing dataset."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir.is_relative_to(output_dir) or output_dir.is_relative_to(input_dir):
        raise ValueError("Import input and output directories must not overlap")
    if source_dir is not None and source_dir.resolve().is_relative_to(output_dir):
        raise ValueError("Import output must not contain the source checkout")
    if source_dir is not None and source_revision is not None:
        raise ValueError("Specify either source_dir or source_revision, not both")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Import already exists: {output_dir}; use --overwrite")
    root, version = select_input(input_dir, version)
    if not root.resolve().is_relative_to(input_dir):
        raise ValueError("Version directory escapes the input directory")
    root = root.resolve()
    record_dirs = (
        [root]
        if any((root / name).exists() for name in RECORD_FILES)
        else sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and any((path / name).exists() for name in RECORD_FILES)
        )
    )
    if not record_dirs:
        raise ValueError(f"No Stage 1–3 query outputs found in {root}")
    stage3_log = stage3_log or default_stage3_log(root, version)
    completion = None
    log_checksum = None
    if stage3_log is not None:
        if stage3_log.resolve().is_relative_to(output_dir):
            raise ValueError("Import output must not contain the Stage 3 run log")
        completion, log_checksum = read_stage3_log(stage3_log)
    revision = source_revision
    if source_dir is not None:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"Cannot determine source revision: {source_dir}")
        revision = result.stdout.strip()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ddb-import-", dir=output_dir.parent
    ) as tmp:
        staged = Path(tmp) / "dataset"
        staged.mkdir()
        rows = []
        final_artifacts = []
        hashes: dict[str, str] = {}
        for directory in record_dirs:
            query = read_object(directory / "query.json", root, hashes)
            stage2 = read_object(
                directory / "stage2_raw.json", root, hashes, nested=True
            )
            stage3 = read_object(directory / "stage3_guideline_raw.json", root, hashes)
            final = read_object(directory / "stage3_raw.json", root, hashes)
            if any(
                final.get(key) != stage3.get(key)
                for key in ("fixtures", "verification_guideline")
            ):
                raise ValueError(f"Stale final Stage 3 artifact: {directory}")
            if completion is not None:
                query_id = query.get("id")
                status = completion.get(query_id) if isinstance(query_id, str) else None
                if status is None or "error" in status:
                    raise ValueError(
                        f"Stage 3 did not complete for query in {directory}"
                    )
                if status.get("raw") != final:
                    raise ValueError(
                        f"Stage 3 run log disagrees with final artifact: {directory}"
                    )
            final_artifacts.append(final)
            guideline = stage3.get("verification_guideline")
            if not isinstance(guideline, str) or not guideline.strip():
                raise ValueError(f"Missing verification guideline: {directory}")
            rows.append(
                {
                    **query,
                    "atomic_functional_checkpoints": stage2.get(
                        "atomic_functional_checkpoints"
                    ),
                    "verification_guideline": guideline,
                    "fixtures": stage3.get("fixtures"),
                }
            )
        data = staged / "query.jsonl"
        data.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        records = load_records(data)
        (staged / "fixtures").mkdir()
        for directory, row, final in zip(
            record_dirs, records.values(), final_artifacts, strict=True
        ):
            expected_contents = generated_contents(final, directory)
            for fixture in row["fixtures"]:
                raw = fixture.get("path")
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError(f"{row['id']}: generated fixture needs a path")
                relative = safe_relative_path(raw)
                original = directory / relative
                if not original.resolve().is_relative_to(directory.resolve()):
                    raise ValueError(f"Fixture escapes query directory: {relative}")
                content = original.read_bytes()
                if relative not in expected_contents:
                    raise ValueError(
                        f"Fixture missing from final Stage 3 generated_files: {relative}"
                    )
                if content != expected_contents[relative]:
                    raise ValueError(
                        f"Fixture differs from final Stage 3 generated content: {original}"
                    )
                hashes[original.relative_to(root).as_posix()] = hashlib.sha256(
                    content
                ).hexdigest()
                target = staged / "fixtures" / relative
                if target.exists() and target.read_bytes() != content:
                    raise ValueError(
                        f"Conflicting fixture path across queries: {relative}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        metadata = {
            "source": "DDB_evaluation Stage 1–3",
            "source_revision": revision,
            "prompt_version": version,
            "generation_model": generation_model,
            "task_count": len(records),
            "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "source_sha256": hashes,
            **({"stage3_log_sha256": log_checksum} if log_checksum is not None else {}),
        }
        data.with_suffix(".provenance.json").write_text(
            json.dumps(metadata, indent=4) + "\n"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(staged), output_dir)
    return output_dir / "query.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--version", choices=VERSIONS)
    parser.add_argument("--generation-model", help="Model used upstream, if known")
    parser.add_argument(
        "--stage3-log",
        type=Path,
        help="Override the automatically located Stage 3 run JSONL",
    )
    provenance = parser.add_mutually_exclusive_group()
    provenance.add_argument(
        "--source-revision", help="Revision recorded with the exported outputs; no checkout needed"
    )
    provenance.add_argument(
        "--source-dir", type=Path, help="Optionally read the revision from a source checkout"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        data = import_outputs(
            args.input_dir,
            args.output_dir,
            version=args.version,
            generation_model=args.generation_model,
            source_dir=args.source_dir,
            source_revision=args.source_revision,
            stage3_log=args.stage3_log,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Prepared DDB dataset at {data}")


if __name__ == "__main__":
    main()
