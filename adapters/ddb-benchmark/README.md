# DDB benchmark adapter

This adapter includes the DDB functional evaluator, dataset, and available upload fixtures. The bundled dataset contains 100 tasks and 3,396 atomic checkpoints. Task generation and custom snapshot creation use these packaged resources by default and require no upstream clone or Git installation. Generating tasks does not make LLM calls.

Run these commands from the Harbor repository root:

```bash
uv run --project adapters/ddb-benchmark ddb-benchmark \
  --output-dir datasets/ddb-benchmark
```

The bundled dataset has seven missing upstream fixtures across S1-038, S1-084, and S1-089. Task generation reports them; verification requires those files.

## Import generated Stage 1–3 outputs

Import a completed Stage 1–3 output bundle from any location, such as an extracted archive. Only the generated artifacts and fixture files are needed:

```bash
uv run --project adapters/ddb-benchmark ddb-benchmark-import \
  --input-dir /path/to/stage123_outputs \
  --version long \
  --output-dir datasets/ddb-generated
```

The input can be the pipeline output root, a `long` or `lite` directory, or a single query directory. If both versions are present, `--version` is required. The importer reads `query.json`, `stage2_raw.json` (a JSON object or a `text` wrapper containing JSON, including Markdown fences or surrounding prose), `stage3_guideline_raw.json`, and the final `stage3_raw.json`. The final artifact must match the guideline and fixture declarations. Fixture bytes must match the final artifact's Base64 `generated_files` payloads. This detects stale fixtures left behind by failed reruns or incomplete file materialization.

When the standard sibling `<version>_stage3.jsonl` run log is present, the importer also requires a matching successful entry for each selected query. Use `--stage3-log PATH` when upstream's `--out-dir` or `--out-jsonl` options put the log elsewhere, or when importing relocated outputs. The log's latest entry for a query takes precedence, including errors. Without a run log, the importer validates artifact consistency but cannot detect a failed rerun that left all previous artifacts unchanged. Import only after the upstream generation run has finished.

The import produces:

- `query.jsonl`: validated seed records, preserving checkpoint and query metadata.
- `fixtures/`: files collected under their original relative fixture paths.
- `query.provenance.json`: input hashes, dataset checksum, prompt version, and optional source revision, generation model, and checked Stage 3 log checksum.

Use `--generation-model MODEL` and `--source-revision REVISION` to record the model and revision supplied with the exported artifacts, if known. These options do not require source files or Git. Unknown values remain unset. For maintainers who already have a checkout, `--source-dir` can optionally read its Git revision instead of supplying `--source-revision`.

Malformed or incomplete records, duplicate task IDs, missing fixtures, escaping paths, and conflicting fixture contents at the same relative path fail the import. Identical shared fixtures are accepted. Existing output requires `--overwrite` and is preserved if validation fails.

Generate tasks directly from the import:

```bash
uv run --project adapters/ddb-benchmark ddb-benchmark \
  --data datasets/ddb-generated/query.jsonl \
  --fixtures-dir datasets/ddb-generated/fixtures \
  --output-dir datasets/ddb-generated-tasks
```

This uses the bundled evaluator. The optional `--source-dir` argument selects another self-contained snapshot or an explicit source checkout; it is not needed for the bundled workflow.

## Select a dataset in the evaluation scripts

`eval-claude.sh` and `eval-glm.sh` accept a leading `--ddb-mode bundled|long|lite` option, or the `DDB_MODE` environment variable. The option takes precedence over the environment variable. The default is `bundled`, which uses `datasets/ddb-benchmark`; `long` and `lite` use `datasets/ddb-benchmark-long` and `datasets/ddb-benchmark-lite`, respectively. These modes select already-generated tasks and do not change evaluator prompts, models, or checkpoint scheduling.

Prepare each desired mode from completed exported artifacts, running these commands from the Harbor repository root (replace `long` with `lite` for that dataset):

```bash
uv run --project adapters/ddb-benchmark ddb-benchmark-import \
  --input-dir /path/to/stage123_outputs \
  --version long \
  --output-dir datasets/ddb-import-long

uv run --project adapters/ddb-benchmark ddb-benchmark \
  --data datasets/ddb-import-long/query.jsonl \
  --fixtures-dir datasets/ddb-import-long/fixtures \
  --output-dir datasets/ddb-benchmark-long
```

Using the scripts' existing launch layout (a working directory containing `venv/` and the `harbor/` checkout), run:

```bash
bash harbor/eval-claude.sh --ddb-mode long --n-tasks 8
bash harbor/eval-glm.sh --ddb-mode lite --n-tasks 8
DDB_MODE=long bash harbor/eval-glm.sh --n-tasks 8
```

Place `--ddb-mode` before any Harbor options; remaining arguments are forwarded to `harbor run`. Configure the scripts' endpoint and credential values as usual. Selecting a missing mode directory fails before evaluation starts. Neither mode requires a local upstream clone.

## Snapshot a custom dataset

Create a self-contained snapshot with the bundled evaluator and imported data:

```bash
uv run --project adapters/ddb-benchmark ddb-benchmark-snapshot \
  --data datasets/ddb-generated/query.jsonl \
  --fixtures-dir datasets/ddb-generated/fixtures \
  --output-dir datasets/ddb-generated-snapshot

uv run --project adapters/ddb-benchmark ddb-benchmark \
  --source-dir datasets/ddb-generated-snapshot \
  --output-dir datasets/ddb-snapshot-tasks
```

Without `--data` or `--fixtures-dir`, snapshot creation uses the bundled dataset and fixtures. Custom inputs are stored at the snapshot's standard internal paths, so a snapshot needs no path overrides when generating tasks. The manifest preserves the evaluator's recorded upstream revision, records the original dataset filename, and hashes all bundled files. An adjacent `<dataset-stem>.provenance.json` file is carried into the snapshot and into generated task provenance; its dataset checksum is checked when present. Snapshot creation reports missing fixture paths in its manifest, matching the existing behavior for the original dataset.
