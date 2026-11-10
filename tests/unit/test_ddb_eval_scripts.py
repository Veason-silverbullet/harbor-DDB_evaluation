import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(params=["eval-claude.sh", "eval-glm.sh"])
def run_eval_script(request, tmp_path):
    script = Path(__file__).resolve().parents[2] / request.param
    activation = tmp_path / "venv/bin/activate"
    activation.parent.mkdir(parents=True)
    activation.write_text("")
    checkout = tmp_path / "harbor"
    checkout.mkdir()
    for mode in ("long", "lite"):
        (checkout / f"datasets/ddb-benchmark-{mode}").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$DDB_TEST_ARGS"\n')
    uv.chmod(0o755)
    capture = tmp_path / "args"

    def run(args=(), mode=None):
        env = dict(os.environ)
        env.pop("DDB_MODE", None)
        if mode is not None:
            env["DDB_MODE"] = mode
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["DDB_TEST_ARGS"] = str(capture)
        capture.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", str(script), *args], cwd=tmp_path, env=env, capture_output=True
        )
        forwarded = (
            capture.read_bytes().decode().split("\0")[:-1] if capture.exists() else []
        )
        return result.returncode, forwarded

    return run, checkout


@pytest.mark.parametrize(
    ("args", "mode", "dataset"),
    [
        ([], None, "ddb-benchmark"),
        (["--ddb-mode", "long"], None, "ddb-benchmark-long"),
        (["--ddb-mode", "lite"], None, "ddb-benchmark-lite"),
        ([], "lite", "ddb-benchmark-lite"),
        (["--ddb-mode", "long"], "lite", "ddb-benchmark-long"),
        (["--ddb-mode", "bundled"], "long", "ddb-benchmark"),
    ],
)
def test_dataset_selection(run_eval_script, args, mode, dataset):
    run, _ = run_eval_script
    extra = ["--n-tasks", "3", "--job-name", "run with spaces"]
    code, forwarded = run([*args, *extra], mode)
    assert code == 0
    assert forwarded[forwarded.index("--path") + 1] == f"datasets/{dataset}"
    assert "--ddb-mode" not in forwarded
    assert forwarded[-len(extra) :] == extra


@pytest.mark.parametrize("args", [["--ddb-mode"], ["--ddb-mode", "invalid"]])
def test_invalid_mode_stops_before_evaluation(run_eval_script, args):
    run, _ = run_eval_script
    assert run(args) == (2, [])


def test_missing_dataset_stops_before_evaluation(run_eval_script):
    run, checkout = run_eval_script
    (checkout / "datasets/ddb-benchmark-long").rmdir()
    assert run(["--ddb-mode", "long"]) == (2, [])
