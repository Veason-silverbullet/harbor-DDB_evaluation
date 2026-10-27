"""Generate DDB tasks from the bundled snapshot or an explicit source checkout."""

import argparse
import logging
from pathlib import Path
from ddb_benchmark.adapter import DDBBenchmarkAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Override the bundled benchmark snapshot with a DDB_evaluation checkout",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--fixtures-dir", type=Path)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        tasks = DDBBenchmarkAdapter(
            args.source_dir,
            args.output_dir,
            data_path=args.data,
            fixtures_dir=args.fixtures_dir,
        ).prepare_tasks(args.task_ids, args.limit, args.overwrite)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Prepared {len(tasks)} DDB tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
