# SPDX-License-Identifier: Apache-2.0
"""Compile in parallel, dispatch serially in C++, then check dumps offline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compile"))
from kernel_case import TEST_KERNEL_ROOT, case_id, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / "build/test/xdna-run")
    parser.add_argument("--tier", choices=["smoke", "all"], default="smoke")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--case", action="append", help="Restrict to case IDs")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.compile_only and args.run_only:
        parser.error("--compile-only and --run-only are mutually exclusive")
    report_dir = ROOT / "reports/runs" / args.tier
    report_dir.mkdir(parents=True, exist_ok=True)
    cases = selected(args.tier)
    cases.append(
        SimpleNamespace(factory="cascade_mm", name="cascade_mm/16x16x16/int16/pair")
    )
    if args.case:
        cases = [c for c in cases if case_id(c) in args.case]
    if not cases:
        parser.error("no cases selected")

    def compile_one(case):
        identifier = case_id(case)
        if args.run_only:
            return (TEST_KERNEL_ROOT / identifier / "manifest.json").is_file()
        log = report_dir / f"{identifier}.compile.log"
        try:
            with log.open("w") as stream:
                command = (
                    [sys.executable, str(ROOT / "tools/compile/cascade.py")]
                    if identifier == "cascade-mm"
                    else [
                        sys.executable,
                        str(ROOT / "tools/compile/kernel_case.py"),
                        "--case",
                        identifier,
                    ]
                )
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=240,
                )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {case_id(c): pool.submit(compile_one, c) for c in cases}
        for index, case in enumerate(cases):
            identifier = case_id(case)
            record = dict(
                id=identifier,
                case=case.name,
                factory=case.factory,
                utc=datetime.now(timezone.utc).isoformat(),
            )
            status_path = report_dir / f"{identifier}.json"
            if not futures[identifier].result():
                record["status"] = "compile_failed"
            elif args.compile_only:
                record["status"] = "compiled"
            else:
                command = [
                    str(args.runner.resolve()),
                    str(TEST_KERNEL_ROOT / identifier / "manifest.json"),
                ]
                try:
                    run = subprocess.run(
                        command, capture_output=True, text=True, timeout=60
                    )
                    (report_dir / f"{identifier}.run.log").write_text(
                        run.stdout + run.stderr
                    )
                    if run.returncode:
                        record.update(
                            status="runtime_failed", detail=run.stderr[-3000:]
                        )
                    else:
                        record["execution"] = json.loads(
                            run.stdout.strip().splitlines()[-1]
                        )
                        check = subprocess.run(
                            [
                                sys.executable,
                                str(ROOT / "tools/validation/check_case.py"),
                                identifier,
                            ],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        (report_dir / f"{identifier}.check.log").write_text(
                            check.stdout + check.stderr
                        )
                        record.update(
                            status=(
                                "passed"
                                if check.returncode == 0
                                else "verification_failed"
                            )
                        )
                        if check.returncode == 0:
                            record["verification"] = json.loads(
                                check.stdout.strip().splitlines()[-1]
                            )
                        else:
                            record["detail"] = check.stderr[-3000:]
                except subprocess.TimeoutExpired:
                    record["status"] = "timeout"
            status_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            (report_dir / "summary.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(
                f"[{index+1}/{len(cases)}] {record['status']}: {case.name}", flush=True
            )
    return int(any(r["status"] not in ("passed", "compiled") for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
