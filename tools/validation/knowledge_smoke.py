# SPDX-License-Identifier: Apache-2.0
"""Serial CPU/NPU greedy generation comparison; short knowledge smoke tests."""

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", type=Path, required=True)
p.add_argument(
    "--output", type=Path, default=ROOT / "reports/runs/decode-opt/knowledge"
)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
records = []
for name, prompt in [
    ("france", "The capital of France is"),
    ("france_qa", "Question: What is the capital of France?\nAnswer:"),
    ("planet", "Question: What is the largest planet in our solar system?\nAnswer:"),
    ("water", "Question: What is the chemical formula of water?\nAnswer:"),
]:
    outputs = {}
    for backend in ["cpu", "npu"]:
        cmd = [
            str(ROOT / "build/host/rwkv-cli"),
            "--model",
            str(a.model),
            "--backend",
            backend,
            "--decode",
            "resident" if backend == "npu" else "graph",
            "--prefill",
            "decode",
            "--prompt",
            prompt,
            "--max-tokens",
            "32",
            "--top-k",
            "1",
        ]
        r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=600)
        (a.output / f"{name}-{backend}.txt").write_text(r.stdout)
        (a.output / f"{name}-{backend}.log").write_text(r.stderr)
        if r.returncode:
            raise RuntimeError(r.stderr)
        outputs[backend] = r.stdout
        timing = re.search(r"Generated (\d+) tokens in ([\d.]+) s", r.stderr)
        record = dict(
            prompt=name,
            backend=backend,
            generated=int(timing[1]),
            seconds=float(timing[2]),
            text=r.stdout,
        )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if outputs["cpu"] != outputs["npu"]:
        raise AssertionError(f"greedy generation divergence: {name}")
(a.output / "results.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2) + "\n"
)
