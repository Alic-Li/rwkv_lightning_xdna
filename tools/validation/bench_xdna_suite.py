# SPDX-License-Identifier: Apache-2.0
"""Serial C++ hardware probes; JSONL on stdout, complete logs in --log-dir."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("kernels", type=Path)
p.add_argument("--host", type=Path, default=Path("build/host"))
p.add_argument("--log-dir", type=Path, required=True)
a = p.parse_args()
a.log_dir.mkdir(parents=True, exist_ok=True)
cases = [
    (f"gemv-{k}-{n}", "bench_xdna_fp16", ["bf16", str(k), str(n)])
    for k, n in [(2048, 2048), (2048, 8192), (8192, 2048), (256, 2048), (2048, 65536)]
]
cases += [
    ("gemm", "rwkv-gemm-probe-test", []),
    ("normalization-elementwise-wkv", "rwkv-array-test", ["--bf16"]),
    ("dma-submission", "rwkv-stream-group-test", []),
]
for name, binary, args in cases:
    command = [str(a.host / binary), str(a.kernels), *args]
    result = subprocess.run(
        command,
        env=dict(os.environ, RWKV_XDNA_MICROBENCH="1"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log = a.log_dir / (name + ".log")
    log.write_text(result.stdout)
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    print(
        json.dumps(
            dict(
                case=name,
                command=command,
                status="passed" if result.returncode == 0 else "failed",
                log=str(log),
                records=records,
            )
        ),
        flush=True,
    )
    if result.returncode:
        sys.exit(result.returncode)
