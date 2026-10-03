# SPDX-License-Identifier: Apache-2.0
"""Write a compact source-coverage and hardware-validation snapshot."""

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compile"))
from kernel_case import ROOT, case_id, selected

identifiers = [case_id(c) for c in selected("all")] + ["cascade-mm"]
records = [
    json.loads((ROOT / "reports/runs/all" / f"{identifier}.json").read_text())
    for identifier in identifiers
]
by_id = {r["id"]: r for r in records}
source_root = ROOT / "third_party/mlir-aie"
coverage = {
    str(p.relative_to(source_root)): []
    for p in (source_root / "aie_kernels").rglob("*.cc")
}
for case in selected("all"):
    fn = case.fn()
    functions = [fn] + [init(fn) for _, init in fn.contract.initializers]
    if fn.contract.setup:
        functions.append(fn.contract.setup())
    for kernel in functions:
        source = str(Path(kernel.source_file).relative_to(source_root))
        if case_id(case) not in coverage[source]:
            coverage[source].append(case_id(case))
coverage["aie_kernels/linalg/cascade_mm.cc"].append("cascade-mm")
upstream = json.loads((ROOT / "third_party/SOURCES.json").read_text())
for path, digest in upstream["mlir_aie"]["sha256"].items():
    actual = hashlib.sha256((source_root / path).read_bytes()).hexdigest()
    if actual != digest:
        raise RuntimeError(f"Vendored source changed without provenance update: {path}")
source_records = []
for source, ids in sorted(coverage.items()):
    passed = [i for i in ids if by_id.get(i, {}).get("status") == "passed"]
    source_records.append(
        dict(source=source, covered=bool(passed), cases=ids, passed=len(passed))
    )
summary = dict(
    local_time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    kernel=platform.release(),
    xrt_examine=subprocess.check_output(["xrt-smi", "examine"], text=True),
    driver_workaround="amdxdna force_cmdlist=N; /etc/modprobe.d/rwkv-lightning-xdna.conf",
    runtime="C++17 / XRT; Python used only for compilation and offline reference experiments",
    mlir_aie=upstream["mlir_aie"]["version"],
    llvm_aie="22.0.0.2026091701+773413fb",
    statuses=dict(Counter(r["status"] for r in records)),
    source_count=len(source_records),
    covered_source_count=sum(r["covered"] for r in source_records),
    dispatch_count=sum(len(r.get("execution", {}).get("timings", [])) for r in records),
    limitations=[
        "One seeded random dataset per upstream configuration, three C++ dispatches each.",
        "295 generic cases include output guard checks; cascade pair checks numerical output only.",
        "Source coverage is not every C++ symbol, datatype, shape or architecture branch.",
        "Four upstream NPU1-only cases are excluded on AIE2P.",
        "This generic kernel sweep does not validate RWKV model inference or its fused kernels.",
    ],
    sources=source_records,
    cases=[{k: r[k] for k in ("id", "case", "factory", "status")} for r in records],
)
output = ROOT / f"reports/validation-{summary['local_time'][:10]}.json"
output.write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps(
        {
            k: summary[k]
            for k in (
                "statuses",
                "source_count",
                "covered_source_count",
                "dispatch_count",
            )
        }
    )
)
if any(not s["covered"] for s in source_records) or any(
    r["status"] != "passed" for r in records
):
    raise SystemExit(1)
