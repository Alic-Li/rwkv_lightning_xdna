#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply predeclared screening gates to C++ NPU accuracy JSONL; no dispatch."""
import json
import sys
from pathlib import Path


def evaluate(path):
    entries = [json.loads(line) for line in Path(path).read_text().splitlines()]
    summaries = [e for e in entries if e.get("type") == "summary"]
    if len(summaries) != 1:
        raise ValueError("Exactly one completed accuracy summary required")
    summary = summaries[0]
    tokens = [e for e in entries if e.get("type") == "token"]
    states = [e["state_error"]["relative_l2"] for e in tokens if "state_error" in e]
    checks = {
        "complete_128_steps": summary["steps"] == 128 and len(tokens) == 128,
        "precision": summary["precision"] == "w8a8_ffn_int32_dot_fp32_partial_others_bf16_fp32_state",
        "mean_kl": summary["mean_kl"] <= 0.01,
        "max_kl": summary["max_kl"] <= 0.1,
        "top1_agreement": summary["top1_agreement"] >= 0.9,
        "state_l2": len(states) == 4 and all(v is not None and v <= 0.1 for v in states),
        "reset_branch": summary["reset_and_branch"] == "passed",
    }
    return {"passed": all(checks.values()), "checks": checks,
            "summary": summary, "checkpoint_state_relative_l2": states,
            "scope": "Experimental screen only; not production quality certification"}


if __name__ == "__main__":
    result = evaluate(sys.argv[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
