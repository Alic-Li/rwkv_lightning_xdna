#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize decoded AIE event intervals; never dispatches hardware.

Input is produced by aie.utils.trace.parse. Timestamps are device cycles, not
microseconds. Utilization is vector-issue activity on the traced cores, not a
percentage of device peak FLOPS. Stall categories may overlap.
"""
import argparse
import json
from collections import defaultdict


def summarize(events):
    names = {}
    per_core = defaultdict(list)
    for event in events:
        if event["name"] == "process_name":
            names[event["pid"]] = event["args"]["name"]
        elif event.get("ph") in ("B", "E"):
            per_core[event["pid"]].append(event)
    result = []
    for pid, core in sorted(per_core.items()):
        core.sort(key=lambda e: e["ts"])
        windows = []
        start = None
        for event in core:
            if event["ph"] != "B":
                continue
            if event["name"] == "INSTR_EVENT_0":
                if start is not None:
                    raise ValueError(f"Overlapping kernel markers on core {pid}")
                start = event["ts"]
            elif event["name"] == "INSTR_EVENT_1" and start is not None:
                windows.append((start, event["ts"]))
                start = None
        if not windows or start is not None:
            raise ValueError(f"Missing complete kernel markers on core {pid}")
        pending = {}
        active = defaultdict(int)
        for event in core:
            name, ts = event["name"], event["ts"]
            if name.startswith("INSTR_EVENT_"):
                continue
            if event["ph"] == "B":
                if name in pending:
                    raise ValueError(f"Unpaired begin for {name} on core {pid}")
                pending[name] = ts
            elif name in pending:
                begin = pending.pop(name)
                for lo, hi in windows:
                    active[name] += max(0, min(hi, ts) - max(lo, begin))
        # An event still active at trace termination covers the rest of any
        # complete marked window. This commonly happens with a final lock wait.
        for name, begin in pending.items():
            for lo, hi in windows:
                active[name] += max(0, hi - max(lo, begin))
        duration = sum(hi - lo for lo, hi in windows)
        if duration <= 0:
            raise ValueError(f"Invalid kernel duration on core {pid}")
        result.append({"core": names.get(pid, str(pid)),
                       "invocations": len(windows), "marked_cycles": duration,
                       "event_cycles": dict(active),
                       "event_fraction": {k: v / duration for k, v in active.items()}})
    if not result:
        raise ValueError("No core events in trace")
    return {"schema_version": 1, "cores": result,
            "scope": "Marked execution on selected cores; excludes waiting for initial activation. Fractions overlap; vector issue is not peak-compute utilization."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="Decoded trace JSON")
    args = parser.parse_args()
    with open(args.trace) as source:
        print(json.dumps(summarize(json.load(source)), indent=2))
