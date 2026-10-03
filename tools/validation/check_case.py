# SPDX-License-Identifier: Apache-2.0
"""Offline numerical experiment: compare C++ output dumps to upstream contracts."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compile"))
from kernel_case import ROOT, TEST_KERNEL_ROOT, find_case, inputs_for, kd, np


def check(identifier):
    if identifier == "cascade-mm":
        directory = TEST_KERNEL_ROOT / "cascade-mm"
        expected = np.fromfile(directory / "expected.bin", dtype=np.int16)
        for repetition in range(3):
            got = np.fromfile(directory / f"output-4.bin.{repetition}", dtype=np.int16)
            np.testing.assert_array_equal(got, expected)
        return dict(
            status="passed",
            case="cascade_mm/16x16x16/int16/pair",
            n_checked=256,
            repetitions=3,
        )
    case = find_case(identifier)
    fn = case.fn()
    directory = TEST_KERNEL_ROOT / identifier
    manifest = json.loads((directory / "manifest.json").read_text())
    # Regenerate typed logical inputs from the pinned case and seed.
    inputs = inputs_for(case, "random", np.random.default_rng(1000))
    expected = fn.expected(inputs, scalars=case.scalars)
    outputs = [b for b in manifest["buffers"] if b["direction"] == "out"]
    dtypes = fn.output_dtype()
    dtypes = dtypes if isinstance(dtypes, tuple) else (dtypes,)
    verdicts = []
    for repetition in range(manifest["repetitions"]):
        arrays = tuple(
            np.fromfile(directory / (buffer["file"] + f".{repetition}"), dtype=dtype)
            for buffer, dtype in zip(outputs, dtypes)
        )
        for buffer, array in zip(outputs, arrays):
            if array.nbytes != buffer["bytes"]:
                raise ValueError("NPU output file has incorrect size")
        got = arrays if len(arrays) > 1 else arrays[0]
        got, overrun = kd.strip_guard(fn, got, calls=case.calls)
        if np.any(np.asarray(overrun)):
            raise AssertionError(f"Output guard overwritten: {overrun}")
        verdict = fn.judge(
            got, expected, calls=case.calls, inputs=inputs, scalars=case.scalars
        )
        if not verdict:
            raise AssertionError(verdict.detail)
        verdicts.append(
            dict(n_checked=verdict.n_checked, max_abs_err=verdict.max_abs_err)
        )
    return dict(status="passed", case=case.name, repetitions=verdicts)


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1])))
