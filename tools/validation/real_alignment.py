# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 PyTorch oracle for a real checkpoint; C++ dispatch only."""

import argparse
import ast
import gc
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from rwkv7_reference import ROOT, oracle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--prompt", default="The capital of France is")
p.add_argument(
    "--output", type=Path, default=ROOT / "reports/runs/decode-opt/real-oracle"
)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
vocab = {}
for line in (ROOT / "assets/rwkv_vocab_v20230424.txt").read_text().splitlines():
    first, last = line.index(" "), line.rindex(" ")
    value = ast.literal_eval(line[first + 1 : last])
    if isinstance(value, str):
        value = value.encode()
    vocab[value] = int(line[:first])
raw = args.prompt.encode()
tokens = []
while raw:
    piece = next(
        (raw[:n] for n in range(min(len(raw), 128), 0, -1) if raw[:n] in vocab), None
    )
    if piece is None:
        raise ValueError("unencodable input")
    tokens.append(vocab[piece])
    raw = raw[len(piece) :]
w = torch.load(args.model, map_location="cpu", weights_only=True)
expected = oracle(w, tokens)
del w
gc.collect()
np.save(args.output / "reference.npy", expected)
records = []
for backend, decode in [("cpu", "graph"), ("npu", "resident")]:
    output = args.output / f"{backend}.bin"
    cmd = [
        str(ROOT / "build/host/rwkv-cli"),
        "--model",
        str(args.model),
        "--backend",
        backend,
        "--decode",
        decode,
        "--prefill",
        "decode",
        "--tokens",
        ",".join(map(str, tokens)),
        "--dump-logits",
        str(output),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    (args.output / f"{backend}.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    actual = np.fromfile(output, np.float32).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=2e-4)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    record = dict(
        backend=backend,
        tokens=tokens,
        max_abs=float(abs(actual - expected).max()),
        relative_l2=float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
        greedy=actual.argmax(-1).tolist(),
        atol=1e-3,
        rtol=2e-4,
    )
    records.append(record)
    print(json.dumps(record), flush=True)
(args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
