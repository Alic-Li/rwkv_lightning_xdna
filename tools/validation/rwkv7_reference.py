# SPDX-License-Identifier: Apache-2.0
"""Independent float64 PyTorch oracle; hardware is invoked only by C++ CLI."""

import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def synthetic():
    torch.manual_seed(703)
    c, n, h, v, rank, hidden = 128, 64, 2, 256, 16, 192
    w = {}

    def rand(name, shape, scale=0.06):
        w[name] = torch.randn(shape) * scale

    def norm(name):
        w[name + ".weight"] = 1 + torch.randn(c) * 0.05
        w[name + ".bias"] = torch.randn(c) * 0.02

    rand("emb.weight", (v, c), 0.5)
    rand("head.weight", (v, c))
    norm("ln_out")
    norm("blocks.0.ln0")
    for l in range(2):
        p = f"blocks.{l}."
        norm(p + "ln1")
        norm(p + "ln2")
        norm(p + "att.ln_x")
        for name in ("r", "w", "k", "v", "a", "g"):
            w[p + "att.x_" + name] = torch.rand(1, 1, c)
        for name in ("w", "a", "g", "v"):
            if l == 0 and name == "v":
                continue
            rand(p + f"att.{name}1", (c, rank))
            rand(p + f"att.{name}2", (rank, c))
            if name != "g":
                rand(p + f"att.{name}0", (1, 1, c), 0.4)
        for name in ("receptance", "key", "value", "output"):
            rand(p + f"att.{name}.weight", (c, c))
        w[p + "att.k_k"] = 1 + torch.randn(1, 1, c) * 0.1
        w[p + "att.k_a"] = torch.rand(1, 1, c)
        rand(p + "att.r_k", (h, n))
        w[p + "ffn.x_k"] = torch.rand(1, 1, c)
        rand(p + "ffn.key.weight", (hidden, c))
        rand(p + "ffn.value.weight", (c, hidden))
    return w


def oracle(weights, tokens):
    w = {k: v.double().squeeze() for k, v in weights.items()}
    c = w["emb.weight"].shape[1]
    h, n = w["blocks.0.att.r_k"].shape
    layers = sum(k.endswith(".ln1.weight") for k in w)
    shift = torch.zeros(layers, 2, c, dtype=torch.float64)
    state = torch.zeros(layers, h, n, n, dtype=torch.float64)
    results = []

    def norm(x, name):
        return F.layer_norm(x, (c,), w[name + ".weight"], w[name + ".bias"], 1e-5)

    for token in tokens:
        x = norm(w["emb.weight"][token], "blocks.0.ln0")
        for l in range(layers):
            p = f"blocks.{l}."
            xx = norm(x, p + "ln1")
            mixed = {
                name: xx + (shift[l, 0] - xx) * w[p + "att.x_" + name]
                for name in "rwkvag"
            }
            shift[l, 0] = xx
            r = w[p + "att.receptance.weight"] @ mixed["r"]
            k = w[p + "att.key.weight"] @ mixed["k"]
            v = w[p + "att.value.weight"] @ mixed["v"]
            decay = torch.exp(
                -np.exp(-0.5)
                * torch.sigmoid(
                    w[p + "att.w0"]
                    + torch.tanh(mixed["w"] @ w[p + "att.w1"]) @ w[p + "att.w2"]
                )
            )
            a = torch.sigmoid(
                w[p + "att.a0"] + (mixed["a"] @ w[p + "att.a1"]) @ w[p + "att.a2"]
            )
            g = torch.sigmoid(mixed["g"] @ w[p + "att.g1"]) @ w[p + "att.g2"]
            if l == 0:
                first_v = v.clone()
            else:
                gate = torch.sigmoid(
                    w[p + "att.v0"] + (mixed["v"] @ w[p + "att.v1"]) @ w[p + "att.v2"]
                )
                v = v + (first_v - v) * gate
            kk = F.normalize((k * w[p + "att.k_k"]).view(h, n), dim=-1, eps=1e-12)
            k = k * (1 + (a - 1) * w[p + "att.k_a"])
            # Independent matrix expression, explicit [key,value] orientation.
            ah = (-kk).unsqueeze(1)
            bh = (kk * a.view(h, n)).unsqueeze(2)
            state[l] = (
                decay.view(h, n, 1) * state[l]
                + bh @ (ah @ state[l])
                + k.view(h, n, 1) @ v.view(h, 1, n)
            )
            y = (r.view(h, 1, n) @ state[l]).reshape(c)
            y = F.group_norm(
                y.view(1, c, 1),
                h,
                w[p + "att.ln_x.weight"],
                w[p + "att.ln_x.bias"],
                64e-5,
            ).reshape(c)
            residual = (r.view(h, n) * k.view(h, n) * w[p + "att.r_k"]).sum(
                -1, keepdim=True
            ) * v.view(h, n)
            x = x + w[p + "att.output.weight"] @ ((y + residual.reshape(c)) * g)
            xx = norm(x, p + "ln2")
            f = xx + (shift[l, 1] - xx) * w[p + "ffn.x_k"]
            shift[l, 1] = xx
            x = (
                x
                + w[p + "ffn.value.weight"]
                @ torch.relu(w[p + "ffn.key.weight"] @ f).square()
            )
        results.append((w["head.weight"] @ norm(x, "ln_out")).numpy())
    return np.asarray(results)


def safetensors(path, weights):
    header = {}
    chunks = []
    offset = 0
    for name, tensor in weights.items():
        raw = tensor.float().contiguous().numpy().astype("<f4").tobytes()
        header[name] = dict(
            dtype="F32",
            shape=list(tensor.shape),
            data_offsets=[offset, offset + len(raw)],
        )
        chunks.append(raw)
        offset += len(raw)
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"".join(chunks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = ROOT / "build/tests/rwkv7"
    out.mkdir(parents=True, exist_ok=True)
    weights = synthetic()
    tokens = [1, 7, 3, 2, 19, 0, 1, 8, 29, 7, 3, 255, 9, 4, 18, 2]
    records = []
    fixtures = []
    for name, dtype in [
        ("f32", torch.float32),
        ("f16", torch.float16),
        ("bf16", torch.bfloat16),
    ]:
        ws = {k: v.to(dtype) for k, v in weights.items()}
        # Noncontiguous storage verifies PTH stride handling, not only dense tensors.
        ws["head.weight"] = ws["head.weight"].t().contiguous().t()
        path = out / (name + ".pth")
        torch.save(ws, path)
        fixtures.append((path, ws))
    safe = out / "f32.safetensors"
    safetensors(safe, weights)
    fixtures.append((safe, weights))
    for path, ws in fixtures:
        expected = oracle(ws, tokens)
        for backend in (["cpu", "npu"] if args.npu else ["cpu"]):
            dump = out / (path.name + "." + backend + ".bin")
            cmd = [
                str(ROOT / "build/host/rwkv-cli"),
                "--model",
                str(path),
                "--backend",
                backend,
                "--tokens",
                ",".join(map(str, tokens)),
                "--dump-logits",
                str(dump),
                "--threads",
                "2",
            ]
            result = subprocess.run(
                cmd, cwd=ROOT, text=True, capture_output=True, timeout=180
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            actual = np.fromfile(dump, np.float32).reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
            error = float(np.max(np.abs(actual - expected)))
            record = dict(
                fixture=path.name,
                backend=backend,
                tokens=len(tokens),
                max_abs_error=error,
            )
            print(record, flush=True)
            records.append(record)
    # Invalid archives/checkpoints fail cleanly in the C++ loader.
    for name, content in [
        ("truncated.safetensors", b"\xff" * 8),
        ("invalid.pth", b"not a zip"),
    ]:
        path = out / name
        path.write_bytes(content)
        result = subprocess.run(
            [
                str(ROOT / "build/host/rwkv-cli"),
                "--model",
                str(path),
                "--backend",
                "cpu",
                "--tokens",
                "1",
            ],
            cwd=ROOT,
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 1:
            raise RuntimeError(f"Invalid checkpoint did not fail cleanly: {name}")
    (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
