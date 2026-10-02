# SPDX-License-Identifier: Apache-2.0
"""Depth-bounded key/activation multicast/value/hierarchical residual join mock."""

import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ml_dtypes import bfloat16
from rkv_dataflow import Fifo


def main():
    rng = np.random.default_rng(571)
    x = rng.integers(-4, 5, 2048).astype(np.float32) / 16
    w1 = rng.integers(-4, 5, (8192, 2048)).astype(np.float32) / 256
    w2 = rng.integers(-4, 5, (2048, 8192)).astype(np.float32) / 256
    residual = x.copy()
    previous = rng.normal(0, 0.125, 2048).astype(np.float32)
    gamma = rng.normal(1, 0.125, 2048).astype(np.float32)
    beta = rng.normal(0, 0.125, 2048).astype(np.float32)
    coefficient = rng.normal(0, 0.125, 2048).astype(np.float32)
    normalized = (x - x.mean()) / np.sqrt(x.var() + 1e-5) * gamma + beta
    expected_mixed = normalized + (previous - normalized) * coefficient
    norm_input, norm_pair, mixed = [Fifo(1) for _ in range(3)]
    key_x = [Fifo(1) for _ in range(4)]
    key_ack = queue.Queue()

    def feed_norm():
        norm_input.put(np.stack([x, gamma, beta, previous]))

    def norm_core():
        p = norm_input.acquire()
        norm_pair.reserve()
        n = (p[0] - p[0].mean()) / np.sqrt(p[0].var() + 1e-5) * p[1] + p[2]
        norm_input.release()
        norm_pair.publish(np.stack([n, p[3]]))

    def mix_core():
        p = norm_pair.acquire()
        mixed.reserve()
        v = p[0] + (p[1] - p[0]) * coefficient
        norm_pair.release()
        mixed.publish(v)

    def input_broadcast():
        v = mixed.acquire()
        np.testing.assert_array_equal(v, expected_mixed)
        for q in key_x:
            q.put(v.astype(bfloat16).astype(np.float32))
        for _ in key_x:
            key_ack.get(timeout=30)
        mixed.release()

    weights = [Fifo(2 if i < 4 else 1) for i in range(8)]
    raw = [Fifo(1) for _ in range(4)]
    act = [Fifo(1) for _ in range(4)]
    broad = [Fifo(1) for _ in range(4)]
    value = [Fifo(1) for _ in range(4)]
    final = Fifo(1)
    ack = queue.Queue()
    raw_out = np.empty(8192, np.float32)
    result = np.empty(4096, np.float32)

    def feed(i):
        matrix = (
            w1[i * 2048 : (i + 1) * 2048]
            if i < 4
            else w2[(i - 4) * 512 : (i - 3) * 512]
        )
        for r in range(0, matrix.shape[0], 16):
            for k in range(0, matrix.shape[1], 256):
                weights[i].put(matrix[r : r + 16, k : k + 256])

    def key(i):
        xv = key_x[i].acquire()
        raw[i].reserve()
        act[i].reserve()
        out = np.zeros(2048, np.float32)
        for r in range(128):
            for k in range(8):
                w = weights[i].acquire()
                out[r * 16 : (r + 1) * 16] += (w * xv[k * 256 : (k + 1) * 256]).sum(
                    axis=1
                )
                weights[i].release()
        key_x[i].release()
        key_ack.put(True)
        raw[i].publish(out)
        act[i].publish(np.maximum(out, 0) ** 2)

    def drain_raw():
        raw_out[:] = np.concatenate([q.acquire() for q in raw])
        for q in raw:
            q.release()

    def broadcast():
        a = (
            np.concatenate([q.acquire() for q in act])
            .astype(bfloat16)
            .astype(np.float32)
        )
        for q in broad:
            q.put(a)
        for _ in range(4):
            ack.get(timeout=30)
        for q in act:
            q.release()

    def val(i):
        a = broad[i].acquire()
        value[i].reserve()
        out = np.zeros(512, np.float32)
        for r in range(32):
            for k in range(w2.shape[1] // 256):
                w = weights[4 + i].acquire()
                out[r * 16 : (r + 1) * 16] += (w * a[k * 256 : (k + 1) * 256]).sum(
                    axis=1
                )
                weights[4 + i].release()
        broad[i].release()
        ack.put(True)
        value[i].publish(out)

    def add():
        v = np.concatenate([q.acquire() for q in value])
        final.reserve()
        for q in value:
            q.release()
        final.publish(np.concatenate([v, v + residual]))

    def drain():
        result[:] = final.acquire()
        final.release()

    with ThreadPoolExecutor(max_workers=30) as pool:
        tasks = (
            [pool.submit(feed, i) for i in range(8)]
            + [pool.submit(key, i) for i in range(4)]
            + [pool.submit(val, i) for i in range(4)]
        )
        tasks += (
            [pool.submit(drain_raw)]
            + [pool.submit(broadcast)]
            + [pool.submit(add), pool.submit(drain)]
            + [
                pool.submit(fn)
                for fn in [feed_norm, norm_core, mix_core, input_broadcast]
            ]
        )
        for t in tasks:
            t.result(timeout=60)
    expected_raw = w1.astype(np.float64) @ expected_mixed.astype(bfloat16).astype(
        np.float64
    )
    np.testing.assert_allclose(raw_out, expected_raw, atol=2e-5, rtol=2e-5)
    expected = w2.astype(np.float64) @ (np.maximum(raw_out, 0) ** 2).astype(
        bfloat16
    ).astype(np.float64)
    np.testing.assert_allclose(result[:2048], expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(result[2048:], expected + residual, atol=1e-7, rtol=1e-6)
    print(
        f"ChannelMix: bounded weight FIFOs, multicast, five-input join and oracle passed"
    )


if __name__ == "__main__":
    main()
