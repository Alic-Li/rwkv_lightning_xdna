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
    residual = rng.integers(-4, 5, 2048).astype(np.float32) / 16
    weights = [Fifo(2 if i < 4 else 1) for i in range(12)]
    raw = [Fifo(1) for _ in range(4)]
    act = [Fifo(1) for _ in range(4)]
    broad = [Fifo(1) for _ in range(8)]
    value = [Fifo(1) for _ in range(8)]
    groups = [Fifo(1) for _ in range(2)]
    final = Fifo(1)
    ack = queue.Queue()
    raw_out = np.empty(8192, np.float32)
    result = np.empty(4096, np.float32)

    def feed(i):
        matrix = (
            w1[i * 2048 : (i + 1) * 2048]
            if i < 4
            else w2[(i - 4) * 256 : (i - 3) * 256]
        )
        for r in range(0, matrix.shape[0], 16):
            for k in range(0, matrix.shape[1], 256):
                weights[i].put(matrix[r : r + 16, k : k + 256])

    def key(i):
        raw[i].reserve()
        act[i].reserve()
        out = np.zeros(2048, np.float32)
        for r in range(128):
            for k in range(8):
                w = weights[i].acquire()
                out[r * 16 : (r + 1) * 16] += (w * x[k * 256 : (k + 1) * 256]).sum(
                    axis=1
                )
                weights[i].release()
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
        for _ in range(8):
            ack.get(timeout=30)
        for q in act:
            q.release()

    def val(i):
        a = broad[i].acquire()
        value[i].reserve()
        out = np.zeros(256, np.float32)
        for r in range(16):
            for k in range(32):
                w = weights[4 + i].acquire()
                out[r * 16 : (r + 1) * 16] += (w * a[k * 256 : (k + 1) * 256]).sum(
                    axis=1
                )
                weights[4 + i].release()
        broad[i].release()
        ack.put(True)
        value[i].publish(out)

    def relay(i):
        qs = value[i * 4 : (i + 1) * 4]
        v = np.concatenate([q.acquire() for q in qs])
        groups[i].reserve()
        for q in qs:
            q.release()
        groups[i].publish(v)

    def add():
        v = np.concatenate([q.acquire() for q in groups])
        final.reserve()
        for q in groups:
            q.release()
        final.publish(np.concatenate([v, v + residual]))

    def drain():
        result[:] = final.acquire()
        final.release()

    with ThreadPoolExecutor(max_workers=30) as pool:
        tasks = (
            [pool.submit(feed, i) for i in range(12)]
            + [pool.submit(key, i) for i in range(4)]
            + [pool.submit(val, i) for i in range(8)]
        )
        tasks += (
            [pool.submit(drain_raw), pool.submit(broadcast)]
            + [pool.submit(relay, i) for i in range(2)]
            + [pool.submit(add), pool.submit(drain)]
        )
        for t in tasks:
            t.result(timeout=60)
    expected_raw = w1.astype(np.float64) @ x
    np.testing.assert_array_equal(raw_out, expected_raw)
    expected = w2.astype(np.float64) @ (np.maximum(expected_raw, 0) ** 2).astype(
        bfloat16
    ).astype(np.float64)
    np.testing.assert_allclose(result[:2048], expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(result[2048:], expected + residual, atol=1e-7, rtol=1e-6)
    print(
        "FFN: depth-two/one weights, four-way activation join, eight-way multicast, relay joins and oracle passed"
    )


if __name__ == "__main__":
    main()
