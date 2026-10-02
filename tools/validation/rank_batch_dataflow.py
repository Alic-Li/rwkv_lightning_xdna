# SPDX-License-Identifier: Apache-2.0
"""Depth-one join/broadcast/release simulation of batched low-rank branches."""

import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ml_dtypes import bfloat16
from rkv_dataflow import Fifo


def main(count, first_cores=4, second_cores=8):
    first_stripe, second_stripe = 256 // first_cores, 2048 // second_cores
    rng = np.random.default_rng(103)
    x = rng.integers(-4, 5, (6, 2048)).astype(np.float32) / 16
    w1 = rng.integers(-4, 5, (count, 256, 2048)).astype(np.float32) / 256
    w2 = rng.integers(-4, 5, (count, 2048, 256)).astype(np.float32) / 256
    raw_out = np.empty((count, 256), np.float32)
    output = np.empty((count, 2048), np.float32)
    raw = [Fifo(1) for _ in range(first_cores)]
    act = [Fifo(1) for _ in range(first_cores)]
    broad = [Fifo(1) for _ in range(second_cores)]
    acknowledgements = queue.Queue()

    def nonlinear(v, p):
        return np.tanh(v) if p == 0 else 1 / (1 + np.exp(-v)) if p == 2 else v

    def first(worker):
        for p, slot in enumerate([1, 4, 5, 3][:count]):
            raw[worker].reserve()
            act[worker].reserve()
            out = np.zeros(first_stripe, np.float32)
            for row in range(first_stripe // 16):
                start = worker * first_stripe + row * 16
                for col in range(8):
                    out[row * 16 : (row + 1) * 16] += (
                        w1[p, start : start + 16, col * 256 : (col + 1) * 256]
                        * x[slot, col * 256 : (col + 1) * 256]
                    ).sum(axis=1)
            raw[worker].publish(out)
            act[worker].publish(nonlinear(out, p))

    def raw_drain():
        for p in range(count):
            raw_out[p] = np.concatenate([q.acquire() for q in raw])
            for q in raw:
                q.release()

    def broadcast():
        for _ in range(count):
            value = np.concatenate([q.acquire() for q in act])
            for q in broad:
                q.put(value)
            # The single joined object remains locked until all downstream
            # consumers release it, exactly like the multicast ObjectFifo.
            for _ in range(second_cores):
                acknowledgements.get(timeout=10)
            for q in act:
                q.release()

    def second(worker):
        for p in range(count):
            value = broad[worker].acquire().astype(bfloat16).astype(np.float32)
            for row in range(second_stripe // 16):
                start = worker * second_stripe + row * 16
                output[p, start : start + 16] = (w2[p, start : start + 16] * value).sum(
                    axis=1
                )
            broad[worker].release()
            acknowledgements.put(True)

    with ThreadPoolExecutor(max_workers=first_cores + second_cores + 2) as pool:
        tasks = [pool.submit(first, i) for i in range(first_cores)]
        tasks += [pool.submit(second, i) for i in range(second_cores)]
        tasks += [pool.submit(raw_drain), pool.submit(broadcast)]
        for t in tasks:
            t.result(timeout=30)
    raw_ref = np.einsum(
        "prk,pk->pr", w1.astype(np.float64), x[[1, 4, 5, 3][:count]].astype(np.float64)
    )
    np.testing.assert_array_equal(raw_out, raw_ref)
    active_ref = (
        np.stack([nonlinear(v, p) for p, v in enumerate(raw_ref)])
        .astype(bfloat16)
        .astype(np.float64)
    )
    expected = np.einsum("prk,pk->pr", w2.astype(np.float64), active_ref)
    np.testing.assert_allclose(output, expected, atol=1e-7, rtol=1e-6)
    print(
        f"rank batch {count}: depth-one join, {second_cores}-way broadcast and numeric oracle passed"
    )


if __name__ == "__main__":
    main(3)
    main(4)
