# SPDX-License-Identifier: Apache-2.0
"""Bounded acquire/release simulation of rwkv7_rkv's exact worker schedule."""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class Fifo:
    def __init__(self, depth=2):
        self.slots = threading.BoundedSemaphore(depth)
        self.queue = queue.Queue()

    def reserve(self):
        if not self.slots.acquire(timeout=10):
            raise RuntimeError("FIFO reserve timeout")

    def publish(self, value):
        self.queue.put(value)

    def put(self, value):
        self.reserve()
        self.publish(value)

    def acquire(self):
        return self.queue.get(timeout=10)

    def release(self):
        self.slots.release()


def main():
    rng = np.random.default_rng(91)
    # Dyadic BF16-exact inputs ensure this topology test has no rounding
    # ambiguity. Independent random floating-point math is tested on hardware.
    x = rng.integers(-4, 5, (6, 2048)).astype(np.float32) / 16
    w = rng.integers(-8, 9, (3, 2048, 2048)).astype(np.float32) / 256
    output = np.empty((3, 2048), np.float32)

    def feed_x(q):
        for projection in [0, 2, 3]:
            for _ in range(16):
                for col in range(8):
                    q.put(x[projection, col * 256 : (col + 1) * 256])

    def feed_w(q, worker):
        for projection in range(3):
            for row in range(16):
                for col in range(8):
                    r = worker * 256 + row * 16
                    q.put(w[projection, r : r + 16, col * 256 : (col + 1) * 256])

    def core(xq, wq, yq):
        for _ in range(3 * 16):
            yq.reserve()
            out = np.zeros(16, np.float32)
            for _ in range(8):
                a, b = xq.acquire(), wq.acquire()
                out += (b * a).sum(axis=1)
                xq.release()
                wq.release()
            yq.publish(out)

    def drain(q, worker):
        for projection in range(3):
            for row in range(16):
                r = worker * 256 + row * 16
                output[projection, r : r + 16] = q.acquire()
                q.release()

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = []
        for i in range(8):
            xq, wq, yq = Fifo(), Fifo(), Fifo()
            futures.extend(
                [
                    pool.submit(feed_x, xq),
                    pool.submit(feed_w, wq, i),
                    pool.submit(core, xq, wq, yq),
                    pool.submit(drain, yq, i),
                ]
            )
        for future in futures:
            future.result(timeout=30)
    expected = np.einsum(
        "pnk,pk->pn", w.astype(np.float64), x[[0, 2, 3]].astype(np.float64)
    )
    np.testing.assert_array_equal(output, expected)
    print(
        "RKV topology: 8 workers, depth 2, 3072 input/weight tiles, 384 output tiles; oracle passed"
    )


if __name__ == "__main__":
    main()
