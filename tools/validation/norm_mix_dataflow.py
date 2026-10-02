# SPDX-License-Identifier: Apache-2.0
"""Bounded norm-pair multicast and six-output mix/shift ownership mock."""

import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rkv_dataflow import Fifo


def main(count):
    rng = np.random.default_rng(583)
    x = rng.normal(size=2048).astype(np.float32)
    w = rng.normal(1, 0.1, 2048).astype(np.float32)
    b = rng.normal(0, 0.1, 2048).astype(np.float32)
    old = rng.normal(size=2048).astype(np.float32)
    coeff = rng.normal(size=(count, 2048)).astype(np.float32)
    expected_norm = (x - x.mean()) / np.sqrt(x.var() + 1e-5) * w + b
    expected = expected_norm + (old - expected_norm) * coeff
    inputs, pair, mixpair, hostpair, c, output = [Fifo(1) for _ in range(6)]
    ack = queue.Queue()
    actual = np.empty_like(coeff)

    def fill():
        inputs.put(np.stack([x, w, b, old]))
        for p in coeff:
            c.put(p)

    def norm():
        p = inputs.acquire()
        pair.reserve()
        value = (p[0] - p[0].mean()) / np.sqrt(p[0].var() + 1e-5) * p[1] + p[2]
        inputs.release()
        pair.publish(np.stack([value, p[3]]))

    def multicast():
        p = pair.acquire()
        mixpair.put(p)
        hostpair.put(p)
        ack.get(timeout=10)
        ack.get(timeout=10)
        pair.release()

    def host():
        p = hostpair.acquire()
        np.testing.assert_array_equal(p[0], expected_norm)
        hostpair.release()
        ack.put(True)

    def mix():
        p = mixpair.acquire()
        for _ in range(count):
            cv = c.acquire()
            output.reserve()
            value = np.stack([p[0] + (p[1] - p[0]) * cv, p[0]])
            c.release()
            output.publish(value)
        mixpair.release()
        ack.put(True)

    def drain():
        for i in range(count):
            p = output.acquire()
            actual[i] = p[0]
            old[:] = p[1]
            output.release()

    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = [pool.submit(fn) for fn in [fill, norm, multicast, host, mix, drain]]
        for t in tasks:
            t.result(timeout=30)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(old, expected_norm)
    print(
        f"norm_mix={count}: depth-one multicast, output/shift drains and oracle passed"
    )


if __name__ == "__main__":
    main(1)
    main(6)
