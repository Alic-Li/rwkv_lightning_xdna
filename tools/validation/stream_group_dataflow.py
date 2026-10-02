# SPDX-License-Identifier: Apache-2.0
"""Two packets through depth-one 2048-element producer / large consumer storage."""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rkv_dataflow import Fifo


def main(count):
    source = np.arange(2 * count * 2048, dtype=np.float32)
    result = np.empty_like(source)
    fi, mid, packet, fo = [Fifo(1) for _ in range(4)]

    def fill():
        for i in range(2 * count):
            fi.put(source[i * 2048 : (i + 1) * 2048])

    def producer():
        for _ in range(2 * count):
            x = fi.acquire()
            mid.reserve()
            v = x.copy()
            fi.release()
            mid.publish(v)

    def regroup():
        for _ in range(2):
            packet.reserve()
            parts = []
            for _ in range(count):
                parts.append(mid.acquire().copy())
                mid.release()
            packet.publish(np.concatenate(parts))

    def consumer():
        for _ in range(2):
            x = packet.acquire()
            for i in range(count):
                fo.put(x[i * 2048 : (i + 1) * 2048].copy())
            packet.release()

    def drain():
        for i in range(2 * count):
            result[i * 2048 : (i + 1) * 2048] = fo.acquire()
            fo.release()

    with ThreadPoolExecutor(max_workers=5) as pool:
        tasks = [pool.submit(fn) for fn in [fill, producer, regroup, consumer, drain]]
        for t in tasks:
            t.result(timeout=30)
    np.testing.assert_array_equal(source, result)
    print(
        f"stream group {count}: two packets, depth-one ownership, ordering and oracle passed"
    )


if __name__ == "__main__":
    main(3)
    main(4)
