"""Repeatable wall-clock samples for 3D spatial layers.

Run from any directory: python tools/benchmark_spatial.py
These are local measurements, not a guarantee of performance on other PCs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from life_lab.instruments import spatial_field
from life_lab.world import Ecosystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Замер пространственных слоёв Life")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 1000:
        parser.error("--repeats должен быть в пределах 3..1000")
    for width, height in ((160, 100), (300, 180), (700, 500)):
        world = Ecosystem(width, height, seed=119)
        world.randomize(.15, seed=119)
        print(f"{width}x{height}")
        for layer in ("Плотность", "Влага", "Глубина родословной"):
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                arr = spatial_field(world, layer, max_side=50)
                samples.append((time.perf_counter() - start) * 1000)
            print(f"  {layer:24s} {statistics.median(samples):8.3f} мс, "
                  f"матрица {arr.shape[1]}x{arr.shape[0]}")


if __name__ == "__main__":
    main()
