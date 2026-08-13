import argparse
from pathlib import Path

import numpy as np


def make_sample(rng, size=128):
    axis = np.linspace(-1, 1, size, dtype=np.float32)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    domain = x * x + y * y <= 1
    conductivity = np.ones((size, size), dtype=np.float32) * domain
    for _ in range(rng.randint(1, 4)):
        cx, cy = rng.uniform(-0.5, 0.5, 2)
        radius = rng.uniform(0.08, 0.25)
        inclusion = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        conductivity[inclusion & domain] = rng.uniform(0.5, 2.0)
    image = conductivity + rng.normal(0, 0.12, conductivity.shape).astype(np.float32) * domain
    voltage = rng.normal(0, 1, 208).astype(np.float32)
    voltage[:8] += np.array([conductivity.mean(), conductivity.std()] * 4, dtype=np.float32)
    return voltage, image.astype(np.float32), conductivity


def main(args):
    rng = np.random.RandomState(args.seed)
    counts = {"train": args.train, "val": args.val, "test": args.test}
    root = Path(args.output)
    for split, count in counts.items():
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            voltage, image, conductivity = make_sample(rng)
            np.savez_compressed(directory / f"sample_{index:04d}.npz", voltage=voltage,
                                image=image, conductivity=conductivity)
    print(f"演示数据已生成到 {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/demo")
    parser.add_argument("--train", type=int, default=8)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    main(parser.parse_args())
