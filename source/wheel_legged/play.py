from __future__ import annotations

import argparse

from .config import WheelLeggedLabCfg
from .task import WheelLeggedLabTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wheel-Legged Isaac Lab playback entry point.")
    parser.add_argument("--checkpoint", type=str, default="")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    cfg = WheelLeggedLabCfg()
    task = WheelLeggedLabTask(cfg)
    task.reset()
    print({"obs_shape": tuple(task.compute_observations().shape)})


if __name__ == "__main__":
    main()
