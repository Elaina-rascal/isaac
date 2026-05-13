from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher
from wheel_legged import *
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wheel-Legged Isaac Lab training entry point.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--num_envs", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    _simulation_app = app_launcher.app
    cfg = WheelLeggedLabCfg()
    cfg.base.num_envs = args.num_envs
    task = WheelLeggedLabTask(cfg)
    reset_info = task.reset()
    sample_actions = task._last_actions.new_zeros(cfg.base.num_envs, task.num_actions)
    obs, reward, done, info = task.step(sample_actions)
    print(
        {
            "reset_envs": int(reset_info["env_ids"].numel()),
            "obs_shape": tuple(obs.shape),
            "reward_shape": tuple(reward.shape),
            "done_shape": tuple(done.shape),
            "step_count": info["step_count"],
        }
    )


if __name__ == "__main__":
    main()
