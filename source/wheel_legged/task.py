from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import WheelLeggedLabCfg
from .kinematics import forward_kinematics


@dataclass
class WheelLeggedLabTask:
    cfg: WheelLeggedLabCfg
    num_actions: int = 6
    num_observations: int = 12
    _step_count: int = field(init=False, default=0)
    _last_actions: torch.Tensor = field(init=False, repr=False)
    _obs: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._last_actions = torch.zeros(self.cfg.base.num_envs, self.num_actions)
        self._obs = torch.zeros(self.cfg.base.num_envs, self.num_observations)

    def compute_vmc_kinematics(
        self, theta1: torch.Tensor, theta2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return forward_kinematics(
            theta1,
            theta2,
            self.cfg.vmc.asset.offset,
            self.cfg.vmc.asset.l1,
            self.cfg.vmc.asset.l2,
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.cfg.base.num_envs)
        self._step_count = 0
        self._last_actions[env_ids] = 0.0
        self._obs[env_ids] = 0.0
        theta1 = torch.zeros(len(env_ids), 2)
        theta2 = torch.zeros(len(env_ids), 2)
        l0, theta0 = self.compute_vmc_kinematics(theta1, theta2)
        return {
            "env_ids": env_ids,
            "l0": l0,
            "theta0": theta0,
            "obs": self._obs[env_ids],
        }

    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[0] != self.cfg.base.num_envs:
            repeat_count = self.cfg.base.num_envs // actions.shape[0]
            actions = actions.repeat(repeat_count, 1)
        self._step_count += 1
        self._last_actions = actions[:, : self.num_actions]

        theta1 = torch.stack(
            (
                self._last_actions[:, 0] * self.cfg.vmc.action_scale_theta,
                self._last_actions[:, 3] * -self.cfg.vmc.action_scale_theta,
            ),
            dim=1,
        )
        theta2 = torch.stack(
            (
                self._last_actions[:, 1] * self.cfg.vmc.action_scale_theta,
                self._last_actions[:, 4] * -self.cfg.vmc.action_scale_theta,
            ),
            dim=1,
        )
        l0, theta0 = self.compute_vmc_kinematics(theta1, theta2)
        self._obs[:, 0:2] = theta1
        self._obs[:, 2:4] = theta2
        self._obs[:, 4:6] = l0
        self._obs[:, 6:8] = theta0
        self._obs[:, 8:] = self._last_actions[:, :4]
        reward = -(self._last_actions**2).sum(dim=1)
        done = torch.zeros(self.cfg.base.num_envs, dtype=torch.bool)
        info = {"step_count": self._step_count}
        return self._obs.clone(), reward, done, info

    def compute_observations(self):
        return self._obs.clone()

    def compute_rewards(self):
        return -(self._last_actions**2).sum(dim=1)
