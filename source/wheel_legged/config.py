from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WheelLeggedAssetCfg:
    file: str = "source/assets/wheel_legged/urdf/wl.urdf"
    name: str = "WheelLegged"
    offset: float = 0.054
    l1: float = 0.15
    l2: float = 0.25
    penalize_contacts_on: tuple[str, ...] = ("lf", "rf", "base")
    terminate_after_contacts_on: tuple[str, ...] = ("base",)
    self_collisions: bool = True
    flip_visual_attachments: bool = False


@dataclass
class WheelLeggedControlCfg:
    pos_action_scale: float = 0.5
    vel_action_scale: float = 10.0
    stiffness: dict[str, float] = field(
        default_factory=lambda: {"f0": 40.0, "f1": 40.0, "wheel": 0.0}
    )
    damping: dict[str, float] = field(
        default_factory=lambda: {"f0": 1.0, "f1": 1.0, "wheel": 0.5}
    )


@dataclass
class WheelLeggedBaseCfg:
    num_envs: int = 4096
    episode_length_s: float = 20.0
    asset: WheelLeggedAssetCfg = field(default_factory=WheelLeggedAssetCfg)
    control: WheelLeggedControlCfg = field(default_factory=WheelLeggedControlCfg)
    default_joint_angles: dict[str, float] = field(
        default_factory=lambda: {
            "lf0_Joint": 0.5,
            "lf1_Joint": 0.35,
            "l_wheel_Joint": 0.0,
            "rf0_Joint": -0.5,
            "rf1_Joint": -0.35,
            "r_wheel_Joint": 0.0,
        }
    )
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.25)


@dataclass
class WheelLeggedVmcCfg(WheelLeggedBaseCfg):
    action_scale_theta: float = 0.5
    action_scale_l0: float = 0.1
    action_scale_vel: float = 10.0
    l0_offset: float = 0.175
    feedforward_force: float = 40.0
    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 900.0
    kd_l0: float = 20.0
    wheel_stiffness: float = 0.0
    wheel_damping: float = 0.5
    num_privileged_obs: int = 0


@dataclass
class WheelLeggedLabCfg:
    base: WheelLeggedBaseCfg = field(default_factory=WheelLeggedBaseCfg)
    vmc: WheelLeggedVmcCfg = field(default_factory=WheelLeggedVmcCfg)
