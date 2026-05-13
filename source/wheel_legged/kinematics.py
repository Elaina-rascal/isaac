from __future__ import annotations

import torch


def forward_kinematics(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    offset: float,
    l1: float,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    end_x = offset + l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
    end_y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)
    l0 = torch.sqrt(end_x**2 + end_y**2)
    theta0 = torch.arctan2(end_y, end_x) - torch.pi / 2
    return l0, theta0
