# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, int] = (32, 32)) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_dim: int, output_dim: int) -> MLPRegressor:
    """构造一个简单的 MLP 回归模型，用于从任务/故障特征预测推力序列。"""
    return MLPRegressor(input_dim=input_dim, output_dim=output_dim)
