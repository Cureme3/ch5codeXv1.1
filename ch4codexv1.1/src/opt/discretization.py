#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SCvx 离散化配置和轨迹数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GridConfig:
    """时间离散化网格配置。

    Attributes
    ----------
    t0 : float
        起始时间 (s)
    tf : float
        终止时间 (s)
    num_nodes : int
        离散节点数
    """
    t0: float
    tf: float
    num_nodes: int

    @property
    def dt(self) -> float:
        """时间步长。"""
        if self.num_nodes <= 1:
            return 0.0
        return (self.tf - self.t0) / (self.num_nodes - 1)

    @property
    def t_nodes(self) -> np.ndarray:
        """时间节点数组。"""
        return np.linspace(self.t0, self.tf, self.num_nodes)

    @property
    def duration(self) -> float:
        """总时长。"""
        return self.tf - self.t0


@dataclass
class DiscreteTrajectory:
    """离散轨迹数据结构。

    Attributes
    ----------
    grid : GridConfig
        时间网格配置
    states : np.ndarray
        状态序列，形状 (num_nodes, state_dim)
    controls : np.ndarray
        控制序列，形状 (num_nodes, control_dim)
    slacks : Optional[np.ndarray]
        松弛变量序列（可选）
    """
    grid: GridConfig
    states: np.ndarray
    controls: np.ndarray
    slacks: Optional[np.ndarray] = None

    @property
    def num_nodes(self) -> int:
        """节点数。"""
        return self.grid.num_nodes

    @property
    def state_dim(self) -> int:
        """状态维度。"""
        return self.states.shape[1] if self.states.ndim > 1 else 1

    @property
    def control_dim(self) -> int:
        """控制维度。"""
        return self.controls.shape[1] if self.controls.ndim > 1 else 1

    @property
    def t_nodes(self) -> np.ndarray:
        """时间节点数组。"""
        return self.grid.t_nodes

    def get_state_at_index(self, idx: int) -> np.ndarray:
        """获取指定索引处的状态。"""
        return self.states[idx]

    def get_control_at_index(self, idx: int) -> np.ndarray:
        """获取指定索引处的控制。"""
        return self.controls[idx]

    def interpolate_state(self, t: float) -> np.ndarray:
        """线性插值获取指定时刻的状态。"""
        t_nodes = self.t_nodes
        if t <= t_nodes[0]:
            return self.states[0]
        if t >= t_nodes[-1]:
            return self.states[-1]

        # 找到 t 所在的区间
        idx = np.searchsorted(t_nodes, t) - 1
        idx = max(0, min(idx, len(t_nodes) - 2))

        # 线性插值
        t0, t1 = t_nodes[idx], t_nodes[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

        return (1 - alpha) * self.states[idx] + alpha * self.states[idx + 1]
