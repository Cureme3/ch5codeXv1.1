#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import math

from opt.socp_problem import PenaltyWeights


@dataclass
class AdaptivePenaltyConfig:
    """自适应权重的基准配置与缩放范围。"""

    base_state_dev: float = 1.0
    base_control_dev: float = 1.0
    base_terminal: float = 50.0
    base_q_slack: float = 10.0
    base_n_slack: float = 10.0
    base_cone_slack: float = 10.0


def _pseudo_noise(eta: float, freq1: float, freq2: float, amp: float) -> float:
    signal = 0.6 * math.sin(freq1 * eta + 0.35) + 0.4 * math.sin(freq2 * eta + 1.2)
    return 1.0 + amp * signal


def compute_adaptive_penalties(
    eta: float,
    cfg: AdaptivePenaltyConfig | None = None,
) -> PenaltyWeights:
    """
    根据归一化故障强度 eta ∈ [0, 1] 计算 SCvx 的罚权重。
    约定：eta 越大，故障越严重。

    设计思路：
    - 状态/控制偏差权重保持常数；
    - 终端偏差权重随 eta 略微下降（严重故障时适当放宽终端性能）；
    - 约束松弛罚权重随 eta 上升（严重故障时更强调安全约束）。
    """
    if cfg is None:
        cfg = AdaptivePenaltyConfig()

    eta_clamped = max(0.0, min(1.0, float(eta)))

    ripple = _pseudo_noise(eta_clamped, 4.5 * math.pi, 7.5 * math.pi, 0.05)
    terminal_scale = (0.55 + 0.45 * math.exp(-3.0 * eta_clamped)) * ripple
    w_terminal = cfg.base_terminal * terminal_scale

    slack_base = 1.0 + 1.2 * eta_clamped + 2.2 * (eta_clamped**2)
    slack_mod = _pseudo_noise(eta_clamped, 5.0 * math.pi, 11.0 * math.pi, 0.08)
    w_q_slack = cfg.base_q_slack * slack_base * slack_mod

    n_base = 1.0 + 1.5 * eta_clamped + 2.5 * eta_clamped**2
    n_mod = _pseudo_noise(eta_clamped, 4.5 * math.pi, 8.5 * math.pi, 0.06)
    w_n_slack = cfg.base_n_slack * n_base * n_mod

    cone_base = 1.0 + 1.0 * eta_clamped + 3.0 * eta_clamped**2
    cone_mod = _pseudo_noise(eta_clamped, 3.5 * math.pi, 9.0 * math.pi, 0.1)
    w_cone_slack = cfg.base_cone_slack * cone_base * cone_mod

    return PenaltyWeights(
        state_dev=cfg.base_state_dev,
        control_dev=cfg.base_control_dev,
        q_slack=w_q_slack,
        n_slack=w_n_slack,
        cone_slack=w_cone_slack,
        terminal_state_dev=w_terminal,
    )
