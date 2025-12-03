"""SCvx 的单次 SOCP 子问题定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from .constraints import OptimizationConfig

import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None  # type: ignore


@dataclass
class Discretization:
    dt: float
    nodes: int
    state_dim: int
    control_dim: int


@dataclass
class TrustRegion:
    state_radius: float
    control_radius: float


@dataclass
class TrustRegionConfig:
    """信赖域配置。"""
    radius_state: float = 100.0  # 状态信赖域半径
    radius_control: float = 10000.0  # 控制信赖域半径
    shrink_factor: float = 0.5
    expand_factor: float = 2.0
    min_radius: float = 1e-3
    max_radius: float = 1e6
    # 兼容旧字段名
    state_radius: float = 100.0
    control_radius: float = 10000.0


@dataclass
class ConstraintBounds:
    """约束边界配置。"""
    max_dynamic_pressure: float = 50000.0  # 最大动压 (Pa)
    max_normal_load: float = 6.0  # 最大过载 (g)
    thrust_cone_deg: float = 15.0  # 推力锥半角 (deg)
    thrust_bounds: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # 各级推力边界
    require_orbit: bool = True  # 是否要求入轨
    # 兼容旧字段名
    q_max: float = 50000.0  # 最大动压 (Pa) - 别名
    n_max: float = 6.0  # 最大过载 (g) - 别名
    thrust_min: float = 0.0  # 最小推力 (N)
    thrust_max: float = 1e6  # 最大推力 (N)
    cone_half_angle_deg: float = 15.0  # 推力锥半角 (deg) - 别名


@dataclass
class PenaltyWeights:
    """SCvx 优化问题的罚权重配置。"""
    state_dev: float = 1.0
    control_dev: float = 1.0
    q_slack: float = 10.0
    n_slack: float = 10.0
    cone_slack: float = 10.0
    terminal_state_dev: float = 50.0


class SOCPProblemBuilder:
    """SOCP 问题构造器。"""

    def __init__(
        self,
        grid_config,
        constraint_bounds: ConstraintBounds,
        penalty_weights: PenaltyWeights,
        trust_region: TrustRegionConfig,
    ):
        self.grid = grid_config
        self.bounds = constraint_bounds
        self.weights = penalty_weights
        self.trust = trust_region

    def build(self, x_ref: np.ndarray, u_ref: np.ndarray):
        """构建 SOCP 问题。"""
        if cp is None:
            raise RuntimeError("cvxpy 未安装，无法构建 SOCP。")
        # 简化实现，实际应根据具体问题构建
        return None, {}


class SOCPBuilder:
    """封装 cvxpy 变量、约束、目标的构造。"""

    def __init__(
        self,
        discretization: Discretization,
        trust_region: TrustRegion,
        weights: Dict[str, float],
        config: OptimizationConfig,
    ):
        if cp is None:
            raise RuntimeError("cvxpy 未安装，无法构建 SOCP。")
        self.disc = discretization
        self.trust = trust_region
        self.weights = weights
        self.config = config

    def build(
        self,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
        x_target: np.ndarray,
        dynamics_matrices: Tuple[List[np.ndarray], List[np.ndarray]],
    ):
        """返回 (problem, variables) 供上层求解。"""
        A_list, B_list = dynamics_matrices
        N = self.disc.nodes
        n = self.disc.state_dim
        m = self.disc.control_dim
        x = cp.Variable((N, n))
        u = cp.Variable((N, m))
        nu = cp.Variable((N - 1, n))
        constraints = [x[0, :] == x_ref[0, :]]
        
        # 预计算推力锥参数
        cone_angle_deg = self.config.nominal_thrust_cone_deg
        tightening_rad = self.config.fault_cone_tightening_rad
        effective_cone_rad = np.deg2rad(cone_angle_deg) - tightening_rad
        tan_theta_cone = np.tan(max(0.0, effective_cone_rad))
        
        for k in range(N - 1):
            # 动力学线性化约束 (A, B 现在是列表)
            # x[k+1] = x[k] + (A_k(x_k - x_ref) + B_k(u_k - u_ref) + f(x_ref, u_ref)) * dt
            # 注意：原代码简化了 f(x_ref, u_ref) 项，假设了 x_ref 满足动力学？
            # 原代码: x[k+1] == x[k] + (A @ (x[k] - x_ref) + B @ (u[k] - u_ref)) * dt + nu[k]
            # 这实际上是 delta_x[k+1] = delta_x[k] + (A delta_x + B delta_u) * dt
            # 如果 x_ref 是动力学可行的，那么 f(x_ref, u_ref) * dt = x_ref[k+1] - x_ref[k]
            # 于是 x[k+1] - x_ref[k+1] = x[k] - x_ref[k] + ...
            # 这是一个标准的线性化。
            
            Ak = A_list[k]
            Bk = B_list[k]
            
            constraints.append(
                x[k + 1, :] == x[k, :] + (Ak @ (x[k, :] - x_ref[k, :]) + Bk @ (u[k, :] - u_ref[k, :])) * self.disc.dt + nu[k, :]
            )
            constraints.append(cp.norm(x[k, :] - x_ref[k, :], 2) <= self.trust.state_radius)
            constraints.append(cp.norm(u[k, :] - u_ref[k, :], 2) <= self.trust.control_radius)
            
            # --- 推力幅值约束 ---
            # T_max_k = T_nom * (1 - drop)
            # 假设推力下降故障在整个 horizon 内生效 (简化)
            # 或者我们应该根据 k * dt 是否 > t_fault 来判断？
            # 鉴于 OptimizationConfig 只有静态参数，我们这里做静态处理。
            # 如果需要动态，需要在 config 里传入 t_fault。
            # 暂时按静态处理 (假设故障已发生或未发生)
            T_max = self.config.nominal_thrust_N * (1.0 - self.config.fault_thrust_drop)
            constraints.append(cp.norm(u[k, :], 2) <= T_max)
            
            # --- 推力锥约束 (SOCP) ---
            # norm(u_xy) <= tan(theta) * u_z
            # u[k, 0:2] is xy, u[k, 2] is z
            constraints.append(cp.norm(u[k, 0:2], 2) <= tan_theta_cone * u[k, 2])
            # 确保 z > 0
            constraints.append(u[k, 2] >= 0)

        cost = 0.0
        
        # --- 终端约束 (软约束) ---
        r_target = self.config.r_target
        v_target = self.config.v_target
        # r: 0:3, v: 3:6
        cost += self.weights["terminal"] * (
            cp.sum_squares(x[-1, 0:3] - r_target) + 
            cp.sum_squares(x[-1, 3:6] - v_target)
        )
        
        cost += self.weights["control"] * cp.sum_squares(u) * self.disc.dt
        cost += self.weights["virtual"] * cp.sum(cp.norm(nu, axis=1))
        problem = cp.Problem(cp.Minimize(cost), constraints)
        return problem, {"x": x, "u": u, "nu": nu}
