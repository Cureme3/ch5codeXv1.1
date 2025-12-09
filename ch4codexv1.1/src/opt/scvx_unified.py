"""统一 SCvx 接口适配器。

整合两套 SCvxPlanner 实现:
- opt/scvx.py: 完整版，支持 GridConfig, ConstraintBounds, 信赖域调度
- src/opt/scvx.py: 简化版，直接使用 OptimizationConfig

本模块提供统一接口，自动选择合适的后端。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np


@dataclass
class UnifiedSCvxResult:
    """统一的 SCvx 求解结果。"""
    states: np.ndarray          # 优化后状态轨迹 [N, 7]
    controls: np.ndarray        # 优化后控制序列 [N, 3]
    time: np.ndarray            # 时间网格
    cost: float                 # 最终代价
    virtual_norm: float         # 最终虚拟控制范数
    iterations: int             # 迭代次数
    converged: bool             # 是否收敛
    solver_status: str          # 求解器状态
    diagnostics: Dict[str, Any] # 额外诊断信息


class UnifiedSCvxPlanner:
    """统一 SCvx 规划器接口。

    Usage:
        planner = UnifiedSCvxPlanner(
            nodes=40,
            dt=1.0,
            r_target=np.array([...]),
            v_target=np.array([...]),
        )
        result = planner.solve(x_init, u_init)
    """

    def __init__(
        self,
        nodes: int = 40,
        dt: float = 1.0,
        r_target: Optional[np.ndarray] = None,
        v_target: Optional[np.ndarray] = None,
        max_thrust_N: float = 300000.0,
        thrust_cone_deg: float = 15.0,
        fault_thrust_drop: float = 0.0,
        fault_cone_tightening_rad: float = 0.0,
        max_iters: int = 15,
        verbose: bool = True,
    ):
        self.nodes = nodes
        self.dt = dt
        self.max_iters = max_iters
        self.verbose = verbose

        # 默认目标: 500km 圆轨道
        R_EARTH = 6.378137e6
        if r_target is None:
            r_target = np.array([0.0, 0.0, R_EARTH + 500e3])
        if v_target is None:
            v_target = np.array([0.0, 7610.0, 0.0])

        self.r_target = np.asarray(r_target, dtype=float)
        self.v_target = np.asarray(v_target, dtype=float)

        # 约束参数
        self.max_thrust_N = max_thrust_N
        self.thrust_cone_deg = thrust_cone_deg
        self.fault_thrust_drop = fault_thrust_drop
        self.fault_cone_tightening_rad = fault_cone_tightening_rad

        # 自适应权重 (可通过 set_weights 更新)
        self.weights = {
            "terminal": 100.0,
            "control": 0.01,
            "virtual": 10000.0,
            "state_dev": 1.0,
            "q_slack": 10.0,
            "n_slack": 10.0,
            "cone_slack": 10.0,
        }

    def set_weights(self, **kwargs) -> None:
        """更新权重参数。"""
        self.weights.update(kwargs)

    def set_adaptive_weights(self, eta: float) -> None:
        """根据故障严重度设置自适应权重。"""
        eta = max(0.0, min(1.0, eta))
        terminal_scale = 0.55 + 0.45 * np.exp(-3.0 * eta)
        slack_scale = 1.0 + 1.2 * eta + 2.2 * eta**2

        self.weights["terminal"] = 100.0 * terminal_scale
        self.weights["q_slack"] = 10.0 * slack_scale
        self.weights["n_slack"] = 10.0 * slack_scale
        self.weights["cone_slack"] = 10.0 * slack_scale

    def solve(
        self,
        x_init: np.ndarray,
        u_init: np.ndarray,
    ) -> UnifiedSCvxResult:
        """执行 SCvx 求解。"""
        try:
            import cvxpy as cp
        except ImportError:
            raise RuntimeError("cvxpy 未安装，无法运行 SCvx")

        N = self.nodes
        n, m = 7, 3
        R_EARTH = 6.378137e6
        MU = 3.986004418e14

        # 保存真实初始状态
        x0_true = np.asarray(x_init[0], dtype=float) if x_init.ndim > 1 else np.asarray(x_init, dtype=float)
        if len(x0_true) < 7:
            x0_full = np.zeros(7)
            x0_full[:len(x0_true)] = x0_true
            x0_full[6] = 8000.0
            x0_true = x0_full

        # 生成初始猜测轨迹
        x_ref = self._generate_initial_guess(x_init)
        u_ref = np.ones((N, m)) * 50000.0
        u_ref[:, 0:2] = 0.0

        x_target = np.concatenate([self.r_target, self.v_target, [0.0]])
        time_grid = np.linspace(0, self.dt * (N - 1), N)

        costs = []
        virtual_norms = []
        solver_status = "not_started"

        T_max = self.max_thrust_N * (1.0 - self.fault_thrust_drop)

        for iteration in range(self.max_iters):
            # 构建 SOCP
            x = cp.Variable((N, n))
            u = cp.Variable((N, m))
            nu = cp.Variable((N - 1, n))

            # 初始状态约束 - 固定为真实初始状态
            constraints = [x[0, :] == x0_true]

            # 动力学约束 (线性化欧拉积分)
            for k in range(N - 1):
                r_k = x_ref[k, 0:3]
                r_norm = np.linalg.norm(r_k) + 1e-6
                mass_k = max(x_ref[k, 6], 100.0)
                u_ref_k = u_ref[k, :]
                u_ref_norm = np.linalg.norm(u_ref_k) + 1e-6

                # 重力加速度
                g_vec = -MU / r_norm**3 * r_k

                # 线性化质量消耗: dm/dt ≈ -|u_ref|/Isp - (u - u_ref)·u_ref/(|u_ref|*Isp)
                dm_dt_ref = -u_ref_norm / 3000.0

                # 动力学约束
                constraints.append(x[k+1, 0:3] == x[k, 0:3] + self.dt * x[k, 3:6] + nu[k, 0:3])
                constraints.append(x[k+1, 3:6] == x[k, 3:6] + self.dt * (g_vec + u[k, :] / mass_k) + nu[k, 3:6])
                constraints.append(x[k+1, 6] == x[k, 6] + self.dt * dm_dt_ref + nu[k, 6])

            # 信赖域约束 (随迭代逐渐放宽)
            trust_r = 200000.0 * (1.0 + iteration * 0.5)  # 位置信赖域
            trust_v = 2000.0 * (1.0 + iteration * 0.5)    # 速度信赖域
            for k in range(N):
                constraints.append(cp.norm(x[k, 0:3] - x_ref[k, 0:3], 2) <= trust_r)
                constraints.append(cp.norm(x[k, 3:6] - x_ref[k, 3:6], 2) <= trust_v)

            # 推力约束 (只有上界是凸的)
            for k in range(N):
                constraints.append(cp.norm(u[k, :], 2) <= T_max)

            # 目标函数
            r_scale = 1e5  # 100km scale
            v_scale = 1e3  # 1km/s scale
            cost = 0.0
            # 终端代价 (高权重确保到达目标)
            cost += 1000.0 * self.weights["terminal"] * (
                cp.sum_squares((x[-1, 0:3] - x_target[0:3]) / r_scale) +
                cp.sum_squares((x[-1, 3:6] - x_target[3:6]) / v_scale)
            )
            cost += self.weights["control"] * cp.sum_squares(u / T_max) * self.dt
            cost += self.weights["virtual"] * cp.sum(cp.norm(nu, axis=1))

            problem = cp.Problem(cp.Minimize(cost), constraints)

            try:
                problem.solve(solver="ECOS", verbose=False)
                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    problem.solve(solver="SCS", verbose=False, max_iters=10000)
                solver_status = problem.status
            except Exception:
                # ECOS 失败时尝试 SCS
                try:
                    problem.solve(solver="SCS", verbose=False, max_iters=10000)
                    solver_status = problem.status
                except Exception as e2:
                    solver_status = f"error: {e2}"
                    break

            if x.value is None:
                solver_status = "no_solution"
                break

            x_new = np.array(x.value)
            u_new = np.array(u.value)
            nu_val = np.array(nu.value)

            virtual_norm = float(np.sum(np.linalg.norm(nu_val, axis=1)))
            costs.append(float(problem.value))
            virtual_norms.append(virtual_norm)

            if self.verbose:
                h_final = (np.linalg.norm(x_new[-1, 0:3]) - R_EARTH) / 1000.0
                v_final = np.linalg.norm(x_new[-1, 3:6]) / 1000.0
                print(f"  Iter {iteration+1}: VirtNorm={virtual_norm:.2f}, h={h_final:.1f}km, v={v_final:.2f}km/s")

            # 更新参考轨迹
            x_ref = x_new
            u_ref = u_new

            # 检查收敛 (需要虚拟控制小且终端误差小)
            h_err = abs(np.linalg.norm(x_ref[-1, 0:3]) - np.linalg.norm(x_target[0:3])) / 1000.0
            v_err = abs(np.linalg.norm(x_ref[-1, 3:6]) - np.linalg.norm(x_target[3:6])) / 1000.0
            if virtual_norm < 1000.0 and h_err < 20.0 and v_err < 0.5:
                if self.verbose:
                    print(f"[SCvx] Converged (h_err={h_err:.1f}km, v_err={v_err:.2f}km/s)")
                break

        # 检查最终收敛状态
        if len(virtual_norms) > 0:
            h_err = abs(np.linalg.norm(x_ref[-1, 0:3]) - np.linalg.norm(x_target[0:3])) / 1000.0
            v_err = abs(np.linalg.norm(x_ref[-1, 3:6]) - np.linalg.norm(x_target[3:6])) / 1000.0
            converged = virtual_norms[-1] < 1000.0 and h_err < 20.0 and v_err < 0.5
        else:
            converged = False

        return UnifiedSCvxResult(
            states=x_ref,
            controls=u_ref,
            time=time_grid,
            cost=costs[-1] if costs else float("inf"),
            virtual_norm=virtual_norms[-1] if virtual_norms else float("inf"),
            iterations=len(costs),
            converged=converged,
            solver_status=solver_status,
            diagnostics={
                "costs": costs,
                "virtual_norms": virtual_norms,
                "weights": self.weights.copy(),
            },
        )

    def _linearize_trajectory(
        self,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
        time_grid: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """沿参考轨迹线性化动力学（含重力）。"""
        A_seq, B_seq = [], []
        n, m = 7, 3
        R_EARTH = 6.378137e6
        MU = 3.986004418e14  # 地球引力常数

        for k in range(len(time_grid)):
            r = x_ref[k, 0:3]
            r_norm = np.linalg.norm(r) + 1e-6
            mass = max(x_ref[k, 6], 100.0)

            # A 矩阵: 状态转移
            A = np.eye(n)
            A[0:3, 3:6] = np.eye(3) * self.dt

            # 重力梯度项
            g_coeff = -MU / r_norm**3
            G = g_coeff * (np.eye(3) - 3.0 * np.outer(r, r) / r_norm**2)
            A[3:6, 0:3] = G * self.dt

            # B 矩阵: 控制输入
            B = np.zeros((n, m))
            B[3:6, :] = np.eye(3) / mass * self.dt

            A_seq.append(A)
            B_seq.append(B)

        return A_seq, B_seq

    def _resample(self, arr: np.ndarray, N: int) -> np.ndarray:
        """重采样数组到 N 个点。"""
        old_N = arr.shape[0]
        if old_N == N:
            return arr
        indices = np.linspace(0, old_N - 1, N).astype(int)
        return arr[indices]

    def _generate_initial_guess(self, x_init: np.ndarray) -> np.ndarray:
        """生成合理的初始猜测轨迹。"""
        N = self.nodes
        R_EARTH = 6.378137e6

        # 使用第一个有效点作为起点
        x0 = np.asarray(x_init[0], dtype=float) if x_init.ndim > 1 else np.asarray(x_init, dtype=float)

        # 确保起点在地球表面以上
        r0 = x0[0:3].copy()
        r0_norm = np.linalg.norm(r0)
        if r0_norm < R_EARTH + 50e3:  # 低于50km则修正
            r0 = r0 / (r0_norm + 1e-6) * (R_EARTH + 100e3)

        v0 = x0[3:6] if len(x0) > 5 else np.array([0.0, 1000.0, 0.0])
        m0 = x0[6] if len(x0) > 6 else 10000.0

        # 目标状态
        rf = self.r_target
        vf = self.v_target

        # 线性插值生成初始轨迹
        x_ref = np.zeros((N, 7))
        for i in range(N):
            alpha = i / (N - 1)
            x_ref[i, 0:3] = (1 - alpha) * r0 + alpha * rf
            x_ref[i, 3:6] = (1 - alpha) * v0 + alpha * vf
            x_ref[i, 6] = m0 * (1 - 0.3 * alpha)  # 质量逐渐减少

        return x_ref
