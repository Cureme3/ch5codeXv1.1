"""时间离散化与线性化接口。

本模块负责把连续时间 6DoF（或其降阶）动力学离散到时间网格，并整理成
SCvx/SCP 循环可复用的数据结构。当前仅给出类定义与 docstring，用于约定接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.sim.run_nominal import NominalResult


@dataclass
class GridConfig:
    """离散化配置。

    Attributes
    ----------
    t0 : float
        起始时间，通常为发射时刻。
    tf : float
        终止时间，可取第四级关机或需要规划的末端。
    num_nodes : int
        离散点数量（包含 t0 和 tf）。如果需要分段加密，可与
        ``segment_density`` 配合使用。
    segment_density : Optional[Dict[str, int]]
        允许按阶段名称（stage1_boost、coast 等）设置节点加密倍率。
        键名须与 `NominalTrajectoryBundle.stage_tags` 对齐。
    grid_type : str
        网格类型标记（uniform / piecewise / chebyshev 等），用于在论文中
        描述具体选型；实现中可据此切换不同策略。
    """

    t0: float
    tf: float
    num_nodes: int
    segment_density: Optional[Dict[str, int]] = None
    grid_type: str = "uniform"


@dataclass
class TimeGrid:
    """时间网格数据结构，包含节点与区间信息。"""

    nodes: np.ndarray
    intervals: np.ndarray
    stage_tags: List[str]

    def __post_init__(self) -> None:
        if self.nodes.ndim != 1:
            raise ValueError("nodes 必须为一维数组")
        if len(self.intervals) != len(self.nodes) - 1:
            raise ValueError("interval 数量应为节点数减一")
        if len(self.stage_tags) != len(self.nodes):
            raise ValueError("stage_tags 数量须与节点一致")

    @property
    def dt(self) -> np.ndarray:
        """返回各区间的时间步长数组。"""

        return self.intervals.copy()


@dataclass
class NominalTrajectoryBundle:
    """封装名义轨迹采样结果，便于做线性化和约束评估。"""

    grid: TimeGrid
    states: np.ndarray
    controls: np.ndarray
    stage_index: np.ndarray
    metadata: Dict[str, np.ndarray] = field(default_factory=dict)

    def sample_state(self, idx: int) -> np.ndarray:
        """返回节点 idx 的状态向量。"""

        return self.states[idx]

    def sample_control(self, idx: int) -> np.ndarray:
        """返回节点 idx 的控制输入（通常为推力方向 + 幅值）。"""

        return self.controls[idx]


@dataclass
class DiscreteDynamics:
    """线性化后的离散动力学 A_k, B_k, c_k 序列。

    数据结构遵循 ``x_{k+1} = A_k x_k + B_k u_k + c_k`` 形式，配合 SCvx 子问题
    的线性等式约束。A/B/c 均按区间存储。
    """

    A: List[np.ndarray]
    B: List[np.ndarray]
    c: List[np.ndarray]
    trust_coeffs: Optional[np.ndarray] = None


@dataclass
class DiscreteTrajectory:
    """离散轨迹解，用于在 SCvx 迭代之间传递。"""

    grid: TimeGrid
    states: np.ndarray
    controls: np.ndarray
    metadata: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dataframe(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """将离散轨迹导出为 DataFrame（后续实现中填充）。"""

        raise NotImplementedError("to_dataframe() 将在实现阶段提供")


class TrajectoryDiscretizer:
    """把连续仿真结果映射到离散网格的工具类。

    典型流程：
        1. 通过 `build_grid` 根据 GridConfig 生成 TimeGrid；
        2. 使用 `project_nominal` 将 NominalResult 投影到网格节点；
        3. 调用 `linearize_dynamics` 计算离散线性化系数；
        4. 将结果组合成 NominalTrajectoryBundle 供 SOCP 构造器使用。
    """

    def __init__(
        self,
        cfg: GridConfig,
        *,
        rocket_cfg_path: Optional[Path] = None,
        dynamics: Optional[object] = None,
    ):
        self.cfg = cfg
        if dynamics is not None:
            self.dynamics = dynamics
        else:
            # For Chapter 4 refactor, we expect dynamics to be injected (Dynamics3DOF)
            # or we can't proceed with default 6-DoF logic.
            raise ValueError("TrajectoryDiscretizer requires 'dynamics' argument for 3-DoF mode.")
            
        self.state_fd_eps = 1e-6
        self.control_fd_eps = 1e-4

    def build_grid(self, *, reference_stage_tags: Optional[Sequence[str]] = None) -> TimeGrid:
        """构建时间网格。"""

        t0 = max(self.cfg.t0, 0.0)
        tf = max(self.cfg.tf, t0 + 1e-3)
        nodes = np.linspace(t0, tf, self.cfg.num_nodes)
        intervals = np.diff(nodes)
        stage_tags = ["segment" for _ in nodes]
        return TimeGrid(nodes=nodes, intervals=intervals, stage_tags=stage_tags)

    def _interpolate_states(self, nominal: NominalResult, sample_times: np.ndarray) -> np.ndarray:
        """在线性网格上插值名义 6DoF 状态."""

        state_history = getattr(nominal, "states", None)
        if state_history is None:
            raise ValueError("NominalResult 缺少 states ??ɢ化无法进行")
        time = nominal.time
        state_history = np.asarray(state_history)
        if state_history.ndim != 2:
            raise ValueError("NominalResult.states 应为二维数组 shape=(N, state_dim)")
        num_samples = len(sample_times)
        state_dim = state_history.shape[1]
        interp = np.zeros((num_samples, state_dim))
        for i in range(state_dim):
            interp[:, i] = np.interp(
                sample_times,
                time,
                state_history[:, i],
                left=state_history[0, i],
                right=state_history[-1, i],
            )
        return interp

    def project_nominal(self, nominal: NominalResult, *, controls: Optional[np.ndarray] = None) -> NominalTrajectoryBundle:
        """把名义仿真结果在 TimeGrid 上采样。"""

        grid = self.build_grid()
        states = self._interpolate_states(nominal, grid.nodes)
        if controls is None:
            controls = np.zeros((len(grid.nodes), 3))
        metadata = {
            "dynamic_pressure_kpa": np.interp(grid.nodes, nominal.time, nominal.dynamic_pressure_kpa),
            "normal_load_g": np.interp(grid.nodes, nominal.time, nominal.normal_load_g),
        }
        return NominalTrajectoryBundle(
            grid=grid,
            states=states,
            controls=controls,
            stage_index=np.zeros(len(grid.nodes), dtype=int),
            metadata=metadata,
        )

    def linearize_dynamics(
        self,
        bundle: NominalTrajectoryBundle,
        *,
        state_dim: int,
        control_dim: int,
        include_gravity_grad: bool = True,  # 保留参数以兼容既有接口
    ) -> DiscreteDynamics:
        """在每个时间区间上对动力学进行一次线性化。"""

        nodes = bundle.grid.nodes
        states = bundle.states
        controls = bundle.controls
        num_intervals = len(nodes) - 1
        A_seq: List[np.ndarray] = []
        B_seq: List[np.ndarray] = []
        c_seq: List[np.ndarray] = []

        for k in range(num_intervals):
            t_k = float(nodes[k])
            dt = float(nodes[k + 1] - nodes[k])
            x_k = states[k]
            u_k = controls[k] if control_dim > 0 else np.zeros(0)
            f_k = self._continuous_dynamics(t_k, x_k, u_k)
            A_c = self._finite_difference_state_jacobian(t_k, x_k, u_k)
            B_c = self._finite_difference_control_jacobian(t_k, x_k, u_k, control_dim)
            A_k = np.eye(state_dim) + dt * A_c
            B_k = dt * B_c
            x_next = states[k + 1]
            xu_term = B_k @ u_k if control_dim > 0 else 0.0
            c_k = x_next - A_k @ x_k - xu_term
            A_seq.append(A_k)
            B_seq.append(B_k)
            c_seq.append(c_k)

        # Debug: verify B_k is non-zero
        if num_intervals > 0 and control_dim > 0:
            B_norms = [np.linalg.norm(B) for B in B_seq[:min(5, num_intervals)]]
            print(f"[DEBUG] B_k norms (first {len(B_norms)} intervals): {B_norms}")
            if all(norm < 1e-10 for norm in B_norms):
                print("[WARNING] All B_k matrices are near-zero! Control may not affect dynamics.")
            else:
                print("[SUCCESS] B_k matrices are non-zero. Control input is active.")

        return DiscreteDynamics(A=A_seq, B=B_seq, c=c_seq)

    def _continuous_dynamics(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """连续时间动力学 f(t, x, u)。

        当 control 为空或维度为 0 时,退化为无控制动力学 f(t, x)。
        当 control 非空时,将其作为 u 传递给 Dynamics3DOF.eom。
        """
        # 防御性处理：允许 control 为标量、一维或空数组
        if control is None or control.size == 0:
            return self.dynamics.eom(t, state)
        else:
            # 确保 control 是 1D 向量
            u = np.asarray(control).reshape(-1)
            # Dynamics3DOF.eom 支持 eom(t, x, u) 签名
            return self.dynamics.eom(t, state, u)

    def _finite_difference_state_jacobian(
        self,
        t: float,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """利用中心差分计算 ∂f/∂x。"""

        state_dim = state.shape[0]
        jac = np.zeros((state_dim, state_dim))
        for idx in range(state_dim):
            delta = max(self.state_fd_eps, abs(state[idx]) * self.state_fd_eps)
            x_plus = state.copy()
            x_minus = state.copy()
            x_plus[idx] += delta
            x_minus[idx] -= delta
            f_plus = self._continuous_dynamics(t, x_plus, control)
            f_minus = self._continuous_dynamics(t, x_minus, control)
            jac[:, idx] = (f_plus - f_minus) / (2.0 * delta)
        return jac

    def _finite_difference_control_jacobian(
        self,
        t: float,
        state: np.ndarray,
        control: np.ndarray,
        control_dim: int,
    ) -> np.ndarray:
        """利用中心差分计算 ∂f/∂u，默认控制不直接作用于动力学。"""

        if control_dim == 0:
            return np.zeros((state.shape[0], 0))
        if control.size == 0:
            control = np.zeros(control_dim)
        jac = np.zeros((state.shape[0], control_dim))
        for idx in range(control_dim):
            delta = max(self.control_fd_eps, abs(control[idx]) * self.control_fd_eps)
            u_plus = control.copy()
            u_minus = control.copy()
            u_plus[idx] += delta
            u_minus[idx] -= delta
            f_plus = self._continuous_dynamics(t, state, u_plus)
            f_minus = self._continuous_dynamics(t, state, u_minus)
            jac[:, idx] = (f_plus - f_minus) / (2.0 * delta)
        return jac

    def refine_grid(self, bundle: NominalTrajectoryBundle, *, indicators: Iterable[float]) -> TimeGrid:
        """根据误差指标对网格进行局部加密。

        `indicators` 可选来自动压约束 slack、信赖域违例等，接口留作后续扩展。
        """

        raise NotImplementedError("refine_grid() 需在实现阶段提供")
