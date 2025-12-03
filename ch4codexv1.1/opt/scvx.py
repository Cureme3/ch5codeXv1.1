"""SCvx 迭代求解器框架。

本模块封装高层循环：线性化 → 构建 SOCP → 求解 → 信赖域更新 → 收敛判据。
实现将遵循《第四章需求 4 说明》中的流程图，此处先提供接口骨架。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np

from src.sim.run_nominal import NominalResult

from .discretization import DiscreteDynamics, DiscreteTrajectory, GridConfig, NominalTrajectoryBundle, TrajectoryDiscretizer
from .socp_problem import ConstraintBounds, PenaltyWeights, SOCPDiagnostics, SOCPProblemBuilder, TrustRegionConfig


@dataclass
class IterationLog:
    """记录单次 SCvx 迭代的诊断量（增强版，包含成本分解）。"""

    iter_idx: int
    rho: float
    cost_nominal: float
    cost_candidate: float
    trust_radius: float
    solver_status: str
    feasibility_violation: float
    # 新增：成本分解
    total_cost: float
    cost_state: float
    cost_control: float
    cost_slack: float
    cost_terminal: float
    max_slack_q: float
    max_slack_n: float
    max_slack_cone: float
    term_error_norm: float


@dataclass
class SCvxResult:
    """SCvx 求解结果，包括轨迹与诊断信息。

    # NOTE: 这是第四章 SCvx 规划器的统一返回结构。
    # 所有调用 SCvxPlanner 的代码都应期望获得此结构。
    # 关键字段说明：
    # - trajectory: 优化后的离散轨迹（状态 + 控制）
    # - logs: 每次外层迭代的详细日志（IterationLog 列表）
    # - success: 求解是否成功（至少有一次迭代达到 optimal 状态）
    # - num_iterations: 实际执行的迭代次数
    # - solve_time_s: 总求解时间（秒）
    """

    trajectory: DiscreteTrajectory
    logs: List[IterationLog]
    success: bool
    message: str = ""
    penalties: PenaltyWeights | None = None
    num_iterations: int = 0
    solve_time_s: float = 0.0  # 新增：求解时间（秒）



class SCvxPlanner:
    """面向 KZ-1A 上升段的 SCvx 规划器。

    Parameters
    ----------
    grid_cfg : GridConfig
        离散化设置，允许后续在 quick demo 中以 python -m scripts.run_scvx_demo 调用。
    bounds : ConstraintBounds
        路径/终端约束，来源于 configs/kz1a_params.yaml。
    weights : PenaltyWeights
        目标函数权重，便于进行敏感性分析。
    trust_region : TrustRegionConfig
        信赖域调度规则。
    solver_name : str
        CVXPy 支持的求解器标识，默认 "ECOS"。
    """

    def __init__(
        self,
        *,
        grid_cfg: GridConfig,
        bounds: ConstraintBounds,
        weights: PenaltyWeights,
        trust_region: TrustRegionConfig,
        solver_name: str = "ECOS",
        solver_opts: Optional[Dict[str, object]] = None,
        dynamics: Optional[object] = None,
    ):
        self.grid_cfg = grid_cfg
        self.bounds = bounds
        self.weights = weights
        self.trust_region = trust_region
        self.solver_name = solver_name
        self.solver_opts = solver_opts or {}
        self.solver_opts = solver_opts or {}
        # Force 7-dim state for Chapter 4 refactor
        if dynamics is None:
             # If no dynamics provided, we assume 3-DoF logic is handled elsewhere or via default
             pass
        
        # We must ensure the discretizer knows we are in 3-DoF mode
        # The TrajectoryDiscretizer might infer dims from dynamics or config.
        # Let's explicitly check or set if possible. 
        # Actually, TrajectoryDiscretizer takes grid_cfg.
        # We should ensure grid_cfg implies 7-dim if it has such fields, 
        # or we rely on the dynamics object to provide dimensions.
        
        self.discretizer = TrajectoryDiscretizer(grid_cfg, dynamics=dynamics)
        self.builder = SOCPProblemBuilder(
            bounds=bounds,
            weights=weights,
            trust_region=trust_region,
            solver_opts=self.solver_opts,
        )
        self.logs: List[IterationLog] = []

    def set_penalty_weights(self, weights: PenaltyWeights) -> None:
        """Update planner to use a new set of penalty weights."""

        self.weights = weights
        self.builder.weights = weights

    def initialize_from_nominal(self, nominal: NominalResult) -> NominalTrajectoryBundle:
        """把名义仿真结果映射到离散轨迹，作为第一次线性化点。"""

        grid = self.discretizer.build_grid()
        bundle = self.discretizer.project_nominal(nominal)
        return bundle

    def iterate(
        self,
        nominal_bundle: NominalTrajectoryBundle,
        max_iters: int = 10,
    ) -> SCvxResult:
        """SCvx outer iteration loop."""
        import time
        t_start = time.perf_counter()

        def _bundle_to_trajectory(bundle: NominalTrajectoryBundle) -> DiscreteTrajectory:
            return DiscreteTrajectory(
                grid=bundle.grid,
                states=np.asarray(bundle.states),
                controls=np.asarray(bundle.controls),
                metadata=dict(bundle.metadata),
            )

        current = nominal_bundle
        logs: List[IterationLog] = []
        for k in range(max_iters):
            dynamics = self._linearize(current)
            candidate_traj, diagnostics = self._solve_socp(current, dynamics)
            solver_status = diagnostics.solver_status
            feasibility = diagnostics.feas_violation

            current_traj = _bundle_to_trajectory(current)
            cost_nominal = self._trajectory_cost(current_traj)
            cost_candidate = self._trajectory_cost(candidate_traj)
            if solver_status not in {"optimal", "optimal_inaccurate"}:
                log = self._build_log(
                    iter_idx=k,
                    rho=0.0,
                    cost_nominal=cost_nominal,
                    cost_candidate=cost_candidate,
                    diagnostics=diagnostics,
                )
                logs.append(log)
                break

            rho = self._evaluate_improvement(current_traj, candidate_traj)
            current = self._accept_or_reject(rho, candidate_traj, current)
            log = self._build_log(
                iter_idx=k,
                rho=rho,
                cost_nominal=cost_nominal,
                cost_candidate=cost_candidate,
                diagnostics=diagnostics,
            )
            logs.append(log)
            if abs(rho) < 1e-3 and feasibility < 1e-3:
                break

        success = any(log.solver_status in {"optimal", "optimal_inaccurate"} for log in logs)
        final_traj = _bundle_to_trajectory(current)
        message = logs[-1].solver_status if logs else ""
        t_end = time.perf_counter()
        result = SCvxResult(
            trajectory=final_traj,
            logs=logs,
            success=success,
            message=message,
            penalties=replace(self.weights),
            num_iterations=len(logs),
            solve_time_s=t_end - t_start,
        )
        return result

    def _linearize(self, bundle: NominalTrajectoryBundle) -> DiscreteDynamics:
        """辅助函数：调用 TrajectoryDiscretizer 对当前轨迹线性化。"""

        state_dim = bundle.states.shape[1]
        control_dim = bundle.controls.shape[1]
        return self.discretizer.linearize_dynamics(bundle, state_dim=state_dim, control_dim=control_dim)

    def _solve_socp(self, bundle: NominalTrajectoryBundle, dynamics: DiscreteDynamics) -> tuple[DiscreteTrajectory, SOCPDiagnostics]:
        """构造并求解一次凸子问题，返回离散轨迹候选解和诊断信息。"""

        problem, variables = self.builder.build_problem(bundle, dynamics)
        solve_kwargs = dict(solver=self.solver_name)
        solve_kwargs.update(self.solver_opts)
        problem.solve(**solve_kwargs)
        trajectory, diagnostics = self.builder.extract_trajectory(variables, bundle, problem)
        return trajectory, diagnostics

    def _trajectory_cost(self, trajectory: DiscreteTrajectory) -> float:
        """Compute simplified cost = terminal deviation + control energy."""

        states = np.asarray(trajectory.states)
        controls = np.asarray(trajectory.controls)
        if states.size == 0:
            return 0.0
        terminal_state = states[-1]
        target = getattr(self.bounds, "terminal_state_target", None)
        term_weights = getattr(self.bounds, "terminal_weights", None)
        terminal_cost = 0.0
        if target is not None:
            diff = terminal_state - target
            if term_weights is not None:
                terminal_cost = float(np.sum((term_weights * diff) ** 2))
            else:
                terminal_cost = float(np.sum(diff**2))
        control_cost = float(np.sum(controls**2))
        return terminal_cost + control_cost

    def _evaluate_improvement(
        self,
        nominal: DiscreteTrajectory,
        candidate: DiscreteTrajectory,
    ) -> float:
        """Compute a heuristic improvement ratio between two trajectories."""

        # NOTE: Use a simplified improvement ratio to steer trust-region updates.
        cost_nominal = self._trajectory_cost(nominal)
        cost_candidate = self._trajectory_cost(candidate)
        numerator = cost_nominal - cost_candidate
        denom = max(abs(cost_nominal), 1e-6)
        rho = numerator / denom
        return float(rho)

    def _accept_or_reject(self, rho: float, candidate: DiscreteTrajectory, current: NominalTrajectoryBundle) -> NominalTrajectoryBundle:
        """Decide whether to accept the candidate trajectory and update the trust region."""

        # NOTE: Simplified trust-region update used repeatedly inside iterate().
        radius = float(self.trust_region.radius_state or 0.0)
        if rho < 0.0:
            accepted = False
            radius *= 0.5
        elif rho < 0.25:
            accepted = True
            radius *= 0.7
        elif rho < 0.75:
            accepted = True
            radius *= 1.0
        else:
            accepted = True
            radius *= 1.5

        min_radius = getattr(self.trust_region, "min_radius", 1e-3)
        max_radius = getattr(self.trust_region, "max_radius", 1e3)
        radius = max(min_radius, min(max_radius, radius))
        self.trust_region.radius_state = radius

        if not accepted:
            return current

        metadata = dict(current.metadata)
        metadata.update(candidate.metadata)
        return NominalTrajectoryBundle(
            grid=candidate.grid,
            states=np.asarray(candidate.states).copy(),
            controls=np.asarray(candidate.controls).copy(),
            stage_index=np.asarray(current.stage_index).copy(),
            metadata=metadata,
        )

    def run(self, nominal: NominalResult, *, max_iters: int = 10) -> SCvxResult:
        """SCvx entry point"""

        bundle = self.initialize_from_nominal(nominal)
        return self.iterate(bundle, max_iters=max_iters)

    def _build_log(
        self,
        *,
        iter_idx: int,
        rho: float,
        cost_nominal: float,
        cost_candidate: float,
        diagnostics: SOCPDiagnostics,
    ) -> IterationLog:
        """Package collected metrics into an IterationLog entry."""

        trust_radius = float(self.trust_region.radius_state or 0.0)
        return IterationLog(
            iter_idx=iter_idx,
            rho=float(rho),
            cost_nominal=float(cost_nominal),
            cost_candidate=float(cost_candidate),
            trust_radius=trust_radius,
            solver_status=diagnostics.solver_status,
            feasibility_violation=diagnostics.feas_violation,
            # 新增字段
            total_cost=diagnostics.total_cost,
            cost_state=diagnostics.cost_state,
            cost_control=diagnostics.cost_control,
            cost_slack=diagnostics.cost_slack,
            cost_terminal=diagnostics.cost_terminal,
            max_slack_q=diagnostics.max_slack_q,
            max_slack_n=diagnostics.max_slack_n,
            max_slack_cone=diagnostics.max_slack_cone,
            term_error_norm=diagnostics.term_error_norm,
        )
