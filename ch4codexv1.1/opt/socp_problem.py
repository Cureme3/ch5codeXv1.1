"""SCvx 子问题（SOCP）构建模板。

这里集中管理所有离散动力学等式约束、推力锥约束、动压/过载限制以及
信赖域策略。当前版本仅暴露接口说明，便于后续逐步补全实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .discretization import DiscreteDynamics, DiscreteTrajectory, NominalTrajectoryBundle
from src.sim.run_nominal import R_EARTH

if TYPE_CHECKING:
    from src.sim.mission_domains import TerminalTarget

try:  # pragma: no cover - 运行环境若缺 cvxpy 会在调用处提示
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None  # type: ignore


@dataclass
class SOCPDiagnostics:
    """单次 SOCP 求解后的诊断信息，用于 SCvx 收敛分析。"""

    total_cost: float
    cost_state: float
    cost_control: float
    cost_slack: float
    cost_terminal: float
    feas_violation: float  # 所有松弛变量的最大值
    max_slack_q: float
    max_slack_n: float
    max_slack_cone: float
    term_error_norm: float  # 终端状态（高度/速度）误差范数
    solver_status: str


@dataclass
class TrustRegionConfig:
    """信赖域参数集合，参照第四章中 ρ/σ 调节策略。"""

    radius_state: float
    radius_control: float
    min_radius: float
    max_radius: float
    shrink_ratio: float = 0.5
    expand_ratio: float = 1.5
    rho_accept: float = 0.1
    rho_expand: float = 0.7


@dataclass
class ConstraintBounds:
    """路径/终端约束的集中描述。"""

    max_dynamic_pressure: float
    max_normal_load: float
    thrust_cone_deg: float
    thrust_bounds: Dict[int, tuple[float, float]]
    terminal_state_target: Optional[np.ndarray] = None
    terminal_weights: Optional[np.ndarray] = None
    require_orbit: bool = True  # 是否要求入轨（控制终端轨道精度项）
    # Domain-specific terminal target (aligned with KZ-1A nominal 500km orbit)
    terminal_target_altitude_km: float = 500.0  # Default to nominal orbit
    terminal_target_velocity_kms: float = 7.61  # Circular orbit at 500km
    terminal_target_downrange_km: Optional[float] = None
    # NEW: Flight path angle target for orbit insertion (gamma -> 0 for circular orbit)
    terminal_target_fpa_deg: float = 0.0  # Target flight path angle (0 for orbit)
    # NEW: Safe area parameters for SAFE_AREA domain
    safe_area_center_downrange_km: Optional[float] = None  # Safe zone center
    safe_area_radius_km: Optional[float] = None  # Safe zone radius
    # NEW: Flag to indicate if this is a landing mission (vs orbit insertion)
    is_landing_mission: bool = False  # True for SAFE_AREA, False for RETAIN/DEGRADED


@dataclass
class PenaltyWeights:
    """目标函数权重，包含状态误差、控制偏差与松弛惩罚。"""

    state_dev: float = 1.0
    control_dev: float = 1.0
    terminal_state_dev: float = 30.0
    q_slack: float = 10.0
    n_slack: float = 10.0
    cone_slack: float = 10.0
    terminal_state: float = 1.0
    terminal_speed: float = 1.0
    thrust_slew: float = 1e-2
    slack_penalty: float = 1e3


class SOCPProblemBuilder:
    """根据线性化结果构造一次凸子问题。

    典型使用方式：
        builder = SOCPProblemBuilder(bounds=..., weights=..., trust_region=...)
        problem, variables = builder.build_problem(bundle, dynamics)
        problem.solve(...)
        solution = builder.extract_trajectory(variables, bundle.grid)
    """

    def __init__(
        self,
        *,
        bounds: ConstraintBounds,
        weights: PenaltyWeights,
        trust_region: TrustRegionConfig,
        solver_opts: Optional[Dict[str, object]] = None,
    ):
        self.bounds = bounds
        self.weights = weights
        self.trust_region = trust_region
        self.solver_opts = solver_opts or {}

    def build_problem(
        self,
        bundle: NominalTrajectoryBundle,
        dynamics: DiscreteDynamics,
        *,
        include_slack: bool = True,
    ):
        """根据当前线性化点构造 CVXPy Problem。

        Returns
        -------
        problem : cvxpy.Problem
            已连接等式/不等式约束与目标函数的 SOCP。
        variables : Dict[str, "cvxpy.Variable"]
            便于迭代器提取轨迹增量的变量映射。
        """

        if cp is None:  # pragma: no cover
            raise RuntimeError("cvxpy 未安装，无法构建 SOCP 子问题")
        x_nom = bundle.states
        u_nom = bundle.controls
        N, state_dim = x_nom.shape
        _, control_dim = u_nom.shape
        x_vars = [cp.Variable(state_dim) for _ in range(N)]
        u_vars = [cp.Variable(control_dim) for _ in range(N)]
        constraints = [x_vars[0] == x_nom[0]]
        slack_q = [cp.Variable(nonneg=True) for _ in range(N)]
        slack_n = [cp.Variable(nonneg=True) for _ in range(N)]
        slack_cone = [cp.Variable(nonneg=True) for _ in range(N)]
        for k in range(N - 1):
            A_k = dynamics.A[k]
            B_k = dynamics.B[k]
            c_k = dynamics.c[k]
            constraints.append(x_vars[k + 1] == A_k @ x_vars[k] + B_k @ u_vars[k] + c_k)
        q_nom = bundle.metadata.get("dynamic_pressure_kpa")
        if q_nom is None:
            q_nom = np.zeros(N)
        n_nom = bundle.metadata.get("normal_load_g")
        if n_nom is None:
            n_nom = np.zeros(N)
        q_max = float(self.bounds.max_dynamic_pressure)
        n_max = float(self.bounds.max_normal_load)
        cone_limit_deg = float(self.bounds.thrust_cone_deg)
        cone_slope = np.tan(np.deg2rad(cone_limit_deg))
        for k in range(N):
            constraints.append(q_nom[k] <= q_max + slack_q[k])
            constraints.append(n_nom[k] <= n_max + slack_n[k])
            u_vec = u_vars[k]
            axial = u_vec[0]
            if control_dim > 1:
                tangential = cp.norm(u_vec[1:], 2)
            else:
                tangential = 0.0
            constraints.append(axial >= 0)
            constraints.append(tangential <= cone_slope * axial + slack_cone[k])
            if self.trust_region.radius_state and self.trust_region.radius_state > 0:
                constraints.append(cp.norm(x_vars[k] - x_nom[k], 2) <= self.trust_region.radius_state)
            if self.trust_region.radius_control and self.trust_region.radius_control > 0:
                constraints.append(cp.norm(u_vars[k] - u_nom[k], 2) <= self.trust_region.radius_control)

            # === ALTITUDE LOWER BOUND CONSTRAINT ===
            # Prevent trajectory from going underground (critical for DEGRADED mode)
            # Use linear approximation: r_hat @ pos >= R_EARTH + min_altitude
            # This ensures altitude stays above minimum (e.g., 0 km for landing, 50 km for orbit)
            if not self.bounds.is_landing_mission:
                # For orbit insertion modes (RETAIN/DEGRADED): minimum altitude 50km during trajectory
                # This prevents the optimizer from finding paths that go through Earth
                pos_nom_k = x_nom[k, 0:3]
                r_nom_k = np.linalg.norm(pos_nom_k)
                if r_nom_k > 1e-6:
                    r_hat_k = pos_nom_k / r_nom_k
                    pos_k = x_vars[k][0:3]
                    # Linear constraint: r_hat_k @ pos_k >= R_EARTH + 50km (with some slack)
                    min_altitude_m = 20.0 * 1000.0  # 20km minimum altitude during flight
                    min_radius = R_EARTH + min_altitude_m
                    # Soft constraint via slack (allows small violations but penalizes them)
                    constraints.append(r_hat_k @ pos_k >= min_radius * 0.95)
        # Cost components
        state_term = cp.sum([cp.sum_squares(x_vars[k] - x_nom[k]) for k in range(N)]) * self.weights.state_dev
        control_term = cp.sum([cp.sum_squares(u_vars[k] - u_nom[k]) for k in range(N)]) * self.weights.control_dev
        slack_term = (
            self.weights.q_slack * cp.sum(slack_q)
            + self.weights.n_slack * cp.sum(slack_n)
            + self.weights.cone_slack * cp.sum(slack_cone)
        )

        # === Domain-Specific Terminal Objective ===
        # Supports two mission types:
        # 1. ORBIT INSERTION (RETAIN/DEGRADED): Target altitude, velocity, and flight path angle -> 0
        # 2. SAFE LANDING (SAFE_AREA): Target ground (0km), low velocity, and downrange position
        #
        # NOTE: We use linear approximations to maintain DCP compliance
        pos_nom_final = x_nom[-1, 0:3]
        vel_nom_final = x_nom[-1, 3:6]
        r_nom = float(np.linalg.norm(pos_nom_final))
        v_nom = float(np.linalg.norm(vel_nom_final))

        # Compute direction vectors (use nominal direction for projection)
        if r_nom < 1e-6:
            r_hat = np.array([0.0, 0.0, 1.0])
        else:
            r_hat = pos_nom_final / r_nom
        if v_nom < 1e-6:
            v_hat = np.array([0.0, 0.0, 1.0])
        else:
            v_hat = vel_nom_final / v_nom

        # Compute tangential direction (horizontal component of velocity)
        # For flight path angle computation
        v_radial = np.dot(vel_nom_final, r_hat) * r_hat
        v_tangent = vel_nom_final - v_radial
        v_tan_norm = np.linalg.norm(v_tangent)
        if v_tan_norm > 1e-6:
            t_hat = v_tangent / v_tan_norm
        else:
            # Use perpendicular to r_hat in orbital plane
            ez = np.array([0.0, 0.0, 1.0])
            t_hat = np.cross(ez, r_hat)
            t_hat = t_hat / max(np.linalg.norm(t_hat), 1e-6)

        # Get domain-specific terminal targets (in meters and m/s)
        target_h_m = self.bounds.terminal_target_altitude_km * 1000.0  # km -> m
        target_v_ms = self.bounds.terminal_target_velocity_kms * 1000.0  # km/s -> m/s
        target_fpa_rad = np.deg2rad(self.bounds.terminal_target_fpa_deg)  # Target FPA in radians

        # Final state expressions
        pos_final = x_vars[-1][0:3]
        vel_final = x_vars[-1][3:6]

        # --- Compute terminal error based on mission type ---
        base_terminal_weight = self.weights.terminal_state_dev

        if self.bounds.is_landing_mission:
            # === SAFE LANDING MODE (SAFE_AREA) ===
            # Target: Ground (0km altitude), MINIMAL vertical velocity, specific downrange
            # Key: Smooth descent with near-zero vertical velocity at touchdown
            # Using simple linear approximations for DCP compliance

            # Altitude error (target = 0km ground level)
            h_final_linear = r_hat @ pos_final - R_EARTH  # Altitude in meters
            h_error_scaled = h_final_linear / 5000.0  # Scale by 5km for even stronger gradient

            # === CRITICAL: Separate radial and tangential velocity constraints ===
            # For smooth landing:
            # - Radial velocity (vertical) should be very small (near 0, gentle touchdown)
            # - Tangential velocity (horizontal) can be moderate

            # Radial velocity component (MUST be near zero for smooth landing)
            vr_final = r_hat @ vel_final  # Radial velocity (positive = upward)
            # Target: vr -> 0 (or slightly negative for descent, but very small)
            # Scale by 10 m/s for VERY strong constraint on vertical velocity
            vr_error_scaled = vr_final / 10.0  # Strong penalty on vertical velocity

            # Tangential velocity component (horizontal, can be larger)
            vt_final = t_hat @ vel_final  # Tangential velocity
            # Target: moderate horizontal velocity (allow some drift)
            # Scale by 500 m/s for weaker constraint on horizontal velocity
            vt_error_scaled = (vt_final - target_v_ms) / 500.0

            # Downrange error (if specified) - Use linearized approximation for DCP compliance
            if self.bounds.safe_area_center_downrange_km is not None:
                # For safe landing, we use a linear approximation of downrange
                # Downrange ~ horizontal component of position projected onto tangential direction
                target_dr_m = self.bounds.safe_area_center_downrange_km * 1000.0
                # Use tangential direction t_hat computed from nominal trajectory
                dr_linear = t_hat @ pos_final  # Linear projection (DCP compliant)
                dr_error_scaled = (dr_linear - target_dr_m) / 50000.0  # Scale by 50km

                # Terminal error: [altitude, radial_velocity, tangential_velocity, downrange]
                # Note: radial velocity has much stronger weight due to smaller scaling
                terminal_error = cp.hstack([h_error_scaled, vr_error_scaled, vt_error_scaled, dr_error_scaled])
            else:
                terminal_error = cp.hstack([h_error_scaled, vr_error_scaled, vt_error_scaled])

            # Very high weight for landing safety - must hit the ground safely with minimal vertical velocity
            terminal_weight_factor = 800.0  # Increased from 500.0
            terminal_term = base_terminal_weight * terminal_weight_factor * cp.sum_squares(terminal_error)

        else:
            # === ORBIT INSERTION MODE (RETAIN/DEGRADED) ===
            # Target: Specific altitude, circular velocity, and flight path angle -> 0

            # Altitude error
            h_final_linear = r_hat @ pos_final - R_EARTH  # Altitude in meters
            h_error_scaled = (h_final_linear - target_h_m) / 20000.0  # Scale by 20km for stronger gradient

            # Velocity magnitude error
            v_final_linear = v_hat @ vel_final  # Velocity in m/s (projected)
            v_error_scaled = (v_final_linear - target_v_ms) / 200.0  # Scale by 200m/s for stronger constraint

            # === KEY: Flight Path Angle (FPA) constraint for orbit insertion ===
            # FPA = arctan(v_radial / v_tangential)
            # For circular orbit: FPA should be 0 (v_radial = 0)
            # Linear approximation: gamma ≈ v_radial / v_tangential (for small angles)
            # Target: v_radial -> 0 for circular orbit

            # Radial velocity component (should approach 0 for circular orbit)
            vr_final = r_hat @ vel_final  # Radial velocity (positive = away from Earth)
            # For orbit insertion, target FPA is typically 0 (horizontal)
            # Target radial velocity = v * sin(target_fpa)
            target_vr = target_v_ms * np.sin(target_fpa_rad)

            # FPA error scaled (penalize radial velocity deviation from target)
            # Scale by 50 m/s for stronger FPA constraint (critical for orbit insertion)
            fpa_error_scaled = (vr_final - target_vr) / 50.0

            # Combined terminal error: [altitude, velocity, flight_path_angle]
            terminal_error = cp.hstack([h_error_scaled, v_error_scaled, fpa_error_scaled])

            # Strong terminal guidance for orbit insertion - FPA must approach 0
            terminal_weight_factor = 200.0  # Increased from 50.0
            terminal_term = base_terminal_weight * terminal_weight_factor * cp.sum_squares(terminal_error)

        objective = cp.Minimize(state_term + control_term + slack_term + terminal_term)
        problem = cp.Problem(objective, constraints)
        variables = {
            "x": x_vars,
            "u": u_vars,
            "nominal_states": x_nom,
            "nominal_controls": u_nom,
            "s_q": slack_q,
            "s_n": slack_n,
            "s_cone": slack_cone,
            "nominal_q": q_nom,
            "nominal_n": n_nom,
            # Cost expression references for diagnostics
            "cost_state_expr": state_term,
            "cost_control_expr": control_term,
            "cost_slack_expr": slack_term,
            "cost_terminal_expr": terminal_term,
            "terminal_error_expr": terminal_error,
        }
        return problem, variables

    def extract_trajectory(
        self,
        variable_dict: Dict[str, "cvxpy.Variable"],  # type: ignore[name-defined]
        bundle: NominalTrajectoryBundle,
        problem: "cp.Problem",  # type: ignore[name-defined]
    ) -> tuple[DiscreteTrajectory, SOCPDiagnostics]:
        """把求解器返回的 Variable 数组解码为离散轨迹，并计算诊断信息。"""

        x_vars: List[cp.Variable] = variable_dict["x"]  # type: ignore[assignment]
        u_vars: List[cp.Variable] = variable_dict["u"]  # type: ignore[assignment]
        s_q_vars: List[cp.Variable] = variable_dict.get("s_q", [])
        s_n_vars: List[cp.Variable] = variable_dict.get("s_n", [])
        s_cone_vars: List[cp.Variable] = variable_dict.get("s_cone", [])
        x_nom = variable_dict["nominal_states"]
        u_nom = variable_dict["nominal_controls"]

        # Get cost expression references
        cost_state_expr = variable_dict.get("cost_state_expr")
        cost_control_expr = variable_dict.get("cost_control_expr")
        cost_slack_expr = variable_dict.get("cost_slack_expr")
        cost_terminal_expr = variable_dict.get("cost_terminal_expr")
        terminal_error_expr = variable_dict.get("terminal_error_expr")

        values_missing = any(var.value is None for var in x_vars + u_vars + s_q_vars + s_n_vars + s_cone_vars)
        if values_missing:
            n, nx = x_nom.shape
            _, nu = u_nom.shape
            nan_states = np.full((n, nx), np.nan)
            nan_controls = np.full((n, nu), np.nan)
            trajectory = DiscreteTrajectory(grid=bundle.grid, states=nan_states, controls=nan_controls)
            diagnostics = SOCPDiagnostics(
                total_cost=float("nan"),
                cost_state=float("nan"),
                cost_control=float("nan"),
                cost_slack=float("nan"),
                cost_terminal=float("nan"),
                feas_violation=float("nan"),
                max_slack_q=float("nan"),
                max_slack_n=float("nan"),
                max_slack_cone=float("nan"),
                term_error_norm=float("nan"),
                solver_status=problem.status,
            )
            return trajectory, diagnostics

        x_opt = np.vstack([np.asarray(var.value).reshape(1, -1) for var in x_vars])
        u_opt = np.vstack([np.asarray(var.value).reshape(1, -1) for var in u_vars])
        q_slack = np.array([float(var.value) for var in s_q_vars]) if s_q_vars else np.zeros(len(x_opt))
        n_slack = np.array([float(var.value) for var in s_n_vars]) if s_n_vars else np.zeros(len(x_opt))
        cone_slack = np.array([float(var.value) for var in s_cone_vars]) if s_cone_vars else np.zeros(len(x_opt))

        trajectory = DiscreteTrajectory(grid=bundle.grid, states=x_opt, controls=u_opt)
        trajectory.metadata["q_slack"] = q_slack
        trajectory.metadata["n_slack"] = n_slack
        trajectory.metadata["cone_slack"] = cone_slack

        # Compute cost components from expressions
        total_cost = float(problem.value) if problem.value is not None else 0.0
        cost_state = float(cost_state_expr.value) if cost_state_expr is not None and cost_state_expr.value is not None else 0.0
        cost_control = float(cost_control_expr.value) if cost_control_expr is not None and cost_control_expr.value is not None else 0.0
        cost_slack = float(cost_slack_expr.value) if cost_slack_expr is not None and cost_slack_expr.value is not None else 0.0
        cost_terminal = float(cost_terminal_expr.value) if cost_terminal_expr is not None and cost_terminal_expr.value is not None else 0.0

        # Compute slack violations
        max_slack_q = float(np.max(q_slack)) if len(q_slack) > 0 else 0.0
        max_slack_n = float(np.max(n_slack)) if len(n_slack) > 0 else 0.0
        max_slack_cone = float(np.max(cone_slack)) if len(cone_slack) > 0 else 0.0
        feas_violation = max(max_slack_q, max_slack_n, max_slack_cone)

        # Compute terminal error norm
        if terminal_error_expr is not None and terminal_error_expr.value is not None:
            term_error_vec = np.asarray(terminal_error_expr.value).flatten()
            term_error_norm = float(np.linalg.norm(term_error_vec))
        else:
            term_error_norm = 0.0

        diagnostics = SOCPDiagnostics(
            total_cost=total_cost,
            cost_state=cost_state,
            cost_control=cost_control,
            cost_slack=cost_slack,
            cost_terminal=cost_terminal,
            feas_violation=feas_violation,
            max_slack_q=max_slack_q,
            max_slack_n=max_slack_n,
            max_slack_cone=max_slack_cone,
            term_error_norm=term_error_norm,
            solver_status=problem.status,
        )

        return trajectory, diagnostics

    def trust_region_adjustment(self, ratio: float) -> None:
        """根据实际改进比 ρ 调整信赖域半径。"""

        raise NotImplementedError("trust_region_adjustment() 将在实现阶段提供")
