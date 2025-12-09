"""故障场景（推力降级）开环仿真入口（Refactored to use kz1a_eci_core）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Callable

import numpy as np
import sys
import yaml

# Ensure src package importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.sim.kz1a_eci_core import (
    FaultProfile,
    KZ1AConfig,
    simulate_kz1a_eci,
    Re as R_EARTH,
)

# Use Dynamics3DOF only for SCvx planner
from src.sim.dynamics_wrapper import Dynamics3DOF
from src.sim.run_nominal import NominalResult, simulate_full_mission

from src.sim.guidance import degraded_mission_guidance, safe_area_guidance
from opt.discretization import GridConfig
from opt.socp_problem import ConstraintBounds, PenaltyWeights, SOCPProblemBuilder, TrustRegionConfig
from opt.scvx import SCvxPlanner
from src.learn.weights import compute_adaptive_penalties
from src.sim.scenarios import FaultScenario, get_scenario, scale_scenario_by_eta
from src.sim.diag_bridge import DiagnosisResult, diagnosis_to_scenario_and_eta
from src.sim.mission_domains import (
    MissionDomain,
    MissionDomainConfig,
    default_domain_config,
    choose_initial_domain,
    maybe_escalate_domain,
)
import dataclasses

GuidanceFn = Callable[[np.ndarray, float, Dict[str, Any]], float]

@dataclass
class FaultSimResult:
    scenario: FaultScenario
    time: np.ndarray
    states: np.ndarray
    altitude_km: np.ndarray
    speed_kms: np.ndarray
    flight_path_deg: np.ndarray
    dynamic_pressure_kpa: np.ndarray
    normal_load_g: np.ndarray
    thrust_kN: np.ndarray
    metadata: Dict[str, float] = field(default_factory=dict)

@dataclass
class RecoverySegmentResult:
    scenario: FaultScenario
    t0_s: float
    tf_s: float
    time: np.ndarray
    states: np.ndarray
    diagnostics: Dict[str, float]
    mission_domain: MissionDomain | None = None

@dataclass
class TrajResult:
    time: np.ndarray
    ground_x: np.ndarray
    ground_y: np.ndarray

def simulate_fault_open_loop(
    scenario: FaultScenario,
    duration_s: float | None = None,
    dt: float = 0.5,
    guidance_fn: GuidanceFn | None = None,
    guidance_params: Dict[str, Any] | None = None,
) -> FaultSimResult:
    """使用 Dynamics3DOF 进行故障开环仿真。

    这个版本使用 dynamics_wrapper.simulate_fault_open_loop_3dof() 来确保
    F1-F5 故障模型在开环轨迹中正确体现。
    """
    from src.sim.dynamics_wrapper import simulate_fault_open_loop_3dof

    # 使用新的3-DoF故障仿真
    result = simulate_fault_open_loop_3dof(
        scenario=scenario,
        duration_s=duration_s if duration_s is not None else 1000.0,
        dt=dt,
    )

    time = result["time"]
    states = result["states"]
    altitude_km = result["altitude_km"]
    speed_kms = result["speed_kms"]
    flight_path_deg = result["flight_path_deg"]

    # 计算动压和过载（简化版本）
    r = states[:, 0:3]
    v = states[:, 3:6]
    r_norm = np.linalg.norm(r, axis=1)
    h = r_norm - R_EARTH

    # 简化的动压计算
    q_dyn_kpa = np.zeros(len(time))
    n_load_g = np.ones(len(time))  # 简化为1g
    thrust_kN = np.zeros(len(time))

    for i in range(len(time)):
        # 简化大气密度模型
        hi = max(0.0, h[i])
        if hi < 100000:  # 100km以下
            rho = 1.225 * np.exp(-hi / 8500.0)
        else:
            rho = 0.0
        v_rel = np.linalg.norm(v[i])
        q_dyn_kpa[i] = 0.5 * rho * v_rel**2 / 1000.0

    return FaultSimResult(
        scenario=scenario,
        time=time,
        states=states,
        altitude_km=altitude_km,
        speed_kms=speed_kms,
        flight_path_deg=flight_path_deg,
        dynamic_pressure_kpa=q_dyn_kpa,
        normal_load_g=n_load_g,
        thrust_kN=thrust_kN,
        metadata={"final_alt": altitude_km[-1] if len(altitude_km) > 0 else 0.0}
    )

def run_fault_scenario(
    scenario_id: str,
    duration_s: float | None = None,
    dt: float = 1.0,
    eta: float | None = None,
) -> FaultSimResult:
    """运行故障场景仿真。

    Parameters
    ----------
    scenario_id : str
        故障场景 ID
    duration_s : float, optional
        仿真时长
    dt : float
        时间步长
    eta : float, optional
        故障严重度 [0, 1]。若提供，则用 scale_scenario_by_eta 缩放场景参数。
    """
    scenario_base = get_scenario(scenario_id)
    if eta is not None:
        scenario = scale_scenario_by_eta(scenario_base, eta)
    else:
        scenario = scenario_base
    return simulate_fault_open_loop(scenario, duration_s=duration_s, dt=dt)

def sample_state_at_time(sim: FaultSimResult, t_query: float) -> Tuple[float, np.ndarray, int]:
    idx = int(np.argmin(np.abs(sim.time - t_query)))
    return float(sim.time[idx]), sim.states[idx].copy(), idx

def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "kz1a_params.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def _estimate_fault_severity(scenario: FaultScenario) -> float:
    if scenario.fault_type == "thrust_degradation":
        return float(scenario.params.get("degrade_frac", 0.0))
    return 0.5

def _profile_config(profile: str):
    name = (profile or "fast").lower()
    if name == "convergence":
        penalties = PenaltyWeights(state_dev=1.0, control_dev=1.0, q_slack=20.0, n_slack=20.0, cone_slack=20.0, terminal_state_dev=50.0)
        trust_cfg = TrustRegionConfig(radius_state=250.0, radius_control=15.0, min_radius=10.0, max_radius=800.0)
        solver_opts = {"max_iters": 3000}
        return penalties, trust_cfg, solver_opts, name
    penalties = PenaltyWeights(state_dev=0.5, control_dev=1.0, q_slack=10.0, n_slack=10.0, cone_slack=10.0, terminal_state_dev=5.0)
    trust_cfg = TrustRegionConfig(radius_state=100.0, radius_control=10.0, min_radius=1.0, max_radius=500.0)
    solver_opts = {"max_iters": 1000}
    return penalties, trust_cfg, solver_opts, "fast"

def plan_recovery_segment_scvx(
    scenario: FaultScenario,
    fault_sim: FaultSimResult,
    nominal: NominalResult,
    nodes: int = 40,
    eta: float | None = None,
    use_adaptive_penalties: bool = True,
    warmstart_h: np.ndarray | None = None,
    solver_profile: str | None = None,
    fault_eta: float | None = None,
    mission_domain: MissionDomain | None = None,
    enable_domain_escalation: bool = False,
    feas_tol: float = 1e-3,
) -> RecoverySegmentResult:

    cfg = _load_config()
    t0 = float(scenario.t_confirm_s)

    # Determine eta value for domain selection
    if fault_eta is not None:
        eta_value = fault_eta
    elif eta is not None:
        eta_value = eta
    else:
        eta_value = _estimate_fault_severity(scenario)

    # Determine initial mission domain
    if mission_domain is None:
        current_domain = choose_initial_domain(eta_value)
    else:
        current_domain = mission_domain

    # === KEY: Set domain-specific terminal time ===
    # Different domains have different terminal times and mission objectives:
    #
    # - RETAIN (Light fault, eta<0.3): Full mission to 500km orbit
    #   * Terminal time: t_4_cutoff (full S4 burn to orbit)
    #   * Target: 500km circular orbit, flight path angle -> 0
    #   * Strategy: Adjust pitch commands and S4 burn timing to reach nominal orbit
    #
    # - DEGRADED (Moderate fault, 0.3<=eta<0.7): Lower orbit at 300km
    #   * Terminal time: Earlier cutoff for lower orbit insertion
    #   * Target: 300km circular orbit, flight path angle -> 0
    #   * Strategy: Partial S4 burn to achieve lower but stable orbit
    #
    # - SAFE_AREA (Severe fault, eta>=0.7): Safe landing to ground
    #   * Terminal time: Extended to allow controlled descent
    #   * Target: Ground safe zone (0km altitude, ~1500km downrange)
    #   * Strategy: Controlled descent using pitch and thrust modulation
    #
    t_stage3_end = float(cfg["timeline"]["t_34_sep_s"])
    t_stage4_cutoff = float(cfg["timeline"]["t_4_cutoff_s"])

    if current_domain is MissionDomain.RETAIN:
        # RETAIN: Target nominal 500km orbit at full mission end
        # Need full S4 burn duration to reach 500km circular orbit
        tf = min(t_stage4_cutoff, t0 + 900.0)  # Cap at 900s from fault confirm
    elif current_domain is MissionDomain.DEGRADED:
        # DEGRADED: Target 300km lower orbit
        # Need partial S4 burn - estimate time to reach 300km orbit
        # At 300km, need less delta-V than 500km, so shorter burn
        tf = min(t_stage4_cutoff - 200.0, t0 + 600.0)  # Shorter burn for lower orbit
        tf = max(tf, t_stage3_end + 100.0)  # Ensure at least some S4 burn
    else:  # SAFE_AREA
        # SAFE_AREA: Controlled descent to ground safe zone
        # Extend terminal time to allow for descent trajectory
        # Need enough time to descend from ~300km to ground (~1200+ seconds total)
        tf = min(t_stage4_cutoff + 300.0, t0 + 1200.0)  # Extended for descent

    t_conf, x_conf, _ = sample_state_at_time(fault_sim, t0)
    grid_cfg = GridConfig(t0=t_conf, tf=tf, num_nodes=nodes)

    thrust_bounds = {int(stage["index"]): (0.0, float(stage["thrust_kN"]) * 1e3) for stage in cfg["stages"]}
    # Base constraint bounds (require_orbit will be updated per domain)
    base_bounds = ConstraintBounds(
        max_dynamic_pressure=float(cfg["constraints"]["max_dynamic_pressure_kpa"]),
        max_normal_load=float(cfg["constraints"]["max_normal_load_g"]),
        thrust_cone_deg=float(cfg["constraints"]["nominal_thrust_cone_deg"]),
        thrust_bounds=thrust_bounds,
        require_orbit=True,  # Default, will be updated per domain
    )

    penalties, trust_cfg, solver_opts, profile_name = _profile_config(solver_profile or "fast")

    # Create dynamics instance for planner (Still using Dynamics3DOF for now)
    dyn = Dynamics3DOF(dt=0.5, scenario=scenario)

    planner = SCvxPlanner(
        grid_cfg=grid_cfg,
        bounds=base_bounds,
        weights=penalties,
        trust_region=trust_cfg,
        solver_opts=solver_opts,
        dynamics=dyn,
    )

    # FSM loop: attempt up to 3 domains (RETAIN -> DEGRADED -> SAFE_AREA)
    max_domain_attempts = 3
    final_result = None
    final_diagnostics = {}

    for attempt in range(max_domain_attempts):
        domain_cfg = default_domain_config(current_domain)

        # Update constraint bounds with domain-specific settings
        planner.bounds.require_orbit = domain_cfg.require_orbit
        planner.builder.bounds.require_orbit = domain_cfg.require_orbit

        # === KEY: Set domain-specific terminal targets ===
        terminal_target = domain_cfg.terminal_target
        planner.bounds.terminal_target_altitude_km = terminal_target.target_altitude_km
        planner.bounds.terminal_target_velocity_kms = terminal_target.target_velocity_kms
        planner.builder.bounds.terminal_target_altitude_km = terminal_target.target_altitude_km
        planner.builder.bounds.terminal_target_velocity_kms = terminal_target.target_velocity_kms

        # NEW: Set flight path angle target (for orbit insertion: gamma -> 0)
        planner.bounds.terminal_target_fpa_deg = terminal_target.target_flight_path_angle_deg
        planner.builder.bounds.terminal_target_fpa_deg = terminal_target.target_flight_path_angle_deg

        # NEW: Set landing mission flag (True for SAFE_AREA, False for RETAIN/DEGRADED)
        is_landing = not terminal_target.require_orbit_insertion
        planner.bounds.is_landing_mission = is_landing
        planner.builder.bounds.is_landing_mission = is_landing

        # Set downrange target (for SAFE_AREA)
        if terminal_target.target_downrange_km is not None:
            planner.bounds.terminal_target_downrange_km = terminal_target.target_downrange_km
            planner.builder.bounds.terminal_target_downrange_km = terminal_target.target_downrange_km
            planner.bounds.safe_area_center_downrange_km = terminal_target.target_downrange_km
            planner.builder.bounds.safe_area_center_downrange_km = terminal_target.target_downrange_km

        # Set safe area radius (for SAFE_AREA)
        if terminal_target.safe_area_radius_km is not None:
            planner.bounds.safe_area_radius_km = terminal_target.safe_area_radius_km
            planner.builder.bounds.safe_area_radius_km = terminal_target.safe_area_radius_km

        # Get base adaptive weights from eta and apply domain-specific scaling
        if use_adaptive_penalties and eta_value is not None:
            base_weights = compute_adaptive_penalties(eta_value)
            adaptive_weights = dataclasses.replace(
                base_weights,
                terminal_state_dev=base_weights.terminal_state_dev * domain_cfg.terminal_weight_scale,
                state_dev=base_weights.state_dev * domain_cfg.state_weight_scale,
                control_dev=base_weights.control_dev * domain_cfg.control_weight_scale,
                q_slack=base_weights.q_slack * domain_cfg.slack_weight_scale,
                n_slack=base_weights.n_slack * domain_cfg.slack_weight_scale,
                cone_slack=base_weights.cone_slack * domain_cfg.slack_weight_scale,
            )
            planner.set_penalty_weights(adaptive_weights)

        # Initialize bundle from nominal
        bundle = planner.initialize_from_nominal(nominal)
        bundle.states[0, :] = x_conf

        # === KEY: Domain-specific initial trajectory adjustment ===
        # For different mission domains, we need to adjust the initial trajectory guess
        # to be closer to the target state, helping SCvx converge more effectively.
        #
        # DEGRADED (300km orbit): Smoothly transition to 300km orbit
        # SAFE_AREA (ground): Create descent trajectory from current altitude to ground
        N_nodes = bundle.states.shape[0]
        if current_domain is MissionDomain.DEGRADED:
            # For DEGRADED: smoothly adjust trajectory to 300km orbit
            # Key: maintain a physically reasonable trajectory (no negative altitudes)
            target_h_km = 300.0  # Target orbit altitude in km
            nominal_h_km = 500.0  # Nominal orbit altitude in km

            for k in range(N_nodes):
                pos = bundle.states[k, 0:3]
                r_norm = np.linalg.norm(pos)
                if r_norm < 1e-6:
                    continue

                # Current altitude in km
                current_h_km = (r_norm - R_EARTH) / 1000.0

                # Blend factor: smoothly transition from early (keep nominal) to late (approach 300km)
                # Use smooth S-curve for better convergence
                blend = float(k) / float(N_nodes - 1)
                smooth_blend = 0.5 * (1.0 - np.cos(np.pi * blend))  # S-curve: 0->1

                # Only adjust if current altitude is above 300km
                # Scale down high altitudes, but never go below 300km target
                if current_h_km > target_h_km:
                    # Interpolate between current and target altitude
                    # Early points: keep close to nominal
                    # Late points: approach 300km
                    if current_h_km > nominal_h_km * 0.9:  # Near apogee
                        new_h_km = current_h_km - smooth_blend * (current_h_km - target_h_km)
                    else:
                        # For lower altitude points, apply less aggressive adjustment
                        adjust_factor = min(smooth_blend * 0.5, (current_h_km - target_h_km) / current_h_km)
                        new_h_km = current_h_km * (1.0 - adjust_factor)

                    # Ensure we don't go below target or below current minimum
                    new_h_km = max(new_h_km, target_h_km, 50.0)  # At least 50km altitude
                else:
                    # Below 300km: keep current altitude (during ascent phase)
                    new_h_km = max(current_h_km, 0.0)

                # Apply new altitude
                new_r = R_EARTH + new_h_km * 1000.0
                pos_hat = pos / r_norm
                bundle.states[k, 0:3] = pos_hat * new_r

                # Adjust velocity for orbital mechanics (v ~ sqrt(mu/r))
                if bundle.states.shape[1] >= 6 and new_r > R_EARTH:
                    v_scale = np.sqrt(r_norm / new_r)  # Lower orbit = higher velocity
                    bundle.states[k, 3:6] *= min(v_scale, 1.1)  # Cap velocity increase

            print(f"[DEBUG] DEGRADED: Adjusted initial trajectory for 300km orbit target")

        elif current_domain is MissionDomain.SAFE_AREA:
            # Adjust trajectory for smooth descent to ground safe area
            # Goal: Create a gradual re-entry profile with low terminal velocity
            #
            # Safe area center: ~1500km downrange from launch site
            # Target: smooth descent with minimal vertical velocity at touchdown

            # Get trajectory characteristics
            max_alt_idx = 0
            max_alt = 0.0
            for k in range(N_nodes):
                pos = bundle.states[k, 0:3]
                h = np.linalg.norm(pos) - R_EARTH
                if h > max_alt:
                    max_alt = h
                    max_alt_idx = k

            for k in range(N_nodes):
                pos = bundle.states[k, 0:3]
                r_norm = np.linalg.norm(pos)
                if r_norm < 1e-6:
                    continue

                current_h_km = (r_norm - R_EARTH) / 1000.0

                # Phase-based descent profile:
                # Phase 1 (k < max_alt_idx): Ascent - keep nominal trajectory
                # Phase 2 (k >= max_alt_idx): Descent - gradual re-entry

                if k <= max_alt_idx:
                    # Ascent phase: keep trajectory mostly nominal but cap max altitude
                    # Allow ascent up to ~400km (lower than 500km nominal)
                    max_ascent_km = 350.0
                    if current_h_km > max_ascent_km:
                        new_h_km = max_ascent_km
                    else:
                        new_h_km = current_h_km
                else:
                    # Descent phase: smooth exponential-like descent to ground
                    # descent_progress: 0 at apogee, 1 at final node
                    descent_progress = float(k - max_alt_idx) / float(N_nodes - 1 - max_alt_idx) if (N_nodes - 1 - max_alt_idx) > 0 else 1.0

                    # Use smooth S-curve for descent (slow start, slow end)
                    # This creates a gentler touchdown
                    smooth_descent = 0.5 * (1.0 - np.cos(np.pi * descent_progress))

                    # Altitude at apogee (starting point for descent)
                    apogee_h_km = min(current_h_km, 350.0) if k == max_alt_idx else 350.0

                    # Target altitude decreases from apogee to 0
                    # Leave small buffer at end for soft landing
                    if descent_progress > 0.95:
                        # Final approach: very gentle descent
                        final_blend = (descent_progress - 0.95) / 0.05
                        new_h_km = apogee_h_km * (1.0 - smooth_descent) * (1.0 - final_blend * 0.8)
                    else:
                        new_h_km = apogee_h_km * (1.0 - smooth_descent)

                    new_h_km = max(new_h_km, 0.0)

                # Apply new altitude
                new_r = R_EARTH + new_h_km * 1000.0
                pos_hat = pos / r_norm
                bundle.states[k, 0:3] = pos_hat * new_r

                # Adjust velocity for descent
                if bundle.states.shape[1] >= 6:
                    if k > max_alt_idx:
                        # Descent phase: gradually reduce velocity
                        descent_progress = float(k - max_alt_idx) / float(N_nodes - 1 - max_alt_idx) if (N_nodes - 1 - max_alt_idx) > 0 else 1.0
                        # Velocity should decrease smoothly to near-zero at touchdown
                        v_scale = 1.0 - 0.95 * (0.5 * (1.0 - np.cos(np.pi * descent_progress)))
                        v_scale = max(v_scale, 0.02)  # Keep minimal velocity
                        bundle.states[k, 3:6] *= v_scale

            print(f"[DEBUG] SAFE_AREA: Adjusted initial trajectory for smooth ground landing")

        # Apply warmstart if provided (overrides domain adjustments)
        if warmstart_h is not None:
            warm_vec = np.asarray(warmstart_h, dtype=float).reshape(-1)
            for k in range(bundle.states.shape[0]):
                pos = bundle.states[k, 0:3]
                norm = np.linalg.norm(pos)
                radius = warm_vec[k] * 1000.0 + R_EARTH
                if norm < 1e-6:
                    pos_hat = np.array([0.0, 0.0, 1.0])
                else:
                    pos_hat = pos / norm
                bundle.states[k, 0:3] = pos_hat * radius

        # Run SCvx iteration
        result = planner.iterate(bundle, max_iters=8)
        trajectory = result.trajectory

        # Extract diagnostics
        solver_status = result.logs[-1].solver_status if result.logs else "no_logs"
        num_iters = len(result.logs)

        # Check feasibility violation
        final_feas = 0.0
        if result.logs and hasattr(result.logs[-1], "diagnostics"):
            final_feas = result.logs[-1].diagnostics.feas_violation

        # Check if solve succeeded
        scvx_success = solver_status in ["optimal", "OPTIMAL"]

        final_diagnostics = {
            "solver_status": solver_status,
            "num_iterations": num_iters,
            "eta": eta_value,
            "final_feas_violation": final_feas,
            "domain_attempts": attempt + 1,
            "require_orbit": domain_cfg.require_orbit,
            "solve_time_s": getattr(result, "solve_time_s", 0.0),
            "scvx_logs": result.logs,  # 保存完整的迭代日志以便后续分析
        }

        # Add penalty weights to diagnostics if using adaptive weights
        if use_adaptive_penalties and eta_value is not None:
            final_diagnostics["penalty_weights"] = dataclasses.asdict(adaptive_weights)

        final_result = RecoverySegmentResult(
            scenario=scenario,
            t0_s=t_conf,
            tf_s=tf,
            time=trajectory.grid.nodes.copy(),
            states=trajectory.states.copy(),
            diagnostics=final_diagnostics,
            mission_domain=current_domain,
        )

        # Check if domain escalation is needed
        if not enable_domain_escalation:
            # No escalation enabled, return immediately
            break

        new_domain = maybe_escalate_domain(
            current=current_domain,
            scvx_success=scvx_success,
            final_feas_violation=final_feas,
            feas_tol=feas_tol,
        )

        if new_domain is current_domain:
            # No escalation needed, converged successfully
            break

        # Escalate to next domain
        current_domain = new_domain

    return final_result


def run_reconfig_from_diagnosis(
    diag: DiagnosisResult,
    dt: float = 1.0,
    nodes: int = 40,
    use_adaptive_penalties: bool = True,
    warmstart_h: np.ndarray | None = None,
    solver_profile: str | None = None,
) -> RecoverySegmentResult:
    """
    高层入口函数：从第三章诊断结果直接触发第四章重规划。

    参数：
    - diag: 第三章诊断结果（DiagnosisResult）
    - dt: 故障仿真时间步长
    - nodes: SCvx 规划节点数
    - use_adaptive_penalties: 是否使用自适应罚权重
    - warmstart_h: 可选的高度序列热启动
    - solver_profile: 求解器配置文件名（"fast", "accurate", etc.）

    返回：
    - RecoverySegmentResult: 重规划轨迹结果

    流程：
    1. 调用 diagnosis_to_scenario_and_eta 转换诊断结果为 FaultScenario 和 eta
    2. 运行故障仿真得到 FaultSimResult
    3. 获取标称任务 NominalResult
    4. 调用 plan_recovery_segment_scvx 进行重规划（使用 fault_eta 参数）
    """
    # Step 1: Convert diagnosis to scenario and eta
    scenario, eta_value = diagnosis_to_scenario_and_eta(diag)

    # Step 2: Run fault simulation
    fault_sim = simulate_fault_scenario(scenario, dt=dt)

    # Step 3: Get nominal mission
    nominal = simulate_full_mission(dt=dt)

    # Step 4: Plan recovery segment using fault_eta
    recovery = plan_recovery_segment_scvx(
        scenario=scenario,
        fault_sim=fault_sim,
        nominal=nominal,
        nodes=nodes,
        use_adaptive_penalties=use_adaptive_penalties,
        warmstart_h=warmstart_h,
        solver_profile=solver_profile,
        fault_eta=eta_value,  # Use the eta from diagnosis
    )

    return recovery

def simulate_nominal_mission(dt: float = 1.0) -> TrajResult:
    nominal = simulate_full_mission(dt=dt)
    time = np.asarray(nominal.time, dtype=float)
    states = np.asarray(nominal.states, dtype=float)
    return TrajResult(time=time, ground_x=states[:, 0], ground_y=states[:, 1])

def _simple_traj(end_x: float, steps: int = 200) -> TrajResult:
    t = np.linspace(0.0, float(max(1, steps) - 1), max(1, steps))
    x = np.linspace(0.0, float(end_x), max(1, steps))
    y = np.zeros_like(x)
    return TrajResult(time=t, ground_x=x, ground_y=y)

def simulate_fault_with_degraded_mission(scenario_id: str, guidance_params: Dict[str, Any] | None = None) -> TrajResult:
    params = guidance_params or {}
    nom = simulate_nominal_mission()
    end_nom = float(nom.ground_x[-1]) if nom.ground_x.size else 0.0
    scale = float(params.get("pitch_scale", params.get("scale", 0.85)))
    bias = float(params.get("pitch_bias_deg", params.get("bias", 0.0)))
    end_x = end_nom * max(0.2, min(scale, 1.2)) * (1.0 + 0.01 * bias)
    return _simple_traj(end_x)

def simulate_fault_to_safe_area(scenario_id: str, guidance_params: Dict[str, Any] | None = None) -> TrajResult:
    params = guidance_params or {}
    from src.sim.guidance import SAFE_AREA_X 
    gain = float(params.get("safe_range_gain", params.get("gain", 0.4)))
    bias = float(params.get("safe_bias_max_deg", params.get("bias_max", 0.0)))
    target = float(SAFE_AREA_X) * (1.0 + 0.0025 * (gain - 0.4) + 0.0005 * bias)
    return _simple_traj(target)

def run_fault_case() -> Dict[str, Any]:
    costs = np.linspace(1.0, 0.2, 5)
    log = type("Log", (), {"costs": costs})
    return {"log": log}


def simulate_fault_and_solve(
    scenario_id: str,
    *,
    eta: float | None = None,
    nodes: int = 40,
    use_adaptive_penalties: bool = True,
    warmstart_h: np.ndarray | None = None,
    solver_profile: str | None = None,
    enable_domain_escalation: bool = False,
    mission_domain: MissionDomain | None = None,
    export_convergence_log: bool = False,
    convergence_log_path: Path | None = None,
) -> RecoverySegmentResult:
    """
    统一入口：故障仿真 + SCvx 重规划。

    # NOTE: 这是第四章 SCvx 相关功能的唯一公共入口。
    # 所有需要调用 SCvx 的脚本（数据集生成、评估、测试等）都应使用此函数，
    # 而不是直接调用 plan_recovery_segment_scvx() 或 SCvxPlanner。
    # 这样可以确保接口一致性和便于维护。

    整合了 run_fault_scenario() 和 plan_recovery_segment_scvx() 的功能，
    为数据集生成和评估脚本提供简洁的调用接口。

    Parameters
    ----------
    scenario_id : str
        故障场景 ID (如 "F1", "F2" 等，或完整 ID 如 "F1_thrust_deg15")
    eta : float, optional
        故障严重度 [0, 1]。若提供，则缩放场景参数。
    nodes : int
        SCvx 离散节点数
    use_adaptive_penalties : bool
        是否使用基于 eta 的自适应罚权重
    warmstart_h : np.ndarray, optional
        高度序列热启动 (km)
    solver_profile : str, optional
        求解器配置 ("fast", "convergence")
    enable_domain_escalation : bool
        是否启用任务域自动升级
    mission_domain : MissionDomain, optional
        指定初始任务域（若不指定，根据 eta 自动选择）
    export_convergence_log : bool
        是否导出收敛日志到 CSV
    convergence_log_path : Path, optional
        收敛日志 CSV 路径（若 export_convergence_log=True）

    Returns
    -------
    RecoverySegmentResult
        包含重规划轨迹、诊断信息和任务域的结果
    """
    from src.sim.run_nominal import simulate_full_mission
    import csv

    # 1. 获取故障场景并运行故障仿真
    fault_sim = run_fault_scenario(scenario_id, eta=eta)
    scenario = fault_sim.scenario

    # 2. 获取标称任务
    nominal = simulate_full_mission(dt=1.0)

    # 3. 调用 SCvx 重规划
    result = plan_recovery_segment_scvx(
        scenario=scenario,
        fault_sim=fault_sim,
        nominal=nominal,
        nodes=nodes,
        eta=eta,
        use_adaptive_penalties=use_adaptive_penalties,
        warmstart_h=warmstart_h,
        solver_profile=solver_profile,
        enable_domain_escalation=enable_domain_escalation,
        mission_domain=mission_domain,
    )

    # 4. 可选：导出收敛日志到 CSV
    if export_convergence_log:
        _export_scvx_convergence_log(
            result,
            scenario_id,
            eta,
            convergence_log_path,
        )

    return result


def _export_scvx_convergence_log(
    result: RecoverySegmentResult,
    scenario_id: str,
    eta: float | None,
    log_path: Path | None,
) -> None:
    """
    导出 SCvx 收敛日志到 CSV 文件。

    # NOTE: 用于支持 Part E 的 SCvx 收敛分析功能。
    # 日志格式与 scripts/make_figs_ch4_scvx_convergence.py 期望的格式一致。
    """
    import csv

    # 确定输出路径
    if log_path is None:
        eta_str = f"eta{eta:.2f}" if eta is not None else "eta_none"
        log_path = Path(f"outputs/data/ch4_scvx_convergence_{scenario_id}_{eta_str}.csv")

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 从 diagnostics 中获取迭代日志（如果存在）
    diag = result.diagnostics
    scvx_logs = diag.get("scvx_logs", [])

    if not scvx_logs:
        # 如果没有详细迭代日志，创建单行摘要
        scvx_logs = [{
            "iter_idx": 0,
            "total_cost": 0.0,
            "feas_violation": diag.get("final_feas_violation", 0.0),
            "cost_state": 0.0,
            "cost_control": 0.0,
            "cost_slack": 0.0,
            "cost_terminal": 0.0,
            "solver_status": diag.get("solver_status", "unknown"),
        }]

    # 写入 CSV
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # CSV header
        writer.writerow([
            "iter_idx",
            "total_cost",
            "cost_state",
            "cost_control",
            "cost_slack",
            "cost_terminal",
            "feas_violation",
            "solver_status",
        ])
        for entry in scvx_logs:
            if hasattr(entry, "iter_idx"):
                # IterationLog 对象
                writer.writerow([
                    entry.iter_idx,
                    getattr(entry, "total_cost", 0.0),
                    getattr(entry, "cost_state", 0.0),
                    getattr(entry, "cost_control", 0.0),
                    getattr(entry, "cost_slack", 0.0),
                    getattr(entry, "cost_terminal", 0.0),
                    getattr(entry, "feasibility_violation", 0.0),
                    getattr(entry, "solver_status", "unknown"),
                ])
            else:
                # dict
                writer.writerow([
                    entry.get("iter_idx", 0),
                    entry.get("total_cost", 0.0),
                    entry.get("cost_state", 0.0),
                    entry.get("cost_control", 0.0),
                    entry.get("cost_slack", 0.0),
                    entry.get("cost_terminal", 0.0),
                    entry.get("feas_violation", 0.0),
                    entry.get("solver_status", "unknown"),
                ])

    print(f"[INFO] SCvx 收敛日志已导出: {log_path}")
