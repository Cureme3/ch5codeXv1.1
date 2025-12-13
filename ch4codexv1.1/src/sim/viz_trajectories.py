#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chapter 4 visualization helper: three-trajectory joint extraction.

This module provides unified interface for extracting three trajectories:
1. Nominal trajectory (名义轨迹)
2. Fault open-loop trajectory (故障开环轨迹)
3. SCvx reconfigured trajectory (SCvx重规划轨迹)

All trajectories are interpolated to a common time grid for easy visualization.
本模块服务于第四章多故障、多严重度的轨迹可视化。

Updated for 4000s simulation with Hohmann transfer orbit insertion:
- S4 ignition at ~287s
- BURN1: Raise apoapsis to 500km
- COAST: Coast to apoapsis (~2700s)
- BURN2: Circularize at apoapsis (~3500s)
- Total simulation: 4000s
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import sys

import numpy as np

# Ensure src package importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.sim.run_nominal import simulate_full_mission, NominalResult, R_EARTH
from src.sim.kz1a_eci_core import std_atm_1976, omega_earth, g0
from src.sim.run_fault import (
    simulate_fault_open_loop,
    plan_recovery_segment_scvx,
    FaultSimResult,
    RecoverySegmentResult,
)
from src.sim.scenarios import get_scenario, FaultScenario, scale_scenario_by_eta
from src.sim.mission_domains import MissionDomain, choose_initial_domain


@dataclass
class ThreeTrajectories:
    """Container for three trajectories (nominal/fault/reconfig) data.

    所有轨迹已插值到统一时间网格 t 上，供第四章可视化使用。
    支持4000s仿真时长和ECI坐标系3D轨迹。

    Attributes
    ----------
    t : np.ndarray
        统一时间网格 [s]，范围 0-4000s
    h_nom, h_fault, h_reconfig : np.ndarray
        高度 [km]
    v_nom, v_fault, v_reconfig : np.ndarray
        速度 [km/s]
    s_nom, s_fault, s_reconfig : np.ndarray
        下行程 [km]
    q_nom, q_fault, q_reconfig : np.ndarray
        动压 [kPa]
    n_nom, n_fault, n_reconfig : np.ndarray
        法向过载 [g]
    gamma_nom, gamma_fault, gamma_reconfig : np.ndarray
        弹道倾角 [deg]
    r_eci_nom, r_eci_fault, r_eci_reconfig : np.ndarray
        ECI位置向量 [m]，形状 (N, 3)
    v_eci_nom, v_eci_fault, v_eci_reconfig : np.ndarray
        ECI速度向量 [m/s]，形状 (N, 3)
    fault_id : str
        故障场景 ID
    eta : float
        故障严重度 (0~1)
    mission_domain : MissionDomain
        任务域
    """

    t: np.ndarray

    h_nom: np.ndarray
    h_fault: np.ndarray
    h_reconfig: np.ndarray

    v_nom: np.ndarray
    v_fault: np.ndarray
    v_reconfig: np.ndarray

    s_nom: np.ndarray
    s_fault: np.ndarray
    s_reconfig: np.ndarray

    q_nom: np.ndarray
    q_fault: np.ndarray
    q_reconfig: np.ndarray

    n_nom: np.ndarray
    n_fault: np.ndarray
    n_reconfig: np.ndarray

    gamma_nom: np.ndarray
    gamma_fault: np.ndarray
    gamma_reconfig: np.ndarray

    # ECI 3D轨迹数据
    r_eci_nom: np.ndarray = None
    r_eci_fault: np.ndarray = None
    r_eci_reconfig: np.ndarray = None
    v_eci_nom: np.ndarray = None
    v_eci_fault: np.ndarray = None
    v_eci_reconfig: np.ndarray = None

    fault_id: str = ""
    eta: float = 0.0
    mission_domain: MissionDomain = None


def _interp_to_grid(t_orig: np.ndarray, y_orig: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Interpolate y_orig from t_orig to unified time grid t_grid."""
    return np.interp(t_grid, t_orig, y_orig)


def _compute_q_n_from_states(
    t: np.ndarray,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """根据 3-DoF 状态序列近似计算动压 q 和法向过载 n。

    参数:
        t: 时间序列 [N]
        states: 状态序列 [N, nx]，包含 [rx, ry, rz, vx, vy, vz, m]

    返回:
        q_kpa: 动压 (kPa)
        n_g:   法向过载 (g)
    """
    N = states.shape[0]
    q_kpa = np.zeros(N, dtype=float)
    n_g = np.ones(N, dtype=float)  # Default to ~1g

    for i in range(N):
        # Extract position and velocity
        r = states[i, 0:3]
        v = states[i, 3:6]

        # Compute altitude
        r_norm = np.linalg.norm(r)
        h_m = r_norm - R_EARTH  # altitude in meters

        # Get atmospheric properties
        if h_m < 0:
            h_m = 0.0
        T, P, rho, a = std_atm_1976(h_m)

        # Compute relative velocity (accounting for Earth rotation)
        # v_atm = omega_earth × r
        omega_vec = np.array([0.0, 0.0, omega_earth])
        v_atm = np.cross(omega_vec, r)
        v_rel = v - v_atm
        v_rel_mag = np.linalg.norm(v_rel)

        # Dynamic pressure: q = 0.5 * rho * v^2
        q_pa = 0.5 * rho * v_rel_mag ** 2
        q_kpa[i] = q_pa / 1000.0  # Pa -> kPa

        # Normal load approximation:
        # n = sqrt(a_total^2 - a_tangential^2) / g0
        # For a simplified estimate, use centrifugal + gravity balance
        if r_norm > 1e-6 and v_rel_mag > 1e-3:
            # Centripetal acceleration magnitude
            v_horizontal = np.sqrt(max(0, v_rel_mag**2 - (np.dot(v_rel, r/r_norm))**2))
            a_centripetal = v_horizontal**2 / r_norm

            # Gravity
            a_gravity = g0 * (R_EARTH / r_norm)**2

            # Simplified normal load: combination of centripetal and gravity
            # In level flight: n ≈ 1 + v^2/(g*r) - 1 = v^2/(g*r)
            # But for ascending flight, it's more complex
            # Use a heuristic: n ≈ sqrt(1 + (a_centripetal/g0)^2)
            n_g[i] = np.sqrt(1.0 + (a_centripetal / g0)**2)
        else:
            n_g[i] = 1.0

    return q_kpa, n_g


def _compute_downrange(states: np.ndarray) -> np.ndarray:
    """Compute cumulative downrange distance (km) from states.

    Uses arc length along Earth surface based on position vectors.
    """
    if states.shape[0] < 2:
        return np.zeros(states.shape[0])

    # Position vectors
    r = states[:, 0:3]
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    r_norm = np.maximum(r_norm, 1.0)  # avoid division by zero
    r_unit = r / r_norm

    # Angular displacement between consecutive points
    dot_products = np.sum(r_unit[:-1] * r_unit[1:], axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products)

    # Arc length on Earth surface (km)
    arc_lengths = angles * (R_EARTH / 1000.0)

    # Cumulative downrange
    downrange = np.zeros(len(states))
    downrange[1:] = np.cumsum(arc_lengths)

    return downrange


def _extract_trajectory_data(
    time: np.ndarray,
    states: np.ndarray,
    altitude_km: np.ndarray,
    speed_kms: np.ndarray,
    dynamic_pressure_kpa: np.ndarray,
    normal_load_g: np.ndarray,
    flight_path_deg: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract trajectory data arrays from simulation result.

    Returns
    -------
    time, h, v, s, q, n, gamma : np.ndarray
    """
    downrange = _compute_downrange(states)
    gamma = flight_path_deg if flight_path_deg is not None else np.zeros_like(time)
    return time, altitude_km, speed_kms, downrange, dynamic_pressure_kpa, normal_load_g, gamma


def build_three_trajectories(
    fault_id: str,
    eta: float,
    mission_domain: Optional[MissionDomain] = None,
    t_step: float = 1.0,
    t_end: Optional[float] = None,
    nodes: int = 40,
    solver_profile: str = "fast",
) -> ThreeTrajectories:
    """根据故障场景 ID 和故障严重度 eta，构造名义轨迹/故障开环/SCvx重规划三条轨迹。

    本函数服务于第四章多故障、多严重度的轨迹可视化。

    Parameters
    ----------
    fault_id : str
        故障场景 ID（如 "F1_thrust_deg15"）
    eta : float
        故障严重度，范围 [0, 1]
    mission_domain : MissionDomain, optional
        任务域。若为 None，则使用 choose_initial_domain(eta) 自动选择。
    t_step : float
        统一时间网格步长 [s]，默认 1.0
    t_end : float, optional
        时间网格终点 [s]。若为 None，自动确定。
    nodes : int
        SCvx 规划节点数，默认 40
    solver_profile : str
        求解器配置，默认 "fast"

    Returns
    -------
    ThreeTrajectories
        包含三条轨迹数据的容器
    """
    # Get fault scenario and apply eta scaling
    scenario_base = get_scenario(fault_id)
    scenario = scale_scenario_by_eta(scenario_base, eta)

    # Determine mission domain if not provided
    if mission_domain is None:
        mission_domain = choose_initial_domain(eta)

    # 1. Run nominal simulation
    print(f"[viz] Running nominal simulation...")
    nominal: NominalResult = simulate_full_mission(dt=t_step, save_csv=False)

    # 2. Run fault open-loop simulation
    print(f"[viz] Running fault open-loop: {fault_id}...")
    fault_sim: FaultSimResult = simulate_fault_open_loop(
        scenario=scenario,
        dt=t_step,
    )

    # 3. Run SCvx reconfiguration
    print(f"[viz] Running SCvx reconfig (eta={eta:.2f}, domain={mission_domain.name})...")
    recovery: RecoverySegmentResult = plan_recovery_segment_scvx(
        scenario=scenario,
        fault_sim=fault_sim,
        nominal=nominal,
        nodes=nodes,
        fault_eta=eta,
        mission_domain=mission_domain,
        enable_domain_escalation=False,
        solver_profile=solver_profile,
    )

    # Extract nominal trajectory data
    t_nom, h_nom, v_nom, s_nom, q_nom, n_nom, gamma_nom = _extract_trajectory_data(
        nominal.time,
        nominal.states,
        nominal.altitude_km,
        nominal.speed_kms,
        nominal.dynamic_pressure_kpa,
        nominal.normal_load_g,
        nominal.flight_path_deg,
    )

    # Extract fault trajectory data
    t_fault, h_fault, v_fault, s_fault, q_fault, n_fault, gamma_fault = _extract_trajectory_data(
        fault_sim.time,
        fault_sim.states,
        fault_sim.altitude_km,
        fault_sim.speed_kms,
        fault_sim.dynamic_pressure_kpa,
        fault_sim.normal_load_g,
        fault_sim.flight_path_deg,
    )

    # Extract reconfig trajectory data
    reconfig_states = recovery.states
    reconfig_time = recovery.time

    # Compute derived quantities for reconfig segment
    r_reconfig = reconfig_states[:, 0:3]
    v_reconfig_vec = reconfig_states[:, 3:6]
    r_norm = np.linalg.norm(r_reconfig, axis=1)
    h_reconfig_raw = (r_norm - R_EARTH) / 1000.0
    v_reconfig_raw = np.linalg.norm(v_reconfig_vec, axis=1) / 1000.0
    s_reconfig_raw = _compute_downrange(reconfig_states)

    # Compute q and n from reconfig states using atmospheric model
    q_reconfig_raw, n_reconfig_raw = _compute_q_n_from_states(reconfig_time, reconfig_states)

    # Compute flight path angle for reconfig
    gamma_reconfig_raw = np.zeros_like(reconfig_time)
    for i in range(len(reconfig_time)):
        r_i = reconfig_states[i, 0:3]
        v_i = reconfig_states[i, 3:6]
        r_norm_i = np.linalg.norm(r_i)
        v_norm_i = np.linalg.norm(v_i)
        if r_norm_i > 1e-6 and v_norm_i > 1e-6:
            r_hat = r_i / r_norm_i
            v_radial = np.dot(v_i, r_hat)
            v_horizontal = np.sqrt(max(0, v_norm_i**2 - v_radial**2))
            gamma_reconfig_raw[i] = np.degrees(np.arctan2(v_radial, v_horizontal + 1e-6))

    # Determine time grid - default to 4000s for full Hohmann transfer simulation
    if t_end is None:
        t_end = max(
            t_nom[-1] if len(t_nom) > 0 else 4000.0,
            t_fault[-1] if len(t_fault) > 0 else 4000.0,
            reconfig_time[-1] if len(reconfig_time) > 0 else 0,
            4000.0,  # Default to full simulation duration
        )

    t_grid = np.arange(0.0, t_end + t_step, t_step)

    # Interpolate nominal to grid
    h_nom_interp = _interp_to_grid(t_nom, h_nom, t_grid)
    v_nom_interp = _interp_to_grid(t_nom, v_nom, t_grid)
    s_nom_interp = _interp_to_grid(t_nom, s_nom, t_grid)
    q_nom_interp = _interp_to_grid(t_nom, q_nom, t_grid)
    n_nom_interp = _interp_to_grid(t_nom, n_nom, t_grid)
    gamma_nom_interp = _interp_to_grid(t_nom, gamma_nom, t_grid)

    # Interpolate fault to grid
    h_fault_interp = _interp_to_grid(t_fault, h_fault, t_grid)
    v_fault_interp = _interp_to_grid(t_fault, v_fault, t_grid)
    s_fault_interp = _interp_to_grid(t_fault, s_fault, t_grid)
    q_fault_interp = _interp_to_grid(t_fault, q_fault, t_grid)
    n_fault_interp = _interp_to_grid(t_fault, n_fault, t_grid)
    gamma_fault_interp = _interp_to_grid(t_fault, gamma_fault, t_grid)

    # Reconfig: use fault trajectory outside planning window, SCvx inside
    t0_reconfig = recovery.t0_s
    tf_reconfig = recovery.tf_s

    # Start with fault trajectory as base
    h_reconfig_interp = h_fault_interp.copy()
    v_reconfig_interp = v_fault_interp.copy()
    s_reconfig_interp = s_fault_interp.copy()
    q_reconfig_interp = q_fault_interp.copy()
    n_reconfig_interp = n_fault_interp.copy()
    gamma_reconfig_interp = gamma_fault_interp.copy()

    # Overlay SCvx segment
    mask = (t_grid >= t0_reconfig) & (t_grid <= tf_reconfig)
    if np.any(mask) and len(reconfig_time) > 0:
        h_reconfig_interp[mask] = _interp_to_grid(reconfig_time, h_reconfig_raw, t_grid[mask])
        v_reconfig_interp[mask] = _interp_to_grid(reconfig_time, v_reconfig_raw, t_grid[mask])
        s_reconfig_interp[mask] = _interp_to_grid(reconfig_time, s_reconfig_raw, t_grid[mask])
        q_reconfig_interp[mask] = _interp_to_grid(reconfig_time, q_reconfig_raw, t_grid[mask])
        n_reconfig_interp[mask] = _interp_to_grid(reconfig_time, n_reconfig_raw, t_grid[mask])
        gamma_reconfig_interp[mask] = _interp_to_grid(reconfig_time, gamma_reconfig_raw, t_grid[mask])

    # Extract ECI position/velocity for 3D trajectory plotting
    # Nominal ECI data
    r_eci_nom_raw = nominal.states[:, 0:3]  # (N, 3) in meters
    v_eci_nom_raw = nominal.states[:, 3:6]  # (N, 3) in m/s

    # Fault ECI data
    r_eci_fault_raw = fault_sim.states[:, 0:3]
    v_eci_fault_raw = fault_sim.states[:, 3:6]

    # Reconfig ECI data
    r_eci_reconfig_raw = reconfig_states[:, 0:3]
    v_eci_reconfig_raw = reconfig_states[:, 3:6]

    # Interpolate ECI data to grid (component-wise)
    def interp_eci_to_grid(t_orig, eci_orig, t_grid):
        """Interpolate 3D ECI vectors to time grid."""
        result = np.zeros((len(t_grid), 3))
        for i in range(3):
            result[:, i] = _interp_to_grid(t_orig, eci_orig[:, i], t_grid)
        return result

    r_eci_nom_interp = interp_eci_to_grid(t_nom, r_eci_nom_raw, t_grid)
    v_eci_nom_interp = interp_eci_to_grid(t_nom, v_eci_nom_raw, t_grid)
    r_eci_fault_interp = interp_eci_to_grid(t_fault, r_eci_fault_raw, t_grid)
    v_eci_fault_interp = interp_eci_to_grid(t_fault, v_eci_fault_raw, t_grid)

    # Reconfig ECI: use fault trajectory outside planning window, SCvx inside
    r_eci_reconfig_interp = r_eci_fault_interp.copy()
    v_eci_reconfig_interp = v_eci_fault_interp.copy()
    if np.any(mask) and len(reconfig_time) > 0:
        r_eci_reconfig_interp[mask] = interp_eci_to_grid(reconfig_time, r_eci_reconfig_raw, t_grid[mask])
        v_eci_reconfig_interp[mask] = interp_eci_to_grid(reconfig_time, v_eci_reconfig_raw, t_grid[mask])

    print(f"[viz] Trajectory construction done: t=[0, {t_end:.1f}]s, dt={t_step}s, {len(t_grid)} points")

    return ThreeTrajectories(
        t=t_grid,
        h_nom=h_nom_interp,
        h_fault=h_fault_interp,
        h_reconfig=h_reconfig_interp,
        v_nom=v_nom_interp,
        v_fault=v_fault_interp,
        v_reconfig=v_reconfig_interp,
        s_nom=s_nom_interp,
        s_fault=s_fault_interp,
        s_reconfig=s_reconfig_interp,
        q_nom=q_nom_interp,
        q_fault=q_fault_interp,
        q_reconfig=q_reconfig_interp,
        n_nom=n_nom_interp,
        n_fault=n_fault_interp,
        n_reconfig=n_reconfig_interp,
        gamma_nom=gamma_nom_interp,
        gamma_fault=gamma_fault_interp,
        gamma_reconfig=gamma_reconfig_interp,
        r_eci_nom=r_eci_nom_interp,
        r_eci_fault=r_eci_fault_interp,
        r_eci_reconfig=r_eci_reconfig_interp,
        v_eci_nom=v_eci_nom_interp,
        v_eci_fault=v_eci_fault_interp,
        v_eci_reconfig=v_eci_reconfig_interp,
        fault_id=fault_id,
        eta=eta,
        mission_domain=mission_domain,
    )


def get_default_fault_ids() -> List[str]:
    """Return default fault scenario IDs (F1~F5)."""
    return [
        "F1_thrust_deg15",
        "F2_tvc_rate4",
        "F3_tvc_stuck3deg",
        "F4_sensor_bias2deg",
        "F5_event_delay5s",
    ]


def get_default_etas() -> List[float]:
    """Return default eta values for multi-severity analysis."""
    return [0.2, 0.5, 0.8]


def load_three_trajectories_from_npz(
    fault_id: str,
    eta: float,
    data_dir: Optional[Path] = None,
    t_step: float = 1.0,
) -> ThreeTrajectories:
    """从已生成的 npz 文件加载三轨迹数据（快速模式）。

    Parameters
    ----------
    fault_id : str
        故障场景 ID（如 "F1_thrust_deg15"）
    eta : float
        故障严重度
    data_dir : Path, optional
        数据目录，默认 outputs/data/ch4_trajectories_replan
    t_step : float
        时间网格步长 [s]

    Returns
    -------
    ThreeTrajectories
        从 npz 文件加载的三轨迹数据
    """
    if data_dir is None:
        data_dir = ROOT / "outputs" / "data" / "ch4_trajectories_replan"

    # 文件名格式: F1_eta02_openloop.npz, F1_eta02_replan.npz
    fault_short = fault_id.split("_")[0]  # F1_thrust_deg15 -> F1
    eta_str = f"{int(eta*10):02d}"  # 0.2 -> 02

    nom_path = data_dir / "nominal.npz"
    openloop_path = data_dir / f"{fault_short}_eta{eta_str}_openloop.npz"
    replan_path = data_dir / f"{fault_short}_eta{eta_str}_replan.npz"

    # 加载数据
    nom_data = np.load(nom_path)
    openloop_data = np.load(openloop_path)
    replan_data = np.load(replan_path)

    # 提取时间和轨迹数据（键名为 't' 而非 'time'）
    t_nom = nom_data['t']
    t_fault = openloop_data['t']
    t_replan = replan_data['t']

    # 确定统一时间网格
    t_end = max(t_nom[-1], t_fault[-1], t_replan[-1], 4000.0)
    t_grid = np.arange(0.0, t_end + t_step, t_step)

    # 插值函数
    def interp(t_orig, y_orig, t_grid):
        return np.interp(t_grid, t_orig, y_orig)

    def interp_eci(t_orig, eci_orig, t_grid):
        result = np.zeros((len(t_grid), 3))
        for i in range(3):
            result[:, i] = np.interp(t_grid, t_orig, eci_orig[:, i])
        return result

    # 从ECI数据计算速度
    def compute_velocity_from_eci(v_eci):
        """从ECI速度向量计算速度大小 (km/s)"""
        return np.linalg.norm(v_eci, axis=1) / 1000.0

    # 标称轨迹
    h_nom = interp(t_nom, nom_data['altitude'], t_grid)
    v_eci_nom_raw = nom_data['v_eci']
    v_nom = interp(t_nom, compute_velocity_from_eci(v_eci_nom_raw), t_grid)
    s_nom = interp(t_nom, nom_data['downrange'], t_grid)
    q_nom = np.zeros_like(t_grid)  # 动压需要大气模型，暂时置零
    n_nom = np.ones_like(t_grid)   # 过载暂时置1
    gamma_nom = np.zeros_like(t_grid)

    # 故障开环轨迹
    h_fault = interp(t_fault, openloop_data['altitude'], t_grid)
    v_eci_fault_raw = openloop_data['v_eci']
    v_fault = interp(t_fault, compute_velocity_from_eci(v_eci_fault_raw), t_grid)
    s_fault = interp(t_fault, openloop_data['downrange'], t_grid)
    q_fault = np.zeros_like(t_grid)
    n_fault = np.ones_like(t_grid)
    gamma_fault = np.zeros_like(t_grid)

    # 重规划轨迹：故障段 + SCvx段
    h_reconfig = h_fault.copy()
    v_reconfig = v_fault.copy()
    s_reconfig = s_fault.copy()
    q_reconfig = q_fault.copy()
    n_reconfig = n_fault.copy()
    gamma_reconfig = gamma_fault.copy()

    # 覆盖SCvx段
    t0_replan = t_replan[0]
    tf_replan = t_replan[-1]
    mask = (t_grid >= t0_replan) & (t_grid <= tf_replan)
    if np.any(mask):
        h_reconfig[mask] = interp(t_replan, replan_data['altitude'], t_grid[mask])
        if 'v_eci' in replan_data:
            v_eci_replan_raw = replan_data['v_eci']
            v_reconfig[mask] = interp(t_replan, compute_velocity_from_eci(v_eci_replan_raw), t_grid[mask])
        s_reconfig[mask] = interp(t_replan, replan_data['downrange'], t_grid[mask])

    # ECI数据
    r_eci_nom = r_eci_fault = r_eci_reconfig = None
    v_eci_nom = v_eci_fault = v_eci_reconfig = None

    if 'r_eci' in nom_data:
        r_eci_nom = interp_eci(t_nom, nom_data['r_eci'], t_grid)
        v_eci_nom = interp_eci(t_nom, nom_data['v_eci'], t_grid)
    if 'r_eci' in openloop_data:
        r_eci_fault = interp_eci(t_fault, openloop_data['r_eci'], t_grid)
        v_eci_fault = interp_eci(t_fault, openloop_data['v_eci'], t_grid)
        r_eci_reconfig = r_eci_fault.copy()
        v_eci_reconfig = v_eci_fault.copy()
        if 'r_eci' in replan_data and np.any(mask):
            r_eci_reconfig[mask] = interp_eci(t_replan, replan_data['r_eci'], t_grid[mask])
            v_eci_reconfig[mask] = interp_eci(t_replan, replan_data['v_eci'], t_grid[mask])

    # 任务域
    domain_str = str(replan_data.get('mission_domain', 'RETAIN'))
    try:
        mission_domain = MissionDomain[domain_str]
    except KeyError:
        mission_domain = MissionDomain.RETAIN

    return ThreeTrajectories(
        t=t_grid,
        h_nom=h_nom, h_fault=h_fault, h_reconfig=h_reconfig,
        v_nom=v_nom, v_fault=v_fault, v_reconfig=v_reconfig,
        s_nom=s_nom, s_fault=s_fault, s_reconfig=s_reconfig,
        q_nom=q_nom, q_fault=q_fault, q_reconfig=q_reconfig,
        n_nom=n_nom, n_fault=n_fault, n_reconfig=n_reconfig,
        gamma_nom=gamma_nom, gamma_fault=gamma_fault, gamma_reconfig=gamma_reconfig,
        r_eci_nom=r_eci_nom, r_eci_fault=r_eci_fault, r_eci_reconfig=r_eci_reconfig,
        v_eci_nom=v_eci_nom, v_eci_fault=v_eci_fault, v_eci_reconfig=v_eci_reconfig,
        fault_id=fault_id, eta=eta, mission_domain=mission_domain,
    )
