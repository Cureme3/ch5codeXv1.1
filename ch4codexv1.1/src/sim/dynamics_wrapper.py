# ch4codexv1.1/src/sim/dynamics_wrapper.py
"""
3-DoF dynamics wrapper with fault injection for SCvx trajectory optimization.

This module wraps the KZ-1A ECI simulation and provides fault-aware dynamics
for F1-F5 fault scenarios:
- F1: Thrust degradation
- F2: TVC rate limit (direction cannot change fast enough)
- F3: TVC stuck (thrust direction locked)
- F4: Sensor bias (thrust direction has fixed offset)
- F5: Event delay (thrust reduced during delay period)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# Ensure src is importable when running from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.sim.kz1a_eci_core import (  # noqa: E402
    KZ1AConfig,
    Re,
    g0,
    mu,
    omega_earth,
    simulate_kz1a_eci,
    std_atm_1976,
)


# ============================================================================
# Helper functions for fault modeling
# ============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / norm


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle (radians) between two vectors."""
    v1_n = normalize(v1)
    v2_n = normalize(v2)
    dot = np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)
    return np.arccos(dot)


def rate_limit_direction(u_prev: np.ndarray, u_target: np.ndarray,
                         rate_limit_deg_s: float, dt: float) -> np.ndarray:
    """
    Limit the rate of change of thrust direction.

    F2 fault model: TVC cannot slew faster than rate_limit_deg_s.
    """
    max_dtheta = np.deg2rad(rate_limit_deg_s) * dt
    dtheta = angle_between(u_prev, u_target)

    if dtheta <= max_dtheta + 1e-9:
        return normalize(u_target)

    # Spherical interpolation toward target, limited by max rate
    ratio = max_dtheta / dtheta
    interpolated = (1.0 - ratio) * normalize(u_prev) + ratio * normalize(u_target)
    return normalize(interpolated)


def rotate_vector(v: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate vector v around axis by angle_deg degrees (Rodrigues' formula).

    F4 fault model: Sensor bias causes thrust direction to be rotated.
    """
    angle = np.deg2rad(angle_deg)
    axis = normalize(np.array(axis, dtype=float))
    v = np.array(v, dtype=float)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Rodrigues' rotation formula
    return (v * cos_a +
            np.cross(axis, v) * sin_a +
            axis * np.dot(axis, v) * (1.0 - cos_a))


def compute_pitch_rotation_axis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the axis perpendicular to the orbital plane for pitch rotation.

    This is the cross product of position and velocity vectors (angular momentum direction).
    Rotating around this axis changes the pitch angle (in-plane rotation).
    """
    h = np.cross(r, v)  # Angular momentum vector
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-12:
        # Fallback to Y-axis if vectors are parallel
        return np.array([0.0, 1.0, 0.0])
    return h / h_norm


class Dynamics3DOF:
    """
    3-DOF wrapper that reuses the unified KZ-1A ascent model.
    State: [rx, ry, rz, vx, vy, vz, m]; Control: thrust vector (N).
    """

    state_dim = 7
    control_dim = 3

    def __init__(self, dt: float = 0.5, rocket_params=None, scenario=None):
        self.dt = dt
        self.scenario = scenario

        cfg = KZ1AConfig(preset="nasaspaceflight", dt=dt, t_end=1100.0)
        sim_data = simulate_kz1a_eci(cfg, fault=None)
        if sim_data is None:
            raise RuntimeError("simulate_kz1a_eci returned None")

        self.time_nom = sim_data["t"]
        self.thrust_interp = interp1d(self.time_nom, sim_data["thrust"], kind="linear", fill_value="extrapolate")
        self.mass_interp = interp1d(self.time_nom, sim_data["mass"], kind="linear", fill_value="extrapolate")
        self.r_nom = sim_data["r_eci"]
        self.v_nom = sim_data["v_eci"]

        self.r0 = self.r_nom[0]
        self.v0 = self.v_nom[0]
        self.mass0 = sim_data["mass"][0]

    def get_nominal_thrust_mag(self, t: float) -> float:
        if t < 0:
            return 0.0
        if t > self.time_nom[-1]:
            return 0.0
        return float(self.thrust_interp(t))

    def get_mass(self, t: float) -> float:
        if t < 0:
            return float(self.mass_interp(0.0))
        if t > self.time_nom[-1]:
            return float(self.mass_interp(self.time_nom[-1]))
        return float(self.mass_interp(t))

    def deriv(self, t, x):
        """Compute state derivative with fault injection for F1-F5.

        Fault models:
        - F1 (thrust_degradation): Reduce thrust magnitude after t_fault_s
        - F2 (tvc_rate_limit): Limit TVC slew rate + apply angle bias
        - F3 (tvc_stuck): Lock thrust direction at fault time
        - F4 (sensor_bias): Rotate thrust direction by bias angle
        - F5 (event_delay): Reduce thrust during delay period
        """
        thrust_mag = self.get_nominal_thrust_mag(t)
        scenario = self.scenario

        # Extract position and velocity
        r = x[0:3]
        v = x[3:6]

        # Compute nominal thrust direction (velocity-aligned)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-3:
            u_dir_nom = v / v_norm
        else:
            u_dir_nom = np.array([0.0, 0.0, 1.0])

        # Compute pitch rotation axis (perpendicular to orbital plane)
        # This is the correct axis for thrust direction perturbations
        pitch_axis = compute_pitch_rotation_axis(r, v)

        # Default: use nominal direction
        u_dir = u_dir_nom.copy()

        # Apply fault models if scenario is set
        if scenario is not None:
            fault_type = getattr(scenario, "fault_type", None)
            params = getattr(scenario, "params", {})
            t_fault_s = getattr(scenario, "t_fault_s", 0.0)

            # F1: Thrust degradation
            if fault_type == "thrust_degradation":
                if t >= t_fault_s:
                    frac = float(params.get("degrade_frac", 0.0))
                    thrust_mag *= max(0.0, 1.0 - frac)
                    # Also apply cone angle restriction (tighter pointing)
                    # This creates a persistent pitch bias effect
                    theta_max = float(params.get("theta_max_deg_after", 8.0))
                    if theta_max < 8.0:
                        # Apply a pitch-down bias proportional to severity
                        bias_deg = (8.0 - theta_max) * 0.8  # ~2.5 deg at theta_max=5
                        u_dir = rotate_vector(u_dir_nom, axis=pitch_axis, angle_deg=-bias_deg)

            # F2: TVC rate limit + angle bias
            elif fault_type == "tvc_rate_limit":
                if t >= t_fault_s:
                    rate_limit_deg_s = float(params.get("tvc_rate_deg_s", 10.0))
                    angle_bias_deg = float(params.get("angle_bias_deg", 0.0))

                    # Apply angle bias first (pitch bias in orbital plane)
                    if abs(angle_bias_deg) > 1e-6:
                        u_dir_biased = rotate_vector(u_dir_nom, axis=pitch_axis, angle_deg=angle_bias_deg)
                    else:
                        u_dir_biased = u_dir_nom

                    # Apply rate limiting
                    if not hasattr(self, "_u_prev"):
                        self._u_prev = u_dir_nom.copy()
                    u_dir = rate_limit_direction(self._u_prev, u_dir_biased, rate_limit_deg_s, self.dt)
                    self._u_prev = u_dir.copy()

            # F3: TVC stuck at fault time direction
            elif fault_type == "tvc_stuck":
                if t >= t_fault_s:
                    stuck_angle_deg = float(params.get("stuck_angle_deg", 0.0))
                    if not hasattr(self, "_u_stuck"):
                        # Lock direction at fault time, rotated by stuck angle in pitch direction
                        # Store the stuck axis too, since it's computed at fault time
                        self._stuck_axis = pitch_axis.copy()
                        self._u_stuck = rotate_vector(u_dir_nom, axis=pitch_axis, angle_deg=stuck_angle_deg)
                    u_dir = self._u_stuck

            # F4: Sensor bias (thrust direction has fixed offset)
            elif fault_type == "sensor_bias":
                if t >= t_fault_s:
                    bias_deg = float(params.get("sensor_bias_deg", 0.0))
                    # Sensor bias causes continuous pitch error in orbital plane
                    u_dir = rotate_vector(u_dir_nom, axis=pitch_axis, angle_deg=bias_deg)

            # F5: Event delay (reduced thrust during delay window)
            elif fault_type == "event_delay":
                delay_s = float(params.get("event_delay_s", 0.0))
                if t_fault_s <= t <= t_fault_s + delay_s:
                    # Significantly reduce thrust during delay period
                    delay_thrust_scale = float(params.get("delay_thrust_scale", 0.15))
                    thrust_mag *= delay_thrust_scale

        # Compute final thrust vector
        u_dir = normalize(u_dir)
        u = thrust_mag * u_dir
        return self.eom(t, x, u)

    def eom(self, t, x, u=None):
        if u is None:
            u = np.zeros(3)

        r = x[0:3]
        v = x[3:6]
        m = max(x[6], 1.0)

        r_norm = np.linalg.norm(r)
        g_vec = -mu * r / (r_norm**3)

        # Simple aero model (atmosphere rotates with Earth)
        h = r_norm - Re
        _, _, rho, _ = std_atm_1976(max(0.0, h))
        v_atm = np.cross(np.array([0.0, 0.0, omega_earth]), r)
        v_rel = v - v_atm
        q_dyn = 0.5 * rho * np.dot(v_rel, v_rel)
        # No explicit Cd/area here; keep aero zero but retain slot for future use
        a_aero = np.zeros(3)

        a_thrust = u / m

        dr = v
        dv = a_thrust + g_vec + a_aero
        dm = 0.0  # mass propagation handled by nominal table in SCvx

        return np.concatenate([dr, dv, [dm]])

    def reset_fault_state(self):
        """Reset internal fault state for a new simulation run."""
        if hasattr(self, "_u_prev"):
            del self._u_prev
        if hasattr(self, "_u_stuck"):
            del self._u_stuck
        if hasattr(self, "_stuck_axis"):
            del self._stuck_axis


def simulate_fault_open_loop_3dof(
    scenario,
    duration_s: float = 1000.0,
    dt: float = 1.0,
) -> dict:
    """
    使用 Dynamics3DOF 进行故障开环仿真。

    策略：
    1. 故障发生前（t < t_fault_s）：直接使用名义轨迹数据
    2. 故障发生后（t >= t_fault_s）：从故障时刻状态开始积分，应用故障效果

    这种方法确保故障开环轨迹与名义轨迹在故障前完全一致，
    故障后则根据故障模型（F1-F5）产生偏离。

    Parameters
    ----------
    scenario : FaultScenario
        故障场景对象
    duration_s : float
        仿真时长（秒）
    dt : float
        时间步长（秒）

    Returns
    -------
    dict
        包含 time, states, altitude_km, speed_kms 等的字典
    """
    from scipy.integrate import solve_ivp

    # 创建带故障的动力学对象
    dyn = Dynamics3DOF(dt=dt, scenario=scenario)
    dyn.reset_fault_state()

    # 获取故障发生时间
    t_fault_s = getattr(scenario, "t_fault_s", 0.0)

    # 获取名义轨迹的时间数组
    t_nom = dyn.time_nom
    r_nom = dyn.r_nom
    v_nom = dyn.v_nom

    # 限制仿真时长不超过名义轨迹
    max_t = min(duration_s, t_nom[-1])

    # 生成时间数组
    t_eval = np.arange(0.0, max_t + dt, dt)
    t_eval = t_eval[t_eval <= max_t]

    # --- Part 1: 故障前使用名义轨迹 ---
    pre_fault_mask = t_eval < t_fault_s
    t_pre = t_eval[pre_fault_mask]

    # 插值获取故障前的名义轨迹状态
    if len(t_pre) > 0:
        r_interp = interp1d(t_nom, r_nom, axis=0, kind="linear", fill_value="extrapolate")
        v_interp = interp1d(t_nom, v_nom, axis=0, kind="linear", fill_value="extrapolate")
        r_pre = r_interp(t_pre)
        v_pre = v_interp(t_pre)
        m_pre = np.array([dyn.get_mass(t) for t in t_pre])
        states_pre = np.column_stack([r_pre, v_pre, m_pre])  # (N_pre, 7)
    else:
        states_pre = np.empty((0, 7))

    # --- Part 2: 故障后进行ODE积分 ---
    post_fault_mask = t_eval >= t_fault_s
    t_post = t_eval[post_fault_mask]

    if len(t_post) > 0:
        # 获取故障时刻的状态作为初始条件
        r_interp = interp1d(t_nom, r_nom, axis=0, kind="linear", fill_value="extrapolate")
        v_interp = interp1d(t_nom, v_nom, axis=0, kind="linear", fill_value="extrapolate")
        r_fault = r_interp(t_fault_s)
        v_fault = v_interp(t_fault_s)
        m_fault = dyn.get_mass(t_fault_s)
        x_fault = np.concatenate([r_fault, v_fault, [m_fault]])

        # 定义 ODE 函数
        def ode_func(t, x):
            return dyn.deriv(t, x)

        # 使用 RK45 积分故障后段
        t_span = (t_fault_s, max_t)
        sol = solve_ivp(
            ode_func,
            t_span,
            x_fault,
            method="RK45",
            t_eval=t_post,
            max_step=dt,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            print(f"[WARN] ODE solver failed: {sol.message}")

        t_post_actual = sol.t
        states_post = sol.y.T  # (N_post, 7)
    else:
        t_post_actual = np.array([])
        states_post = np.empty((0, 7))

    # --- 合并故障前后轨迹 ---
    if len(t_pre) > 0 and len(t_post_actual) > 0:
        time = np.concatenate([t_pre, t_post_actual])
        states = np.vstack([states_pre, states_post])
    elif len(t_pre) > 0:
        time = t_pre
        states = states_pre
    else:
        time = t_post_actual
        states = states_post

    # 计算派生量
    r = states[:, 0:3]
    v = states[:, 3:6]
    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)

    altitude_km = (r_norm - Re) / 1000.0
    speed_kms = v_norm / 1000.0

    # 检测坠毁（高度 < 0）
    crash_idx = np.where(altitude_km < 0)[0]
    if len(crash_idx) > 0:
        crash_t = time[crash_idx[0]]
        print(f"CRASH: Altitude {altitude_km[crash_idx[0]]} < 0 at t={crash_t}")
        # 截断到坠毁点
        end_idx = crash_idx[0] + 1
        time = time[:end_idx]
        states = states[:end_idx]
        altitude_km = altitude_km[:end_idx]
        speed_kms = speed_kms[:end_idx]

    # 计算飞行路径角
    flight_path_deg = np.zeros(len(time))
    for i in range(len(time)):
        ri = states[i, 0:3]
        vi = states[i, 3:6]
        r_hat = ri / (np.linalg.norm(ri) + 1e-12)
        v_horiz = vi - np.dot(vi, r_hat) * r_hat
        v_vert = np.dot(vi, r_hat)
        flight_path_deg[i] = np.rad2deg(np.arctan2(v_vert, np.linalg.norm(v_horiz) + 1e-12))

    return {
        "time": time,
        "states": states,
        "altitude_km": altitude_km,
        "speed_kms": speed_kms,
        "flight_path_deg": flight_path_deg,
        "scenario": scenario,
    }

