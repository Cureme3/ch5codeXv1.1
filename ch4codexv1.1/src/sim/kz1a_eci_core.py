# -*- coding: utf-8 -*-
"""
Unified KZ-1A ECI ascent dynamics core shared by chapters 2/3/4.
- Single source for vehicle parameters, timeline, atmosphere, and guidance.
- Supports nominal runs (fault=None) and fault-injected runs via FaultProfile.
- Revised for robust 500km SSO injection (High-Loft S1-S3 + Feedback S4).
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Physical constants
g0 = 9.80665
Re = 6371000.0
mu = 3.986004418e14
omega_earth = 7.2921159e-5
gamma_air = 1.4
R_air = 287.05287

# --- 1. Revised Guidance Profiles ---
# S1-S3 Open Loop Pitch Profile (Lofted trajectory strategy)
# Time (s) -> Pitch Angle (deg, 90=Vertical, 0=Horizontal)
# Note: Pitch is defined relative to local horizontal (Local-Level Frame)
DEFAULT_PITCH_PROFILE: List[Tuple[float, float]] = [
    (0.0, 90.000),    # Vertical launch
    (10.0, 89.000),   # Kick start
    (40.0, 67.000),   # S1 gravity turn start (Tuned)
    (80.0, 54.000),   # S1 Sep approx
    (120.0, 38.000),  # S2 Mid
    (160.0, 30.000),  # S2 Sep approx
    (200.0, 22.000),  # S3 Start
    (250.0, 18.000),  # S3 Mid
    (290.0, 14.000),  # S3 Sep (Precision Landing: 14.0 deg)
]

# (Optional) Flight Path Angle Reference - mostly for analysis, pitch profile drives S1-S3
DEFAULT_FPA_PROFILE: List[Tuple[float, float]] = [
    (0.0, 90.0),
    (290.0, 20.0),
]

# --- Standard atmosphere (US Std 1976, geopotential height) ---
_layers = [
    (0.0, -6.5, 288.15),
    (11.0, 0.0, 216.65),
    (20.0, 1.0, 216.65),
    (32.0, 2.8, 228.65),
    (47.0, 0.0, 270.65),
    (51.0, -2.8, 270.65),
    (71.0, -2.0, 214.65),
]


def _precompute_base_pressures():
    P = [101325.0]
    for i in range(1, len(_layers)):
        h0, L0, T0 = _layers[i - 1]
        h1, _, _ = _layers[i]
        if abs(L0) > 1e-12:
            Tb = T0 + L0 * (h1 - h0)
            P1 = P[-1] * (Tb / T0) ** (-g0 / (L0 * 1e-3) / R_air)
        else:
            P1 = P[-1] * np.exp(-g0 * ((h1 - h0) * 1000.0) / (R_air * T0))
        P.append(P1)
    return P


_Pb = _precompute_base_pressures()


def geopotential_from_geometric(h_m: float) -> float:
    return (Re * h_m) / (Re + h_m)


def std_atm_1976(h_m: float) -> Tuple[float, float, float, float]:
    h_m = max(0.0, float(h_m))
    if h_m > 86000.0:
        T = 186.0
        P = 0.0
        rho = 0.0
        a = math.sqrt(max(0.0, gamma_air * R_air * T))
        return T, P, rho, a
    h_km = geopotential_from_geometric(h_m) / 1000.0
    idx = len(_layers) - 1
    for i in range(len(_layers) - 1):
        if _layers[i][0] <= h_km < _layers[i + 1][0]:
            idx = i
            break
    h0, L, T0 = _layers[idx]
    Pb = _Pb[idx]
    dh = h_km - h0
    if abs(L) > 1e-12:
        T = T0 + L * dh
        P = Pb * (T / T0) ** (-g0 / (L * 1e-3) / R_air)
    else:
        T = T0
        P = Pb * np.exp(-g0 * (dh * 1000.0) / (R_air * T0))
    rho = P / (R_air * T)
    a = math.sqrt(max(0.0, gamma_air * R_air * T))
    return T, P, rho, a


def drag_force(v_rel: np.ndarray, rho: float, Cd: float, A: float) -> np.ndarray:
    v = np.linalg.norm(v_rel)
    if v < 1e-9 or rho <= 0.0:
        return np.zeros(3)
    D_mag = 0.5 * rho * v * v * Cd * A
    return -D_mag * (v_rel / v)


def enu_to_eci_from_latlon(lat_deg: float, lon_deg: float):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = Re * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )
    e_up = r / np.linalg.norm(r)
    e_east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    e_north = np.cross(e_up, e_east)
    return r, e_east, e_north, e_up


@dataclass
class KZ1AStage:
    name: str
    mass_total_kg: float
    prop_mass_kg: float
    Isp_s: float
    burn_time_s: float
    thrust_N: float
    diameter_m: float
    start_time_s: float
    end_time_s: float
    sep_time_s: float


@dataclass
class KZ1AConfig:
    preset: str = "kz1a_public"
    mass_liftoff_kg: Optional[float] = None
    payload_mass_kg: float = 260.0
    
    # Aero
    Cd_fairing: float = 0.5
    Cd_post: float = 0.5
    fairing_diam_m: float = 1.4
    post_fairing_diam_m: float = 1.2
    
    # Sim settings
    dt: float = 0.05
    t_end: float = 1200.0
    
    # Launch site (Jiuquan)
    site_lat_deg: float = 40.96
    site_lon_deg: float = 100.28
    
    # --- Target Orbit (500 km SSO / Circular) ---
    target_alt_m: float = 500e3
    target_alt_tol_m: float = 10e3  # For success check
    target_ecc_max: float = 0.005
    target_inclination_deg: float = 97.4  # Not strictly enforced by planar guidance logic
    
    # --- S4 MECO / Guidance Settings ---
    # S4 logic: Single burn until orbit condition is met
    s4_meco_min_burn_s: float = 400.0  # Force burn at least this long
    target_circ_alt_m: float = 500e3
    # MECO tolerances
    target_alt_tolerance_m: float = 5e3   # Stop when |a - Re - target| < this
    target_vr_tolerance_mps: float = 20.0 # And |vr| < this (optional check)
    
    # S4 Guidance Gains
    # Max pitch angle relative to velocity vector (AoA-like) or local horizon
    s4_max_radial_tilt_deg: float = 35.0  # Increased for authority
    s4_gain_alt: float = 0.15             # deg per km error
    s4_gain_vr: float = 0.5               # deg per m/s error
    
    # S1-S3 Guidance
    # Pitch program (open loop)
    pitch_start_s: float = 5.0
    pitch_mid_s: float = 70.0
    pitch_horiz_s: float = 290.0
    pitch_mid_deg: float = 65.0
    pitch_final_deg: float = 20.0 # High loft at S3 burnout
    
    # VR feedback (S1-S3)
    use_fpa_vr_feedback: bool = True
    gamma_feedback_gain: float = 0.0  # Disable gamma FB, rely on pitch table
    vr_feedback_gain: float = 0.0     # Disable VR FB, simple table is robust
    
    # Limits
    pitch_min_deg: float = -30.0
    pitch_max_deg: float = 90.0
    fpa_profile: Optional[List[Tuple[float, float]]] = None


@dataclass
class FaultProfile:
    thrust_drop: float = 0.0  # relative drop (0.2 => -20%)
    tvc_rate_lim_deg_s: Optional[float] = None
    tvc_stick_window: Optional[Tuple[float, float]] = None
    event_delay: Optional[Dict[str, float]] = None
    sensor_bias_body: Optional[np.ndarray] = None
    noise_std: float = 0.0
    seed: int = 42


def _kz1a_stage_baseline():
    # S1
    S1_total = 16600.0
    S1_pm = 0.90 * S1_total
    S1_Isp = 2352.0 / g0
    S1_burn = 65.0
    S1_thrust = (S1_pm / S1_burn) * S1_Isp * g0

    # S2
    S2_total = 8686.0
    S2_pm = 0.90 * S2_total
    S2_Isp = 2810.0 / g0
    S2_burn = 62.0
    S2_thrust = (S2_pm / S2_burn) * S2_Isp * g0

    # S3
    S3_total = 3183.0
    S3_pm = 0.90 * S3_total
    S3_Isp = 2850.0 / g0
    S3_burn = 55.0
    S3_thrust = (S3_pm / S3_burn) * S3_Isp * g0

    # S4 (Long burn, low thrust)
    S4_total = 1300.0
    S4_pm = 1000.0
    S4_Isp = 3050.0 / g0  # ~311 s
    S4_burn_nominal = 765.0 # Max duration
    S4_thrust = (S4_pm / S4_burn_nominal) * S4_Isp * g0

    return [
        ("S1", S1_total, S1_pm, S1_Isp, S1_burn, S1_thrust, 1.4),
        ("S2", S2_total, S2_pm, S2_Isp, S2_burn, S2_thrust, 1.4),
        ("S3", S3_total, S3_pm, S3_Isp, S3_burn, S3_thrust, 1.2),
        ("S4", S4_total, S4_pm, S4_Isp, S4_burn_nominal, S4_thrust, 1.2),
    ]


def build_timeline_kz1a(
    cfg: KZ1AConfig, event_delay: Optional[Dict[str, float]] = None
) -> Tuple[List[KZ1AStage], Dict[str, float]]:
    """Construct unified KZ-1A stage objects and event times."""
    # Typical Event Schedule
    events = {
        "S1_ign": 0.0,
        "S1_sep": 83.0,
        "S2_ign": 85.0,
        "S2_sep": 161.0,
        "Fairing_jettison": 176.1,
        "S3_ign": 192.1,
        "S3_sep": 284.2,
        "S4_ign": 287.2,
        "SC_sep": 1150.0, # Placeholder, will be triggered after MECO
    }
    
    if event_delay:
        for k, dv in event_delay.items():
            if k in events:
                events[k] = float(events[k]) + float(dv)

    base = _kz1a_stage_baseline()
    stages: List[KZ1AStage] = []
    
    for name, m_tot, m_prop, Isp_s, burn, thrust, diam in base:
        if name == "S1":
            t0 = events["S1_ign"]
            sep = events["S1_sep"]
        elif name == "S2":
            t0 = events["S2_ign"]
            sep = events["S2_sep"]
        elif name == "S3":
            t0 = events["S3_ign"]
            sep = events["S3_sep"]
        else: # S4
            t0 = events["S4_ign"]
            sep = events["SC_sep"] 
            # Note: burn is set to max possible (prop limited). 
            # Actual cutoff handled by simulation logic.
            
        stages.append(
            KZ1AStage(
                name=name,
                mass_total_kg=m_tot,
                prop_mass_kg=m_prop,
                Isp_s=Isp_s,
                burn_time_s=burn,
                thrust_N=thrust,
                diameter_m=diam,
                start_time_s=t0,
                end_time_s=t0 + burn,
                sep_time_s=sep,
            )
        )
    return stages, events


def _stage_index_from_time(t: float, stages: List[KZ1AStage]) -> int:
    for i, stg in enumerate(stages):
        if t <= stg.sep_time_s:
            return i
    return len(stages) - 1


def local_level_axes(r: np.ndarray, v: Optional[np.ndarray] = None):
    r_norm = np.linalg.norm(r)
    er = r / max(r_norm, 1e-9)
    
    # East/North definition
    ez = np.array([0.0, 0.0, 1.0])
    e_east = np.cross(ez, er)
    nrm = np.linalg.norm(e_east)
    if nrm < 1e-9:
        e_east = np.array([1.0, 0.0, 0.0]) # Polar singularity
    else:
        e_east = e_east / nrm
    e_north = np.cross(er, e_east)
    
    # If velocity provided, define 'Tangential' frame (Trajectory frame)
    # e_tan is horizontal component of velocity
    if v is not None:
        v_horiz = v - np.dot(v, er) * er
        v_h_norm = np.linalg.norm(v_horiz)
        if v_h_norm > 1.0:
            e_tan = v_horiz / v_h_norm
            # e_cross = np.cross(er, e_tan)
            return er, e_tan, np.cross(er, e_tan)
            
    # Fallback to East/North if v is vertical or small
    return er, e_east, e_north


def pitch_cmd_deg_open_loop(t: float, cfg: KZ1AConfig) -> float:
    """Interpolate from DEFAULT_PITCH_PROFILE."""
    prof = DEFAULT_PITCH_PROFILE
    if t <= prof[0][0]:
        return prof[0][1]
    if t >= prof[-1][0]:
        return prof[-1][1]
    for i in range(len(prof) - 1):
        t0, a0 = prof[i]
        t1, a1 = prof[i + 1]
        if t0 <= t <= t1:
            ratio = (t - t0) / (t1 - t0)
            return a0 + ratio * (a1 - a0)
    return 0.0


def build_dir_from_pitch(er: np.ndarray, e_tan: np.ndarray, pitch_deg: float) -> np.ndarray:
    """Pitch is angle from local horizontal (e_tan) towards local vertical (er)."""
    rad = math.radians(pitch_deg)
    # Pitch 90 = Vertical (er)
    # Pitch 0 = Horizontal (e_tan)
    u = math.sin(rad) * er + math.cos(rad) * e_tan
    return u / max(np.linalg.norm(u), 1e-9)


def s4_guidance_logic(
    r: np.ndarray, 
    v: np.ndarray, 
    cfg: KZ1AConfig, 
    t: float, 
    s4_ign_t: float
) -> Tuple[np.ndarray, float, str]:
    """
    Closed-loop guidance for S4 to circularize at target altitude.
    Uses Cascaded Control: Altitude -> Target Vr -> Pitch.
    User Feedback Iteration: Precision Landing (Tuned S3 + Strict Vr).
    """
    # Current state
    r_norm = np.linalg.norm(r)
    h = r_norm - Re
    v_norm = np.linalg.norm(v)
    er = r / max(r_norm, 1e-9)
    vr = np.dot(v, er)
    
    # Flight Path Angle (Gamma)
    sin_gamma = np.clip(vr / max(v_norm, 1e-9), -1.0, 1.0)
    gamma_deg = math.degrees(math.asin(sin_gamma))
    
    # Reference Frame
    _, e_tan, _ = local_level_axes(r, v)
    
    # --- Guidance Strategy: Cascaded Control ---
    
    orb = estimate_orbit_from_state(r, v)
    sma_err = (Re + cfg.target_circ_alt_m) - orb["a_m"]
    
    # Outer Loop: Altitude Error -> Target Radial Velocity (Vr_cmd)
    h_err = cfg.target_alt_m - h # Positive if low
    
    # Terminal Phase Logic (Force Flat Insertion)
    # Trigger if we are close to target altitude OR close to target energy
    
    close_to_alt = abs(h_err) < 5000.0 # Within 5km
    close_to_energy = sma_err < 30e3   # Within 30km SMA
    
    if close_to_alt or close_to_energy: 
        # TERMINAL LOCK: Force Vr to 0
        vr_cmd = 0.0
        k_vr = 4.0 # Extreme gain to kill Vr instantly
        status_prefix = "S4_LOCK"
    else:
        # Normal Phase (Climb)
        k_alt = 2.200 # Slightly increased (was 2.000)
        vr_cmd = k_alt * (h_err / 1000.0)
        vr_cmd = max(-50.0, min(300.0, vr_cmd)) 
        k_vr = 0.600 
        status_prefix = "S4_CL"
        
    # Inner Loop: Vr Error -> Pitch Command
    vr_err = vr_cmd - vr # Positive if we need more climb rate
    pitch_cmd = k_vr * vr_err
    
    # Clamp Pitch
    max_pitch = 45.0
    min_pitch = -20.0 
    
    # Force Logic: Don't pitch down if low (unless in LOCK mode)
    if h < 480e3 and status_prefix != "S4_LOCK":
        min_pitch = 0.0
        
    pitch_cmd = max(min_pitch, min(max_pitch, pitch_cmd))
    
    # Compute direction vector
    u_cmd = build_dir_from_pitch(er, e_tan, pitch_cmd)
    
    status = f"{status_prefix}: h={h/1000:.1f}km, vr_cmd={vr_cmd:.1f}, vr={vr:.1f}, P={pitch_cmd:.1f}"
    
    return u_cmd, pitch_cmd, status


def orbital_metrics(r: np.ndarray, v: np.ndarray, mu_val: float):
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    er = r / max(r_norm, 1e-9)
    vr = np.dot(v, er)
    h_vec = np.cross(r, v)
    h_norm = max(np.linalg.norm(h_vec), 1e-9)
    e_vec = (np.cross(v, h_vec) / mu_val) - er
    e = np.linalg.norm(e_vec)
    a = 1.0 / (2.0 / r_norm - v_norm * v_norm / mu_val)
    rp = a * (1 - e)
    ra = a * (1 + e)
    return {"r_norm": r_norm, "v_norm": v_norm, "vr": vr, "a": a, "e": e, "rp": rp, "ra": ra}


def estimate_orbit_from_state(r: np.ndarray, v: np.ndarray, mu_val: float = mu) -> Dict[str, float]:
    metrics = orbital_metrics(r, v, mu_val)
    return {
        "a_m": metrics["a"],
        "e": metrics["e"],
        "rp_m": metrics["rp"],
        "ra_m": metrics["ra"],
        "hp_m": metrics["rp"] - Re,
        "ha_m": metrics["ra"] - Re,
        "r_norm": metrics["r_norm"],
        "v_norm": metrics["v_norm"],
        "vr": metrics["vr"],
    }


def _apply_tvc(u_des: np.ndarray, t: float, dt: float, prev: Optional[np.ndarray], fault: Optional[FaultProfile]):
    """Apply TVC rate limits and fault logic."""
    u_des = u_des / max(np.linalg.norm(u_des), 1e-9)
    
    # Fault: Stuck TVC
    if fault and fault.tvc_stick_window:
        t0, dur = fault.tvc_stick_window
        if t0 is not None and dur is not None and (t0 <= t <= t0 + dur) and prev is not None:
            return prev

    if prev is None:
        return u_des
        
    # Rate limit (Fault-based or nominal)
    rate_limit = 10.0 # deg/s nominal
    if fault and fault.tvc_rate_lim_deg_s is not None:
        rate_limit = fault.tvc_rate_lim_deg_s

    dot = float(np.clip(np.dot(prev, u_des), -1.0, 1.0))
    ang = math.acos(dot)
    max_ang = math.radians(rate_limit) * dt
    
    if ang <= max_ang + 1e-9:
        return u_des
        
    # Rotate prev towards u_des by max_ang
    axis = np.cross(prev, u_des)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-9:
        return prev # Parallel
    axis = axis / nrm
    c = math.cos(max_ang)
    s = math.sin(max_ang)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K
    u_new = R @ prev
    return u_new / max(np.linalg.norm(u_new), 1e-9)


def kz1a_ascent_rhs(
    t: float,
    state: np.ndarray,
    thrust_vec: np.ndarray,
    mdot: float,
    Cd: float,
    A: float,
    use_atmosphere: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    r = state[0:3]
    v = state[3:6]
    m = max(state[6], 1.0)
    rnorm = np.linalg.norm(r)
    h = rnorm - Re
    
    # Atmosphere
    rho = 0.0
    v_rel = v
    F_D = np.zeros(3)
    if use_atmosphere:
        _, _, rho, _ = std_atm_1976(max(0.0, h))
        v_atm = np.cross(np.array([0.0, 0.0, omega_earth]), r)
        v_rel = v - v_atm
        F_D = drag_force(v_rel, rho, Cd, A)
        
    # Forces
    a_grav = -mu * r / (rnorm ** 3)
    a_thrust = thrust_vec / m
    a_drag = F_D / m
    acc = a_thrust + a_drag + a_grav
    
    # Aux
    q_dyn = 0.5 * rho * np.dot(v_rel, v_rel)
    n_load = np.linalg.norm(acc + (-a_grav)) / g0 # G-load sensed (Thrust + Drag)
    
    # Projection
    er = r / rnorm
    a_r_total = float(np.dot(acc, er))
    
    deriv = np.array([v[0], v[1], v[2], acc[0], acc[1], acc[2], -mdot])
    aux = {
        "q_dyn": q_dyn,
        "n_load": n_load,
        "h": h,
        "rho": rho,
        "a_r_total": a_r_total,
    }
    return deriv, aux


def simulate_kz1a_eci(
    cfg: KZ1AConfig,
    fault: Optional[FaultProfile] = None,
    use_atmosphere: bool = True,
    force_fpa_control: bool = False,
    log_pitch: bool = True,
):
    fault = fault or FaultProfile()
    stages, events = build_timeline_kz1a(cfg, event_delay=fault.event_delay)
    
    A_fair = math.pi * (cfg.fairing_diam_m * 0.5) ** 2
    A_post = math.pi * (cfg.post_fairing_diam_m * 0.5) ** 2

    # Mass setup
    mass_budget = sum(st.mass_total_kg for st in stages) + cfg.payload_mass_kg
    m0 = cfg.mass_liftoff_kg if cfg.mass_liftoff_kg is not None else mass_budget
    m0 = max(m0, mass_budget)

    # Initial State (Launch Site)
    r0, _, _, _ = enu_to_eci_from_latlon(cfg.site_lat_deg, cfg.site_lon_deg)
    v0 = np.cross(np.array([0.0, 0.0, omega_earth]), r0)
    state = np.zeros(7)
    state[0:3] = r0
    state[3:6] = v0
    state[6] = m0

    dt = cfg.dt
    t = 0.0
    t_end = cfg.t_end

    prop_left = [st.prop_mass_kg for st in stages]
    dry_left = [st.mass_total_kg - st.prop_mass_kg for st in stages]
    
    fairing_off = False
    s4_burn_on = False
    s4_ign_time = None
    
    tvc_prev = None
    rng = np.random.default_rng(fault.seed)

    # Logging
    hist = {
        "t": [], "h": [], "speed": [], "q_dyn": [], "n_load": [], "mass": [], 
        "thrust": [], "stage": [], "r_eci": [], "v_eci": [], 
        "pitch_cmd_deg": [], "fpa_deg": [], "vr": [], "vt": [], "a_r_total": [],
        "a_true": [], "a_meas": []
    }
    
    events["S4_cutoff"] = 99999.0 # Placeholder
    
    while t <= t_end:
        # 1. Event Triggers
        if (not fairing_off) and t >= events["Fairing_jettison"]:
            fairing_off = True
            
        # Staging mass drop
        stage_idx = _stage_index_from_time(t, stages)
        curr_stg = stages[stage_idx]
        
        # Drop dry mass if separation time passed
        for i, stg in enumerate(stages):
            if dry_left[i] > 0.0 and t >= stg.sep_time_s:
                state[6] = max(1.0, state[6] - dry_left[i])
                dry_left[i] = 0.0
                
        # 2. S4 Logic (Ignition & MECO)
        if curr_stg.name == "S4":
            # Ignition
            if (not s4_burn_on) and t >= events["S4_ign"] and prop_left[stage_idx] > 0:
                s4_burn_on = True
                s4_ign_time = t
            
            # MECO Logic (Only check if burning)
            if s4_burn_on:
                burn_duration = t - s4_ign_time
                orb = estimate_orbit_from_state(state[0:3], state[3:6])
                sma_err = abs((orb["a_m"] - Re) - cfg.target_circ_alt_m)
                
                # Conditions
                cond_time = burn_duration >= cfg.s4_meco_min_burn_s
                
                # Robust MECO: Check Semi-Major Axis (Energy) ONLY
                # We want a ~= target_alt + Re
                
                sma_target = Re + cfg.target_circ_alt_m
                sma_current = orb["a_m"]
                
                # Stop if we have enough energy (SMA >= Target)
                # We trust the guidance to have circularized by now.
                # If we wait for e < 0.005, we might overshoot SMA significantly if guidance is imperfect.
                
                have_energy = sma_current >= sma_target
                
                # Safety cutoff if we overshoot altitude too much
                overshoot = sma_current > (sma_target + 5e3) 
                
                # Note: We relax the min burn time constraint if we are clearly overshooting
                if (cond_time and have_energy) or overshoot:
                    s4_burn_on = False
                    events["S4_cutoff"] = t
                    # Update stage end time so 'burning' flag works
                    curr_stg.end_time_s = t
        
        # 3. Thrust & MDOT
        burning = False
        thrust_mag = 0.0
        mdot = 0.0
        
        if curr_stg.name == "S4":
            if s4_burn_on and prop_left[stage_idx] > 0:
                burning = True
        else:
            if t >= curr_stg.start_time_s and t < curr_stg.end_time_s and prop_left[stage_idx] > 0:
                burning = True
        
        if burning:
            thrust_scale = 1.0 - fault.thrust_drop
            thrust_mag = curr_stg.thrust_N * thrust_scale
            mdot = curr_stg.prop_mass_kg / curr_stg.burn_time_s
            # Clip if near empty
            dt_eff = max(dt, 1e-6)
            if prop_left[stage_idx] < mdot * dt_eff:
                mdot = prop_left[stage_idx] / dt_eff
                
        # 4. Guidance (Direction)
        pitch_cmd_deg = 0.0
        u_des = np.array([1.0, 0.0, 0.0])
        
        if curr_stg.name == "S4":
            # Closed Loop S4
            ref_t = s4_ign_time if s4_ign_time else events["S4_ign"]
            u_des, pitch_cmd_deg, _ = s4_guidance_logic(state[0:3], state[3:6], cfg, t, ref_t)
        else:
            # Open Loop S1-S3
            pitch_cmd_deg = pitch_cmd_deg_open_loop(t, cfg)
            er, e_east, e_north = local_level_axes(state[0:3], state[3:6])
            # For S1-S3, pitch is usually in the orbital plane (e_east approx)
            # We assume launch azimuth is roughly handled by initial velocity/setup
            # Simplified: Pitch in the (er, e_tan) plane
            _, e_tan, _ = local_level_axes(state[0:3], state[3:6])
            u_des = build_dir_from_pitch(er, e_tan, pitch_cmd_deg)

        # 5. Dynamics Step
        u_cmd = _apply_tvc(u_des, t, dt, tvc_prev, fault)
        tvc_prev = u_cmd
        thrust_vec = thrust_mag * u_cmd
        
        Cd = cfg.Cd_post if fairing_off else cfg.Cd_fairing
        A = A_post if fairing_off else A_fair
        
        # RK4 Integration
        k1, aux = kz1a_ascent_rhs(t, state, thrust_vec, mdot, Cd, A, use_atmosphere)
        k2, _ = kz1a_ascent_rhs(t, state + 0.5*dt*k1, thrust_vec, mdot, Cd, A, use_atmosphere)
        k3, _ = kz1a_ascent_rhs(t, state + 0.5*dt*k2, thrust_vec, mdot, Cd, A, use_atmosphere)
        k4, _ = kz1a_ascent_rhs(t, state + dt*k3, thrust_vec, mdot, Cd, A, use_atmosphere)
        
        state_next = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        # 6. Logging & Updates
        if burning:
            prop_left[stage_idx] = max(0.0, prop_left[stage_idx] - mdot * dt)
            if prop_left[stage_idx] <= 1e-9 and s4_burn_on:
                s4_burn_on = False # Fuel exhaustion
                events["S4_cutoff"] = t

        # Compute derived metrics for log
        r_now = state[0:3]
        v_now = state[3:6]
        r_norm = np.linalg.norm(r_now)
        er = r_now / max(r_norm, 1e-9)
        vr_val = np.dot(v_now, er)
        v_tan_vec = v_now - vr_val * er
        vt_val = np.linalg.norm(v_tan_vec)
        fpa = math.degrees(math.atan2(vr_val, vt_val))
        
        hist["t"].append(t)
        hist["h"].append(aux["h"])
        hist["speed"].append(np.linalg.norm(v_now))
        hist["q_dyn"].append(aux["q_dyn"])
        hist["n_load"].append(aux["n_load"])
        hist["mass"].append(state[6])
        hist["thrust"].append(thrust_mag)
        hist["stage"].append(stage_idx + 1)
        hist["r_eci"].append(r_now.copy())
        hist["v_eci"].append(v_now.copy())
        hist["pitch_cmd_deg"].append(pitch_cmd_deg)
        hist["fpa_deg"].append(fpa)
        hist["vr"].append(vr_val)
        hist["vt"].append(vt_val)
        hist["a_r_total"].append(aux["a_r_total"])

        # --- Acceleration Calculation (True & Meas) ---
        # 1. True Kinematic Acceleration (ECI)
        # kz1a_ascent_rhs returns deriv = [v, acc, -mdot]
        # We need to re-evaluate forces to get exact components or just use the deriv from RK4?
        # RK4 averages slopes. For logging, we usually just evaluate at current state.
        # Re-evaluate forces at current state for consistent logging
        Cd_log = cfg.Cd_post if fairing_off else cfg.Cd_fairing
        A_log = A_post if fairing_off else A_fair
        
        # Recalculate forces for logging (consistent with current state)
        # Atmosphere
        rho_log = 0.0
        v_rel_log = v_now
        F_D_log = np.zeros(3)
        if use_atmosphere:
            _, _, rho_log, _ = std_atm_1976(max(0.0, aux["h"]))
            v_atm_log = np.cross(np.array([0.0, 0.0, omega_earth]), r_now)
            v_rel_log = v_now - v_atm_log
            F_D_log = drag_force(v_rel_log, rho_log, Cd_log, A_log)
            
        a_grav_log = -mu * r_now / (r_norm ** 3)
        a_thrust_log = thrust_vec / max(state[6], 1.0)
        a_drag_log = F_D_log / max(state[6], 1.0)
        a_true_eci = a_thrust_log + a_drag_log + a_grav_log
        
        # 2. Measured Acceleration (Specific Force) in Body Frame
        # a_spec_eci = a_thrust + a_drag
        a_spec_eci = a_thrust_log + a_drag_log
        
        # Construct Body Frame (X=Thrust, Y=Right, Z=Down/Belly)
        # x_b = u_cmd (Thrust Direction)
        x_b = u_cmd
        
        # y_b perpendicular to x_b and local vertical (er) -> "Wings Level"
        # If x_b is vertical, use East
        y_b = np.cross(x_b, er)
        if np.linalg.norm(y_b) < 1e-2:
            # Vertical flight, use East as Y
            _, e_east_log, _ = local_level_axes(r_now)
            y_b = e_east_log
            
        y_b = y_b / max(np.linalg.norm(y_b), 1e-9)
        z_b = np.cross(x_b, y_b)
        
        # Project ECI specific force to Body
        a_meas_body = np.array([
            np.dot(a_spec_eci, x_b),
            np.dot(a_spec_eci, y_b),
            np.dot(a_spec_eci, z_b)
        ])
        
        # Add Sensor Bias
        
        # Add Sensor Bias
        if fault.sensor_bias_body is not None:
            a_meas_body += fault.sensor_bias_body
            
        # Add Noise
        if fault.noise_std > 0.0:
            noise = rng.normal(0.0, fault.noise_std, 3)
            a_meas_body += noise
            
        hist["a_true"].append(a_true_eci)
        hist["a_meas"].append(a_meas_body)

        t += dt
        state = state_next
        
        if aux["h"] < -100.0 and t > 10.0:
            print(f"CRASH: Altitude {aux['h']} < 0 at t={t}")
            break

    # --- Post-process & Summary ---
    # Convert lists to arrays
    for k in hist:
        hist[k] = np.array(hist[k])
        
    events_sample = {}
    for k, te in events.items():
        # Find closest index
        idx = (np.abs(hist["t"] - te)).argmin() if len(hist["t"]) > 0 else 0
        if 0 <= idx < len(hist["t"]):
             events_sample[k] = {
                "t_s": float(hist["t"][idx]),
                "h_m": float(hist["h"][idx]),
                "v_mps": float(hist["speed"][idx]),
            }

    # Orbital parameters at end
    final_orb = estimate_orbit_from_state(state[0:3], state[3:6])
    
    summary = {
        "preset": cfg.preset,
        "final_state": {
            "t_s": float(t),
            "h_m": float(hist["h"][-1]), 
            "v_mps": float(hist["speed"][-1]),
            "orbit_a_km": float(final_orb["a_m"] / 1000.0),
            "orbit_e": float(final_orb["e"]),
            "orbit_hp_km": float(final_orb["hp_m"] / 1000.0),
            "orbit_ha_km": float(final_orb["ha_m"] / 1000.0),
        },
        "events": events_sample,
        "max_stats": {
            "q_peak_Pa": float(np.max(hist["q_dyn"])),
            "n_max_g": float(np.max(hist["n_load"])),
        }
    }

    # Return structure matching old format for compatibility
    return {
        "t": hist["t"],
        "h": hist["h"],
        "speed": hist["speed"],
        "q_dyn": hist["q_dyn"],
        "n_load": hist["n_load"],
        "mass": hist["mass"],
        "thrust": hist["thrust"],
        "a_true": hist["a_true"],
        "a_meas": hist["a_meas"],
        "stage": hist["stage"],
        "r_eci": hist["r_eci"],
        "v_eci": hist["v_eci"],
        "pitch_cmd_deg": hist["pitch_cmd_deg"],
        "fpa_deg": hist["fpa_deg"],
        "vr": hist["vr"],
        "vt": hist["vt"],
        "a_r_total": hist["a_r_total"],
        "events": events_sample,
        "summary": summary,
        # Placeholders for fields not strictly needed but expected by plotter
        "pitch_actual_deg": hist["pitch_cmd_deg"], # Assume perfect control for nominal
        "gamma_cmd_deg": hist["pitch_cmd_deg"],
        "e_gamma_deg": np.zeros_like(hist["t"]),
        "radial_component": np.zeros_like(hist["t"]),
        "tilt_cmd_deg": np.zeros_like(hist["t"]),
        "s4_phase": np.zeros_like(hist["t"]),
        "alt_ok": np.zeros_like(hist["t"]),
        "vr_ok": np.zeros_like(hist["t"]),
        "e_ok": np.zeros_like(hist["t"]),
        "dv_ok": np.zeros_like(hist["t"]),
        "all_ok": np.zeros_like(hist["t"]),
        "vr_target": np.zeros_like(hist["t"]),
        "e_vr": np.zeros_like(hist["t"]),
        "base_pitch_open_loop_deg": np.zeros_like(hist["t"]),
        "delta_pitch_gamma_deg": np.zeros_like(hist["t"]),
        "delta_pitch_vr_deg": np.zeros_like(hist["t"]),
        "a_r_thrust": np.zeros_like(hist["t"]),
        "a_r_drag": np.zeros_like(hist["t"]),
        "a_r_grav": np.zeros_like(hist["t"]),
    }
