# ch4codexv1.1/src/sim/run_nominal.py

import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# Ensure src package importable when executed from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.sim.eci_verified import simulate_kz1a_ascent

# Align with core dynamics (Ch2) to avoid altitude offsets
R_EARTH = 6371000.0

@dataclass
class NominalResult:
    time: np.ndarray
    altitude_km: np.ndarray
    speed_kms: np.ndarray
    flight_path_deg: np.ndarray
    dynamic_pressure_kpa: np.ndarray
    normal_load_g: np.ndarray
    thrust_kN: np.ndarray
    cone_margin_deg: np.ndarray
    states: np.ndarray  # (N, 7) [rx, ry, rz, vx, vy, vz, m]

def simulate_nominal(
    duration_s: float = 4000.0, # Extended to allow S4 Hohmann transfer orbit circularization
    dt: float = 0.01, # Passed to simulate_kz1a_ascent
    save_csv: bool = True,
    guidance_fn: None = None, # Ignored
    use_fpa_vr_feedback: bool = True,
    gamma_feedback_gain: float = 0.5,
    vr_feedback_gain: float = 0.5,
) -> NominalResult:
    
    print("Running Nominal Simulation using Verified Kernel (3-DoF)...")
    # Call the verified wrapper
    res = simulate_kz1a_ascent(
        t_end=duration_s,
        dt=dt,
        preset="nasaspaceflight",
        use_fpa_vr_feedback=use_fpa_vr_feedback,
        gamma_feedback_gain=gamma_feedback_gain,
        vr_feedback_gain=vr_feedback_gain,
    )
    
    # Extract data
    t = res["t"]
    r = res["r_eci"] # (N, 3)
    v = res["v_eci"] # (N, 3)
    mass = res["mass"] # (N,)
    
    # Derived metrics
    r_norm = np.linalg.norm(r, axis=1)
    altitude = r_norm - R_EARTH
    speed = np.linalg.norm(v, axis=1)
    
    # Flight Path Angle
    r_hat = np.divide(r, r_norm[:, None], out=np.zeros_like(r), where=r_norm[:, None] > 1.0)
    v_radial = np.sum(v * r_hat, axis=1)
    v_horizontal = np.sqrt(np.clip(speed**2 - v_radial**2, 0.0, None))
    gamma = np.degrees(np.arctan2(v_radial, v_horizontal + 1e-6))
    
    # Dynamics Pressure & Load
    q_dyn = res["q_dyn"] / 1000.0 # Pa -> kPa
    n_load = res["n_load"]
    thrust = res["thrust"] / 1000.0 # N -> kN
    
    # Cone Margin (Placeholder)
    cone_margin = res.get("cone_angle_deg", np.zeros_like(t))
    
    # Construct States [r, v, m] (7-dim)
    states = np.zeros((len(t), 7))
    states[:, 0:3] = r
    states[:, 3:6] = v
    states[:, 6] = mass
    
    result = NominalResult(
        time=t,
        altitude_km=altitude / 1000.0,
        speed_kms=speed / 1000.0,
        flight_path_deg=gamma,
        dynamic_pressure_kpa=q_dyn,
        normal_load_g=n_load,
        thrust_kN=thrust,
        cone_margin_deg=cone_margin,
        states=states,
    )
    
    if save_csv:
        df = pd.DataFrame(
            {
                "time_s": result.time,
                "altitude_km": result.altitude_km,
                "speed_kms": result.speed_kms,
                "flight_path_deg": result.flight_path_deg,
                "dynamic_pressure_kpa": result.dynamic_pressure_kpa,
                "normal_load_g": result.normal_load_g,
                "thrust_kN": result.thrust_kN,
                "cone_margin_deg": result.cone_margin_deg,
                # Add state columns for debugging/reference
                "rx": states[:, 0], "ry": states[:, 1], "rz": states[:, 2],
                "vx": states[:, 3], "vy": states[:, 4], "vz": states[:, 5],
                "mass": states[:, 6]
            }
        )
        outdir = Path("outputs") / "nominal"
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "nominal_traj.csv", index=False)
        print(f"[Nominal] Saved trajectory to {outdir / 'nominal_traj.csv'}")

    # Print key metrics
    s4_ign_t = res["events"].get("S4_ign", {}).get("t_s", None)
    if s4_ign_t is not None:
        idx = int(np.searchsorted(t, s4_ign_t))
        vr_idx = float(np.dot(v[idx], r_hat[idx]))
        print(f"[Nominal] S4 ignition t={s4_ign_t:.2f}s, altitude={altitude[idx]:.1f} km, gamma={gamma[idx]:.2f} deg, vr={vr_idx:.1f} m/s")
    # final metrics (ecc, dv_circ)
    r_last = r[-1]; v_last = v[-1]
    r_norm = np.linalg.norm(r_last); v_norm = np.linalg.norm(v_last)
    er = r_last / max(r_norm, 1e-9)
    vr_last = float(np.dot(v_last, er))
    vcirc = np.sqrt(3.986004418e14 / r_norm)
    dv_circ = abs(v_norm - vcirc)
    # simple eccentricity via energy/angular momentum
    h_vec = np.cross(r_last, v_last); h_norm = np.linalg.norm(h_vec)
    e_vec = (np.cross(v_last, h_vec) / 3.986004418e14) - er
    ecc = np.linalg.norm(e_vec)
    print(f"[Nominal] Final alt={altitude[-1]:.1f} km, vr={vr_last:.1f} m/s, ecc={ecc:.4f}, dv_circ={dv_circ:.1f} m/s")
    
    return result

if __name__ == "__main__":
    simulate_nominal()

# Alias for compatibility
simulate_full_mission = simulate_nominal

# Backward-compatible alias expected by some table scripts
def simulate(duration_s: float = 120.0, dt: float = 1.0):
    return simulate_nominal(duration_s=duration_s, dt=dt, save_csv=False)
