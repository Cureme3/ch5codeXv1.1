#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Only regenerate fault openloop trajectories (not nominal or replan)."""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from src.sim.kz1a_eci_core import KZ1AConfig, simulate_kz1a_eci, FaultProfile, Re
from src.sim.scenarios import get_scenario, scale_scenario_by_eta

FAULT_ID_MAP = {
    "F1": "F1_thrust_deg15",
    "F2": "F2_tvc_rate4",
    "F3": "F3_tvc_stuck3deg",
    "F4": "F4_sensor_bias2deg",
    "F5": "F5_event_delay5s",
}

ETA_VALUES = [0.2, 0.5, 0.8]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data" / "ch4_trajectories_replan"


def compute_downrange_from_pos(pos, pos0):
    """Compute downrange distance in km."""
    r0 = pos0 / np.linalg.norm(pos0)
    downrange = []
    for p in pos:
        r_norm = np.linalg.norm(p)
        r_unit = p / max(r_norm, 1e-9)
        cos_angle = np.clip(np.dot(r0, r_unit), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        downrange.append(angle * Re / 1000.0)
    return np.array(downrange)


def compute_altitude_from_pos(pos):
    """Compute altitude in km."""
    return np.array([np.linalg.norm(p) - Re for p in pos]) / 1000.0


def generate_openloop(fault_key, eta):
    """Generate single openloop trajectory."""
    scenario_id = FAULT_ID_MAP[fault_key]
    scenario_base = get_scenario(scenario_id)
    scenario = scale_scenario_by_eta(scenario_base, eta)

    fault_profile = FaultProfile()
    fault_profile.t_fault_s = scenario.t_fault_s
    fault_type = scenario.fault_type
    params = scenario.params

    if fault_type == "thrust_degradation":
        fault_profile.thrust_drop = params.get("degrade_frac", 0.0)
    elif fault_type == "tvc_rate_limit":
        fault_profile.tvc_rate_lim_deg_s = params.get("tvc_rate_deg_s", 10.0)
    elif fault_type == "tvc_stuck":
        stuck_dur = params.get("stuck_duration_s", 100.0)
        fault_profile.tvc_stick_window = (scenario.t_fault_s, stuck_dur)
        fault_profile.tvc_stuck_angle_deg = params.get("stuck_angle_deg", 0.0)
    elif fault_type == "sensor_bias":
        bias_deg = params.get("sensor_bias_deg", 0.0)
        fault_profile.sensor_bias_body = np.array([0.0, np.radians(bias_deg), 0.0])
    elif fault_type == "event_delay":
        delay_s = params.get("event_delay_s", 0.0)
        fault_profile.event_delay = {"S4_ign": delay_s}

    cfg = KZ1AConfig(
        preset="nasaspaceflight",
        dt=0.01,
        t_end=4000.0,
        target_circ_alt_m=500e3,
    )
    sim_data = simulate_kz1a_eci(cfg, fault=fault_profile)

    if sim_data is None:
        return fault_key, eta, False, "sim returned None"

    time = sim_data["t"]
    pos = sim_data["r_eci"]
    vel = sim_data["v_eci"]

    downrange = compute_downrange_from_pos(pos, pos[0])
    altitude = compute_altitude_from_pos(pos)

    # Save
    eta_str = f"{int(eta*10):02d}"
    fname = f"{fault_key}_eta{eta_str}_openloop.npz"
    fpath = OUTPUT_DIR / fname

    np.savez_compressed(
        fpath,
        time=time,
        downrange=downrange,
        altitude=altitude,
        r_eci=pos,
        v_eci=vel,
        eta=eta,
        fault_type=scenario.fault_type,
        t_fault=scenario.t_fault_s,
        t_confirm=scenario.t_confirm_s,
    )

    final_h = altitude[-1] if len(altitude) > 0 else -999
    crashed = final_h < 10
    return fault_key, eta, True, f"h_final={final_h:.1f}km, crashed={crashed}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for fault_key in ["F2", "F3", "F4", "F5"]:  # Skip F1 (already correct)
        for eta in [0.2, 0.5, 0.8]:  # All severities
            tasks.append((fault_key, eta))
    
    print(f"Regenerating {len(tasks)} openloop trajectories...")
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(generate_openloop, fk, e): (fk, e) for fk, e in tasks}
        for future in as_completed(futures):
            fk, e = futures[future]
            try:
                fault_key, eta, success, msg = future.result()
                print(f"  {fault_key} eta={eta}: {msg}")
            except Exception as ex:
                print(f"  {fk} eta={e}: ERROR - {ex}")
    
    print("Done!")


if __name__ == "__main__":
    main()
