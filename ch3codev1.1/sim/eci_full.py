# -*- coding: utf-8 -*-
"""
Chapter 3 faulted ECI wrapper on top of the unified KZ-1A core.
Keeps the original simulate_ecifull interface while delegating dynamics to kz1a_eci_core.
"""
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Make core available (ch4codexv1.1 is the package root for src/)
REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT / "ch4codexv1.1"
if str(CORE_ROOT) not in sys.path:
    sys.path.append(str(CORE_ROOT))

from src.sim.kz1a_eci_core import (  # noqa: E402
    FaultProfile,
    KZ1AConfig,
    simulate_kz1a_eci,
)


def _sensor_bias_vector(sensor_bias) -> np.ndarray:
    if sensor_bias is None:
        return np.zeros(3)
    arr = np.asarray(sensor_bias, dtype=float).ravel()
    if arr.size == 3:
        return arr
    if arr.size == 1:
        return np.array([arr[0], 0.0, 0.0])
    out = np.zeros(3)
    out[: min(3, arr.size)] = arr[: min(3, arr.size)]
    return out


def simulate_ecifull(
    dt: float = 0.05,
    t_end: float = 1100.0,
    preset: str = "nasaspaceflight",
    mass_liftoff: float = 30000.0,
    payload: float = 260.0,
    Cd_fair: float = 0.5,
    Cd_post: float = 0.5,
    fair_diam: float = 1.4,
    post_diam: float = 1.2,
    site_lat: float = 40.96,
    site_lon: float = 100.28,
    pitch_start_s: float = 5.0,
    pitch_mid_s: float = 70.0,
    pitch_horiz_s: float = 170.0,
    pitch_mid_deg: float = 30.0,
    pitch_final_deg: float = 0.0,
    thrust_drop: float = 0.0,
    tvc_rate_lim_deg_s: float = None,
    tvc_stick: Tuple[float, float] = (None, None),
    event_delay: Dict[str, float] = None,
    sensor_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    noise_std: float = 0.3,
    seed: int = 0,
):
    """
    Thin wrapper returning t, r, v, a_true, a_meas, events, mass, stage.
    """
    cfg = KZ1AConfig(
        preset=preset,
        dt=dt,
        t_end=t_end,
        mass_liftoff_kg=mass_liftoff,
        payload_mass_kg=payload,
        Cd_fairing=Cd_fair,
        Cd_post=Cd_post,
        fairing_diam_m=fair_diam,
        post_fairing_diam_m=post_diam,
        site_lat_deg=site_lat,
        site_lon_deg=site_lon,
        pitch_start_s=pitch_start_s,
        pitch_mid_s=pitch_mid_s,
        pitch_horiz_s=pitch_horiz_s,
        pitch_mid_deg=pitch_mid_deg,
        pitch_final_deg=pitch_final_deg,
    )
    fp = FaultProfile(
        thrust_drop=thrust_drop,
        tvc_rate_lim_deg_s=tvc_rate_lim_deg_s,
        tvc_stick_window=tvc_stick if tvc_stick != (None, None) else None,
        event_delay=event_delay,
        sensor_bias_body=_sensor_bias_vector(sensor_bias),
        noise_std=noise_std,
        seed=seed,
    )

    res = simulate_kz1a_eci(cfg, fault=fp)

    return {
        "t": res["t"],
        "r": res["r_eci"],
        "v": res["v_eci"],
        "a_true": res["a_true"],
        "a_meas": res["a_meas"],
        "mass": res["mass"],
        "thrust": res["thrust"],
        "q_dyn": res["q_dyn"],
        "n_load": res["n_load"],
        "pitch_cmd_deg": res["pitch_cmd_deg"],
        "fpa_deg": res["fpa_deg"],
        "stage": res["stage"],
        "events": res["events"],
    }
