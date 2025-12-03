# -*- coding: utf-8 -*-
"""
Chapter 4 verified wrapper that delegates to the unified KZ-1A ECI core.
"""
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure src package is importable when running from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.sim.kz1a_eci_core import (  # noqa: E402
    FaultProfile,
    KZ1AConfig,
    drag_force,
    g0,
    mu,
    omega_earth,
    Re,
    simulate_kz1a_eci,
    std_atm_1976,
)

# Backward-compatible alias
Config = KZ1AConfig


def simulate(cfg: KZ1AConfig):
    return simulate_kz1a_eci(cfg, fault=None)


def simulate_kz1a_ascent(
    t_end: float = 1100.0,
    dt: float = 0.5,
    preset: str = "nasaspaceflight",
    use_fpa_vr_feedback: bool = True,
    gamma_feedback_gain: float = 0.5,
    vr_feedback_gain: float = 0.5,
):
    """
    Nominal ascent wrapper used by Chapter 4 scripts.
    Returns the core result plus convenience fields altitude_km/speed_kms/gamma_deg.
    """
    cfg = KZ1AConfig(
        dt=dt,
        t_end=t_end,
        preset=preset,
        use_fpa_vr_feedback=use_fpa_vr_feedback,
        gamma_feedback_gain=gamma_feedback_gain,
        vr_feedback_gain=vr_feedback_gain,
    )
    res = simulate_kz1a_eci(cfg, fault=None)
    res_out = dict(res)
    r = res["r_eci"]
    v = res["v_eci"]
    r_norm = np.linalg.norm(r, axis=1)
    altitude_km = (r_norm - Re) / 1000.0
    speed_kms = np.linalg.norm(v, axis=1) / 1000.0
    r_hat = np.divide(r, r_norm[:, None], out=np.zeros_like(r), where=r_norm[:, None] > 1.0)
    v_radial = np.sum(v * r_hat, axis=1)
    v_horizontal = np.sqrt(np.clip(np.linalg.norm(v, axis=1) ** 2 - v_radial ** 2, 0.0, None))
    gamma_deg = np.degrees(np.arctan2(v_radial, v_horizontal + 1e-6))

    res_out["altitude_km"] = altitude_km
    res_out["speed_kms"] = speed_kms
    res_out["gamma_deg"] = gamma_deg
    res_out.setdefault("cone_angle_deg", np.zeros_like(gamma_deg))
    return res_out


def run_eci(
    out_npz: Optional[str] = None,
    out_json: Optional[str] = None,
    preset: str = "nasaspaceflight",
    fault: Optional[FaultProfile] = None,
    controls_u=None,
    controls_alpha_deg=None,
):
    """
    Compatibility wrapper that runs the unified core and optionally writes NPZ/JSON.
    controls_* are kept for signature compatibility but ignored by the core wrapper.
    """
    cfg = KZ1AConfig(preset=preset)
    res = simulate_kz1a_eci(cfg, fault=fault)

    if out_npz:
        theta = np.zeros_like(res["t"]) if controls_alpha_deg is None else np.asarray(controls_alpha_deg).ravel()
        np.savez(out_npz, t=res["t"], h=res["h"], v=res["speed"], q=res["q_dyn"], n=res["n_load"], theta=theta)
    if out_json:
        meta = {
            "preset": preset,
            "final_state": {"t_s": float(res["t"][-1]), "h_m": float(res["h"][-1]), "v_mps": float(res["speed"][-1])},
            "fault": fault,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return res
