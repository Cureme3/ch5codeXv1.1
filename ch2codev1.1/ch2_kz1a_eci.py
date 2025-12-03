# -*- coding: utf-8 -*-
"""
Chapter 2 CLI: wraps the unified KZ-1A ECI core and keeps plotting/output format.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 出版级绘图风格（宋体 + Times New Roman，符合中文期刊标准）
import matplotlib
import platform

system_name = platform.system()
if system_name == "Windows":
    font_serif = ["SimSun", "Times New Roman", "SimHei"]
    font_sans = ["SimHei", "Arial", "Microsoft YaHei"]
else:
    font_serif = ["Noto Serif CJK SC", "STSong", "Times New Roman"]
    font_sans = ["Noto Sans CJK SC", "STHeiti", "Arial"]

matplotlib.rcParams.update({
    # 字体核心设置
    "font.family": "serif",
    "font.serif": font_serif + list(matplotlib.rcParams["font.serif"]),
    "font.sans-serif": font_sans + list(matplotlib.rcParams["font.sans-serif"]),
    # 数学公式设置
    "mathtext.fontset": "stix",
    # 布局与线条
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.5,
    # 字号设置
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (6, 4.5),
    # 输出设置
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Allow importing the shared core (ch4codexv1.1/src as package root)
REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "ch4codexv1.1"
if str(CORE_ROOT) not in sys.path:
    sys.path.append(str(CORE_ROOT))

from src.sim.kz1a_eci_core import KZ1AConfig, FaultProfile, build_timeline_kz1a, simulate_kz1a_eci  # noqa: E402


def _save_plot(x, y, xlabel, ylabel, title, fname, outdir, events=None):
    plt.figure()
    plt.plot(x, y)
    color_map = {
        "S1": "tab:blue",
        "S2": "tab:orange",
        "FAIR": "tab:green",
        "S3": "tab:red",
        "S4": "tab:purple",
        "SC": "tab:brown",
    }
    label_map = {
        "S1": "Stage 1",
        "S2": "Stage 2",
        "FAIR": "Fairing",
        "S3": "Stage 3",
        "S4": "Stage 4",
        "SC": "SC Sep",
    }
    shown = set()
    if events:
        xarr = np.asarray(x)
        yarr = np.asarray(y)
        for name, te in events.items():
            if te < xarr[0] or te > xarr[-1]:
                continue
            if name.startswith("S1_"):
                key = "S1"
            elif name.startswith("S2_"):
                key = "S2"
            elif name.startswith("S3_"):
                key = "S3"
            elif name.startswith("S4_"):
                key = "S4"
            elif name.startswith("Fairing"):
                key = "FAIR"
            elif name.startswith("SC_"):
                key = "SC"
            else:
                key = "SC"
            idx = np.searchsorted(xarr, te)
            if idx <= 0 or idx >= len(xarr):
                yi = yarr[min(max(idx, 0), len(yarr) - 1)]
            else:
                x0, x1 = xarr[idx - 1], xarr[idx]
                y0, y1 = yarr[idx - 1], yarr[idx]
                if x1 == x0:
                    yi = y0
                else:
                    w = (te - x0) / (x1 - x0)
                    yi = (1 - w) * y0 + w * y1
            if key not in shown:
                plt.scatter([te], [yi], s=28, c=color_map[key], label=label_map[key])
                shown.add(key)
            else:
                plt.scatter([te], [yi], s=28, c=color_map[key])
    plt.grid(alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if events:
        plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=180)
    plt.close()


def simulate(cfg: KZ1AConfig, outdir: str, fault: FaultProfile = None):
    res = simulate_kz1a_eci(cfg, fault=fault)
    os.makedirs(outdir, exist_ok=True)

    t = res["t"]
    h = res["h"]
    speed = res["speed"]
    q_dyn = res["q_dyn"]
    n_load = res["n_load"]
    mass = res["mass"]
    thrust = res["thrust"]
    stage = res["stage"]
    # derive vr/vt/fpa if available or compute from state
    vr = res["vr"] if "vr" in res else None
    vt = res["vt"] if "vt" in res else None
    fpa_deg = res["fpa_deg"] if "fpa_deg" in res else None
    r_eci = res["r_eci"]
    v_eci = res["v_eci"]
    if vr is None or vt is None or fpa_deg is None:
        vr_list = []
        vt_list = []
        fpa_list = []
        for ri, vi in zip(r_eci, v_eci):
            rnorm = np.linalg.norm(ri)
            er = ri / max(rnorm, 1e-9)
            vr_i = np.dot(vi, er)
            vnorm = np.linalg.norm(vi)
            vt_i = math.sqrt(max(vnorm * vnorm - vr_i * vr_i, 0.0))
            gamma = math.degrees(math.atan2(vr_i, vt_i + 1e-9))
            vr_list.append(vr_i)
            vt_list.append(vt_i)
            fpa_list.append(gamma)
        vr = np.array(vr_list)
        vt = np.array(vt_list)
        fpa_deg = np.array(fpa_list)
    # guidance diagnostics (compute if missing)
    vr = res["vr"] if "vr" in res else None
    vt = res["vt"] if "vt" in res else None
    fpa_deg = res["fpa_deg"] if "fpa_deg" in res else None
    r_eci = res["r_eci"]
    v_eci = res["v_eci"]
    if vr is None or vt is None or fpa_deg is None:
        vr_list = []
        vt_list = []
        fpa_list = []
        for ri, vi in zip(r_eci, v_eci):
            rnorm = np.linalg.norm(ri)
            er = ri / max(rnorm, 1e-9)
            vr_i = np.dot(vi, er)
            vnorm = np.linalg.norm(vi)
            vt_i = math.sqrt(max(vnorm * vnorm - vr_i * vr_i, 0.0))
            gamma = math.degrees(math.atan2(vr_i, vt_i + 1e-9))
            vr_list.append(vr_i)
            vt_list.append(vt_i)
            fpa_list.append(gamma)
        vr = np.array(vr_list)
        vt = np.array(vt_list)
        fpa_deg = np.array(fpa_list)

    np.savez(
        os.path.join(outdir, "timeseries_eci.npz"),
        t=t,
        h=h,
        speed=speed,
        vr=vr,
        vt=vt,
        fpa_deg=fpa_deg,
        q_dyn=q_dyn,
        n_load=n_load,
        mass=mass,
        thrust=thrust,
        stage=stage,
    )

    idx_q = int(np.nanargmax(q_dyn)) if q_dyn.size else 0
    idx_n = int(np.nanargmax(n_load)) if n_load.size else 0

    summary = {
        "preset": cfg.preset,
        "q_peak_ascent_Pa": float(q_dyn[idx_q]) if q_dyn.size else None,
        "t_at_q_peak_ascent_s": float(t[idx_q]) if t.size else None,
        "h_at_q_peak_ascent_m": float(h[idx_q]) if h.size else None,
        "v_at_q_peak_ascent_mps": float(speed[idx_q]) if speed.size else None,
        "n_max_ascent_g": float(n_load[idx_n]) if n_load.size else None,
        "t_at_n_max_ascent_s": float(t[idx_n]) if t.size else None,
        "h_at_n_max_ascent_m": float(h[idx_n]) if h.size else None,
        "v_at_n_max_ascent_mps": float(speed[idx_n]) if speed.size else None,
        "events": res["events"],
        "final_state": res["summary"]["final_state"],
    }
    with open(os.path.join(outdir, "summary_eci.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    events_time = {k: v["t_s"] for k, v in res["events"].items()}
    _save_plot(t, h / 1000.0, "时间 (s)", "高度 (km)", "高度—时间", "fig_h_t.png", outdir, events=events_time)
    _save_plot(t, speed, "时间 (s)", "速度 (m/s)", "速度—时间", "fig_v_t.png", outdir, events=events_time)
    _save_plot(t, q_dyn, "时间 (s)", "动压 q (Pa)", "动压—时间", "fig_q_t.png", outdir, events=events_time)
    _save_plot(t, n_load, "时间 (s)", "过载 n (g₀)", "过载—时间", "fig_n_t.png", outdir, events=events_time)
    _save_plot(t, mass, "时间 (s)", "质量 (kg)", "质量—时间", "fig_m_t.png", outdir, events=events_time)
    _save_plot(t, vr / 1000.0, "时间 (s)", "径向速度 vr (km/s)", "径向速度—时间", "fig_vr_t.png", outdir, events=events_time)
    _save_plot(t, fpa_deg, "时间 (s)", "航迹角 γ (deg)", "航迹角—时间", "fig_fpa_t.png", outdir, events=events_time)

    # Radial acceleration diagnostics
    diag_path = Path(outdir) / "radial_accel_diagnostics.csv"
    # derive vr, vt if not present
    vr = res["vr"] if "vr" in res else None
    vt = res["vt"] if "vt" in res else None
    if vr is None or vt is None:
        r_eci = res["r_eci"]
        v_eci = res["v_eci"]
        vr_list = []
        vt_list = []
        for ri, vi in zip(r_eci, v_eci):
            rnorm = np.linalg.norm(ri)
            er = ri / max(rnorm, 1e-9)
            vr_i = np.dot(vi, er)
            vnorm = np.linalg.norm(vi)
            vt_i = math.sqrt(max(vnorm * vnorm - vr_i * vr_i, 0.0))
            vr_list.append(vr_i)
            vt_list.append(vt_i)
        vr = np.array(vr_list)
        vt = np.array(vt_list)
    a_r_total = res.get("a_r_total")
    a_r_thrust = res.get("a_r_thrust")
    a_r_drag = res.get("a_r_drag")
    a_r_grav = res.get("a_r_grav")
    pitch_cmd = res.get("pitch_cmd_deg")
    fpa_deg = res.get("fpa_deg")
    diag_data = np.column_stack(
        [
            t,
            h,
            vr,
            vt,
            a_r_total,
            a_r_thrust,
            a_r_drag,
            a_r_grav,
            pitch_cmd,
            fpa_deg,
        ]
    )
    np.savetxt(
        diag_path,
        diag_data,
        delimiter=",",
        header="t,alt,vr,vt,a_r_total,a_r_thrust,a_r_drag,a_r_grav,pitch_cmd_deg,fpa_deg",
        comments="",
    )
    print(f"[INFO] Saved radial acceleration diagnostics to {diag_path}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] Saved to {outdir}")
    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["nasaspaceflight", "everydayastronaut", "kz1a_public"], default="kz1a_public")
    p.add_argument("--outdir", type=str, default="./exports/ch2_orbit")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--t_end", type=float, default=1100.0)
    p.add_argument("--mass_liftoff_kg", type=float, default=None)
    p.add_argument("--payload_mass_kg", type=float, default=260.0)
    p.add_argument("--Cd_fairing", type=float, default=0.5)
    p.add_argument("--Cd_post", type=float, default=0.5)
    p.add_argument("--pitch_start_s", type=float, default=5.0)
    p.add_argument("--pitch_mid_s", type=float, default=70.0)
    p.add_argument("--pitch_horiz_s", type=float, default=170.0)
    p.add_argument("--pitch_mid_deg", type=float, default=30.0)
    p.add_argument("--pitch_final_deg", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = KZ1AConfig(
        preset=args.preset,
        dt=args.dt,
        t_end=args.t_end,
        mass_liftoff_kg=args.mass_liftoff_kg,
        payload_mass_kg=args.payload_mass_kg,
        Cd_fairing=args.Cd_fairing,
        Cd_post=args.Cd_post,
        pitch_start_s=args.pitch_start_s,
        pitch_mid_s=args.pitch_mid_s,
        pitch_horiz_s=args.pitch_horiz_s,
        pitch_mid_deg=args.pitch_mid_deg,
        pitch_final_deg=args.pitch_final_deg,
    )
    simulate(cfg, outdir=args.outdir, fault=None)


if __name__ == "__main__":
    main()
