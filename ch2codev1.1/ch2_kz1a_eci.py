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
import matplotlib
import numpy as np
import platform

# 出版级绘图配置
system_name = platform.system()
if system_name == "Windows":
    font_serif = ["SimSun", "Times New Roman", "SimHei"]
    font_sans = ["SimHei", "Arial", "Microsoft YaHei"]
else:
    font_serif = ["Noto Serif CJK SC", "STSong", "Times New Roman"]
    font_sans = ["Noto Sans CJK SC", "STHeiti", "Arial"]

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": font_serif + list(matplotlib.rcParams["font.serif"]),
    "font.sans-serif": font_sans + list(matplotlib.rcParams["font.sans-serif"]),
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.5,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (8, 5),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Allow importing the shared core
REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "ch4codexv1.1"
if str(CORE_ROOT) not in sys.path:
    sys.path.append(str(CORE_ROOT))

from src.sim.kz1a_eci_core import KZ1AConfig, FaultProfile, simulate_kz1a_eci  # noqa: E402

# 出版级配色
DEFAULT_COLORS = {
    "nominal": "#00468B",
    "fault": "#ED0000",
    "replan": "#42B540",
}
LINE_WIDTH = 2.0

# 中文阶段标签
STAGE_LABELS = {
    "S1_ign": "一级点火",
    "S1_sep": "一级分离",
    "S2_ign": "二级点火",
    "S2_sep": "二级分离",
    "Fairing_jettison": "整流罩分离",
    "S3_ign": "三级点火",
    "S3_sep": "三级分离",
    "S4_ign": "四级点火",
    "S4_cutoff": "四级关机",
    "SC_sep": "星箭分离",
}

# 阶段颜色
STAGE_COLORS = {
    "S1": "#00468B",
    "S2": "#ED0000",
    "S3": "#42B540",
    "S4": "#925E9F",
    "FAIR": "#ADB6B6",
    "SC": "#0099B4",
}


def _get_stage_key(event_name):
    if event_name.startswith("S1_"):
        return "S1"
    elif event_name.startswith("S2_"):
        return "S2"
    elif event_name.startswith("S3_"):
        return "S3"
    elif event_name.startswith("S4_"):
        return "S4"
    elif event_name.startswith("Fairing"):
        return "FAIR"
    elif event_name.startswith("SC_"):
        return "SC"
    return None


def _save_plot(x, y, xlabel, ylabel, title, fname, outdir, events=None, xlim=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color=DEFAULT_COLORS["nominal"], linewidth=LINE_WIDTH)

    legend_handles = []
    if events:
        xarr = np.asarray(x)
        yarr = np.asarray(y)

        for name, te in events.items():
            if te < xarr[0] or te > xarr[-1]:
                continue
            if xlim and (te < xlim[0] or te > xlim[1]):
                continue

            key = _get_stage_key(name)
            if key is None:
                continue

            idx = np.searchsorted(xarr, te)
            if idx <= 0:
                yi = yarr[0]
            elif idx >= len(xarr):
                yi = yarr[-1]
            else:
                x0, x1 = xarr[idx - 1], xarr[idx]
                y0, y1 = yarr[idx - 1], yarr[idx]
                w = (te - x0) / (x1 - x0) if x1 != x0 else 0
                yi = (1 - w) * y0 + w * y1

            color = STAGE_COLORS.get(key, "#000000")
            label = STAGE_LABELS.get(name, name)
            sc = ax.scatter([te], [yi], s=50, c=color, zorder=5, edgecolors='white', linewidths=0.5, label=label)
            legend_handles.append(sc)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    ax.grid(True, linestyle='--', alpha=0.5)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='best', fontsize=9, framealpha=0.9)

    outpath = Path(outdir) / fname
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    fig.savefig(outpath.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {outpath}")


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
    r_eci = res["r_eci"]
    v_eci = res["v_eci"]

    vr_list, vt_list, fpa_list = [], [], []
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
        t=t, h=h, speed=speed, vr=vr, vt=vt, fpa_deg=fpa_deg,
        q_dyn=q_dyn, n_load=n_load, mass=mass, thrust=thrust, stage=stage,
    )

    idx_q = int(np.nanargmax(q_dyn)) if q_dyn.size else 0
    idx_n = int(np.nanargmax(n_load)) if n_load.size else 0

    r_final = r_eci[-1]
    v_final = v_eci[-1]
    r_norm = np.linalg.norm(r_final)
    v_norm = np.linalg.norm(v_final)
    Re = 6371000.0
    mu = 3.986004418e14
    h_vec = np.cross(r_final, v_final)
    e_vec = np.cross(v_final, h_vec) / mu - r_final / r_norm
    ecc = np.linalg.norm(e_vec)
    a = 1.0 / (2.0 / r_norm - v_norm * v_norm / mu)
    ha = a * (1 + ecc) - Re
    hp = a * (1 - ecc) - Re

    summary = {
        "preset": cfg.preset,
        "dt": cfg.dt,
        "t_end": cfg.t_end,
        "q_peak_ascent_Pa": float(q_dyn[idx_q]) if q_dyn.size else None,
        "t_at_q_peak_ascent_s": float(t[idx_q]) if t.size else None,
        "n_max_ascent_g": float(n_load[idx_n]) if n_load.size else None,
        "t_at_n_max_ascent_s": float(t[idx_n]) if t.size else None,
        "events": res["events"],
        "final_orbit": {
            "altitude_km": float(h[-1] / 1000),
            "sma_km": float((a - Re) / 1000),
            "apoapsis_km": float(ha / 1000),
            "periapsis_km": float(hp / 1000),
            "eccentricity": float(ecc),
        },
    }
    with open(os.path.join(outdir, "summary_eci.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    events_time = {k: v["t_s"] for k, v in res["events"].items()}
    t_ascent_end = 500.0

    # 全程视图
    _save_plot(t, h / 1000.0, "时间 (s)", "高度 (km)",
               "高度—时间曲线", "fig_h_t.png", outdir, events=events_time)
    _save_plot(t, speed / 1000.0, "时间 (s)", "速度 (km/s)",
               "速度—时间曲线", "fig_v_t.png", outdir, events=events_time)
    _save_plot(t, mass, "时间 (s)", "质量 (kg)",
               "质量—时间曲线", "fig_m_t.png", outdir, events=events_time)
    _save_plot(t, vr / 1000.0, "时间 (s)", "径向速度 (km/s)",
               "径向速度—时间曲线", "fig_vr_t.png", outdir, events=events_time)
    _save_plot(t, fpa_deg, "时间 (s)", "航迹角 (°)",
               "航迹角—时间曲线", "fig_fpa_t.png", outdir, events=events_time)

    # 动力上升段放大图
    _save_plot(t, q_dyn / 1000.0, "时间 (s)", "动压 (kPa)",
               "动压—时间曲线（动力上升段）", "fig_q_t.png", outdir,
               events=events_time, xlim=(0, t_ascent_end))
    _save_plot(t, n_load, "时间 (s)", "过载 (g)",
               "过载—时间曲线（动力上升段）", "fig_n_t.png", outdir,
               events=events_time, xlim=(0, t_ascent_end))
    _save_plot(t, thrust / 1000.0, "时间 (s)", "推力 (kN)",
               "推力—时间曲线（动力上升段）", "fig_thrust_t.png", outdir,
               events=events_time, xlim=(0, t_ascent_end))

    print("\n" + "="*50)
    print("入轨仿真结果")
    print("="*50)
    print(f"仿真时长: {t[-1]:.1f} s")
    print(f"最大动压: {q_dyn[idx_q]/1000:.2f} kPa @ t={t[idx_q]:.1f}s")
    print(f"最大过载: {n_load[idx_n]:.2f} g @ t={t[idx_n]:.1f}s")
    print(f"\n最终轨道:")
    print(f"  半长轴: {(a-Re)/1000:.2f} km")
    print(f"  远地点: {ha/1000:.2f} km")
    print(f"  近地点: {hp/1000:.2f} km")
    print(f"  偏心率: {ecc:.5f}")
    print("="*50)

    return res


def parse_args():
    p = argparse.ArgumentParser(description="KZ-1A 入轨仿真（第二章）")
    p.add_argument("--preset", choices=["nasaspaceflight", "everydayastronaut", "kz1a_public"],
                   default="nasaspaceflight")
    p.add_argument("--outdir", type=str, default="./exports")
    p.add_argument("--dt", type=float, default=0.01, help="仿真步长 (s)")
    p.add_argument("--t_end", type=float, default=4000.0, help="仿真结束时间 (s)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = KZ1AConfig(
        preset=args.preset,
        dt=args.dt,
        t_end=args.t_end,
    )
    simulate(cfg, outdir=args.outdir, fault=None)


if __name__ == "__main__":
    main()
