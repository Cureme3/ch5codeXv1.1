#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成2x2组合轨迹图，使用ch4_trajectories_replan中的数据。

输出: outputs/figures/ch4_mission_domains/combined_2x2/
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.plots.plotting import setup_matplotlib

FAULTS = ['F1', 'F2', 'F3', 'F4', 'F5']
FAULT_NAMES = {
    'F1': 'F1_thrust_deg15',
    'F2': 'F2_tvc_rate4',
    'F3': 'F3_tvc_stuck3deg',
    'F4': 'F4_sensor_bias2deg',
    'F5': 'F5_event_delay5s',
}
ETAS = [0.2, 0.5, 0.8, 1.0]

DATA_DIR = PROJECT_ROOT / 'outputs' / 'data' / 'ch4_trajectories_replan'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures' / 'ch4_mission_domains' / 'combined_2x2'


def load_data(fault_key: str, eta: float):
    """加载开环轨迹数据。"""
    eta_str = f'{eta:.1f}'.replace('.', '')
    if eta == 1.0:
        eta_str = '10'
    fpath = DATA_DIR / f'{fault_key}_eta{eta_str}_openloop.npz'
    if fpath.exists():
        data = np.load(fpath)
        # 计算速度大小 (km/s)
        v_eci = data['v_eci'] / 1000.0  # m/s -> km/s
        velocity = np.linalg.norm(v_eci, axis=1)
        # 兼容不同的时间字段名
        t = data['t'] if 't' in data else data['time']
        return {
            't': t,
            'h': data['altitude'],
            'v': velocity,
            's': data['downrange'],
        }
    return None


def generate_combined_2x2(fault_key: str):
    """生成单个故障的2x2组合图。"""
    # 加载名义轨迹
    nom_file = DATA_DIR / 'nominal.npz'
    if not nom_file.exists():
        print(f"  名义轨迹文件不存在: {nom_file}")
        return
    nom = np.load(nom_file)

    t_nom = nom['t'] if 't' in nom else nom['time']
    h_nom = nom['altitude']
    v_eci_nom = nom['v_eci'] / 1000.0  # m/s -> km/s
    v_nom = np.linalg.norm(v_eci_nom, axis=1)
    s_nom = nom['downrange']

    # 加载各eta的故障数据（只用0.2, 0.5, 0.8）
    fault_data = {}
    for eta in [0.2, 0.5, 0.8]:
        data = load_data(fault_key, eta)
        if data is not None:
            fault_data[eta] = data

    if not fault_data:
        print(f"  {fault_key} 没有故障数据")
        return

    # 确定时间范围 - F4/F5故障影响在后期，需要更长时间
    if fault_key in ['F4', 'F5']:
        t_max = min(t_nom[-1], min(d['t'][-1] for d in fault_data.values()))
        t_max = min(t_max, 1200)  # F4/F5显示到1200s
    else:
        t_max = min(t_nom[-1], min(d['t'][-1] for d in fault_data.values()))
        t_max = min(t_max, 400)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(fault_data)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    mask_nom = t_nom <= t_max

    # (0,0) 高度 vs 时间
    ax = axes[0, 0]
    ax.plot(t_nom[mask_nom], h_nom[mask_nom], 'b-', linewidth=2, label='名义轨迹')
    for (eta, result), color in zip(fault_data.items(), colors):
        mask = result['t'] <= t_max
        ax.plot(result['t'][mask], result['h'][mask], '-', color=color,
                linewidth=1.5, label=f'故障轨迹 (η={eta})')
    ax.set_xlabel('时间 t (s)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('高度 vs 时间')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)

    # (0,1) 速度 vs 时间
    ax = axes[0, 1]
    ax.plot(t_nom[mask_nom], v_nom[mask_nom], 'b-', linewidth=2, label='名义轨迹')
    for (eta, result), color in zip(fault_data.items(), colors):
        mask = result['t'] <= t_max
        ax.plot(result['t'][mask], result['v'][mask], '-', color=color,
                linewidth=1.5, label=f'故障轨迹 (η={eta})')
    ax.set_xlabel('时间 t (s)')
    ax.set_ylabel('速度 v (km/s)')
    ax.set_title('速度 vs 时间')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)

    # (1,0) 下航程 vs 时间
    ax = axes[1, 0]
    ax.plot(t_nom[mask_nom], s_nom[mask_nom], 'b-', linewidth=2, label='名义轨迹')
    for (eta, result), color in zip(fault_data.items(), colors):
        mask = result['t'] <= t_max
        ax.plot(result['t'][mask], result['s'][mask], '-', color=color,
                linewidth=1.5, label=f'故障轨迹 (η={eta})')
    ax.set_xlabel('时间 t (s)')
    ax.set_ylabel('下航程 s (km)')
    ax.set_title('下航程 vs 时间')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)

    # (1,1) 高度-速度相图
    ax = axes[1, 1]
    ax.plot(v_nom[mask_nom], h_nom[mask_nom], 'b-', linewidth=2, label='名义轨迹')
    for (eta, result), color in zip(fault_data.items(), colors):
        mask = result['t'] <= t_max
        ax.plot(result['v'][mask], result['h'][mask], '-', color=color,
                linewidth=1.5, label=f'故障轨迹 (η={eta})')
    ax.set_xlabel('速度 v (km/s)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('高度-速度相图')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{FAULT_NAMES[fault_key]}: 不同故障严重度下的轨迹对比', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    setup_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    for fault_key in FAULTS:
        print(f"生成 {FAULT_NAMES[fault_key]} 2x2组合图...")
        fig = generate_combined_2x2(fault_key)
        if fig:
            outpath = OUTPUT_DIR / f'{FAULT_NAMES[fault_key]}_combined_2x2.png'
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            fig.savefig(outpath.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
            print(f"  -> {outpath}")
            plt.close(fig)

    print("=" * 60)
    print("完成！")


if __name__ == '__main__':
    main()
