#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成每种故障的1x3 ECI 3D轨迹对比图。

每个故障一张图（1行3列），分别对应轻度/中度/重度故障。
每个子图包含：名义轨迹、故障开环、重规划轨迹，稍微拉开距离便于区分。

输出: outputs/figures/ch4_mission_domains/fault_comparison/
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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.plots.plotting import setup_matplotlib

# 故障配置
FAULTS = {
    'F1': {'name': 'F1: 推力降级'},
    'F2': {'name': 'F2: TVC速率限制'},
    'F3': {'name': 'F3: TVC卡滞'},
    'F4': {'name': 'F4: 传感器偏置'},
    'F5': {'name': 'F5: 事件延迟'},
}

ETAS = [0.2, 0.5, 0.8]
ETA_LABELS = {
    0.2: '轻度故障 (η=0.2)\nRETAIN域',
    0.5: '中度故障 (η=0.5)\nDEGRADED域',
    0.8: '重度故障 (η=0.8)\nSAFE_AREA域',
}

DATA_DIR = PROJECT_ROOT / 'outputs' / 'data' / 'ch4_trajectories_replan'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures' / 'ch4_mission_domains' / 'fault_comparison'

# 地球半径 (km)
R_EARTH_KM = 6371.0


def load_trajectory_data(fault_key: str, eta: float):
    """加载轨迹数据。"""
    eta_str = f'{eta:.1f}'.replace('.', '')
    openloop_file = DATA_DIR / f'{fault_key}_eta{eta_str}_openloop.npz'
    replan_file = DATA_DIR / f'{fault_key}_eta{eta_str}_replan.npz'

    openloop_data = np.load(openloop_file) if openloop_file.exists() else None
    replan_data = np.load(replan_file) if replan_file.exists() else None

    return openloop_data, replan_data




def plot_fault_3d_1x3(fault_key: str, fault_info: dict):
    """为单个故障生成1x3的3D轨迹图。"""
    fig = plt.figure(figsize=(18, 6))

    # 加载名义轨迹
    nom_file = DATA_DIR / 'nominal.npz'
    nom_data = np.load(nom_file) if nom_file.exists() else None

    for col, eta in enumerate(ETAS):
        ax = fig.add_subplot(1, 3, col + 1, projection='3d')

        # Y方向偏移量，拉开三条轨迹（减小偏移使轨迹更紧凑）
        OFFSETS = {'nominal': -600, 'openloop': 0, 'replan': 600}

        # 名义轨迹（蓝色）
        if nom_data is not None and 'r_eci' in nom_data:
            r = nom_data['r_eci'] / 1000.0
            step = max(1, len(r) // 600)
            ax.plot(r[::step, 0], r[::step, 1] + OFFSETS['nominal'], r[::step, 2],
                    '-', color='#0000CD', linewidth=2.5, label='名义轨迹', alpha=0.9)
            ax.scatter(r[-1, 0], r[-1, 1] + OFFSETS['nominal'], r[-1, 2],
                      color='#0000CD', s=80, marker='o')

        # 加载故障数据
        openloop_data, replan_data = load_trajectory_data(fault_key, eta)

        # 故障开环（红色虚线）
        if openloop_data is not None and 'r_eci' in openloop_data:
            r = openloop_data['r_eci'] / 1000.0
            step = max(1, len(r) // 600)
            ax.plot(r[::step, 0], r[::step, 1] + OFFSETS['openloop'], r[::step, 2],
                    '--', color='#DC143C', linewidth=2, label='故障开环', alpha=0.9)
            ax.scatter(r[-1, 0], r[-1, 1] + OFFSETS['openloop'], r[-1, 2],
                      color='#DC143C', s=100, marker='x', linewidths=2)

        # 重规划轨迹（绿色）
        if replan_data is not None and 'r_eci' in replan_data:
            r = replan_data['r_eci'] / 1000.0
            step = max(1, len(r) // 600)
            ax.plot(r[::step, 0], r[::step, 1] + OFFSETS['replan'], r[::step, 2],
                    '-', color='#228B22', linewidth=2, label='重规划轨迹', alpha=0.9)
            ax.scatter(r[-1, 0], r[-1, 1] + OFFSETS['replan'], r[-1, 2],
                      color='#228B22', s=80, marker='*')

        # 设置坐标轴
        ax.set_xlabel('X (km)', fontsize=10)
        ax.set_ylabel('Y (km)', fontsize=10)
        ax.set_zlabel('Z (km)', fontsize=10)

        ax.set_xlim([-6000, 12000])
        ax.set_ylim([-6000, 6000])
        ax.set_zlim([-4000, 12000])

        ax.set_title(ETA_LABELS[eta], fontsize=11)
        if col == 0:
            ax.legend(loc='upper left', fontsize=9)

        ax.view_init(elev=25, azim=-50)

    fig.suptitle(f"{fault_info['name']} - 3D轨迹对比 (ECI)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def main():
    setup_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    for fault_key, fault_info in FAULTS.items():
        print(f"生成 {fault_info['name']} 1x3 3D图...")

        fig = plot_fault_3d_1x3(fault_key, fault_info)

        outpath = OUTPUT_DIR / f'{fault_key}_3d_trajectory.png'
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        fig.savefig(outpath.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
        print(f"  -> {outpath}")
        plt.close(fig)

    print("=" * 60)
    print("完成！")


if __name__ == '__main__':
    main()
