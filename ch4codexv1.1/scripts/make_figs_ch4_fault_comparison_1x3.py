#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成每种故障的1行3列对比图：轻度/中度/重度故障下的名义/开环/重规划轨迹。

输出: outputs/figures/ch4_mission_domains/fault_comparison/
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.plots.plotting import setup_matplotlib, save_figure

# 故障配置
FAULTS = {
    'F1': {'id': 'F1_thrust_deg15', 'name': 'F1: 推力降级'},
    'F2': {'id': 'F2_tvc_rate4', 'name': 'F2: TVC速率限制'},
    'F3': {'id': 'F3_tvc_stuck3deg', 'name': 'F3: TVC卡滞'},
    'F4': {'id': 'F4_sensor_bias2deg', 'name': 'F4: 传感器偏置'},
    'F5': {'id': 'F5_event_delay5s', 'name': 'F5: 事件延迟'},
}

ETAS = [0.2, 0.5, 0.8]
ETA_LABELS = {
    0.2: '轻度故障 (η=0.2)\nRETAIN域',
    0.5: '中度故障 (η=0.5)\nDEGRADED域',
    0.8: '重度故障 (η=0.8)\nSAFE_AREA域',
}

DATA_DIR = PROJECT_ROOT / 'outputs' / 'data' / 'ch4_trajectories_replan'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures' / 'ch4_mission_domains' / 'fault_comparison'


def load_trajectory_data(fault_key: str, eta: float):
    """加载轨迹数据。"""
    # 名义轨迹
    nom_file = DATA_DIR / 'nominal.npz'
    nom_data = np.load(nom_file)

    # 故障开环和重规划
    eta_str = f'{eta:.1f}'.replace('.', '')  # 0.2 -> 02
    openloop_file = DATA_DIR / f'{fault_key}_eta{eta_str}_openloop.npz'
    replan_file = DATA_DIR / f'{fault_key}_eta{eta_str}_replan.npz'

    openloop_data = np.load(openloop_file)
    replan_data = np.load(replan_file)

    return {
        'nominal': {
            'downrange': nom_data['downrange'],
            'altitude': nom_data['altitude'],
        },
        'openloop': {
            'downrange': openloop_data['downrange'],
            'altitude': openloop_data['altitude'],
        },
        'replan': {
            'downrange': replan_data['downrange'],
            'altitude': replan_data['altitude'],
        }
    }


def plot_fault_comparison(fault_key: str, fault_info: dict):
    """为单个故障生成1行3列对比图。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for col, eta in enumerate(ETAS):
        ax = axes[col]

        try:
            data = load_trajectory_data(fault_key, eta)

            # 绘制名义轨迹（蓝色实线）
            ax.plot(data['nominal']['downrange'], data['nominal']['altitude'],
                   'b-', linewidth=2, label='名义轨迹')

            # 绘制故障开环（红色虚线）
            ax.plot(data['openloop']['downrange'], data['openloop']['altitude'],
                   'r--', linewidth=1.5, label='故障开环')

            # 绘制重规划轨迹（绿色点划线）
            ax.plot(data['replan']['downrange'], data['replan']['altitude'],
                   'g-.', linewidth=2, label='重规划轨迹')

            # 标记终点
            ax.scatter(data['nominal']['downrange'][-1], data['nominal']['altitude'][-1],
                      color='blue', s=100, marker='*', zorder=10)
            ax.scatter(data['replan']['downrange'][-1], data['replan']['altitude'][-1],
                      color='green', s=80, marker='^', zorder=10)

            # 如果重规划落地，标记安全落区
            if data['replan']['altitude'][-1] < 50:
                ax.axhline(y=0, color='brown', linestyle=':', alpha=0.5)
                ax.annotate('安全落区',
                           xy=(data['replan']['downrange'][-1], 10),
                           fontsize=9, color='brown')

        except Exception as e:
            ax.text(0.5, 0.5, f'数据加载失败\n{e}',
                   transform=ax.transAxes, ha='center', va='center')

        ax.set_xlabel('地面行距 (km)')
        ax.set_title(ETA_LABELS[eta], fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0, 12000)
        ax.set_ylim(-20, 550)

        if col == 0:
            ax.set_ylabel('高度 (km)')
            ax.legend(loc='upper left', fontsize=9)

    fig.suptitle(fault_info['name'], fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def main():
    setup_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    for fault_key, fault_info in FAULTS.items():
        print(f"生成 {fault_info['name']} 对比图...")

        fig = plot_fault_comparison(fault_key, fault_info)

        outpath = OUTPUT_DIR / f'{fault_key}_comparison_1x3.png'
        save_figure(fig, outpath)
        print(f"  -> {outpath}")
        plt.close(fig)

    print("=" * 60)
    print("完成！")


if __name__ == '__main__':
    main()
