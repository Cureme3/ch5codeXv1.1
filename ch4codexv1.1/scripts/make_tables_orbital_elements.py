#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成轨道六根数对比表格。

每个故障一个表格，对比名义入轨与不同故障程度重规划后的轨道六根数。
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'outputs' / 'data' / 'ch4_trajectories_replan'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures' / 'ch4_mission_domains' / 'orbital_elements'

# 地球引力常数 (m^3/s^2)
MU = 3.986004418e14

FAULTS = ['F1', 'F2', 'F3', 'F4', 'F5']
FAULT_NAMES = {
    'F1': 'F1: 推力降级',
    'F2': 'F2: TVC速率限制',
    'F3': 'F3: TVC卡滞',
    'F4': 'F4: 传感器偏置',
    'F5': 'F5: 事件延迟',
}
ETAS = [0.2, 0.5, 0.8]


def rv_to_orbital_elements(r_vec, v_vec):
    """从位置速度向量计算轨道六根数。

    Args:
        r_vec: 位置向量 [m]
        v_vec: 速度向量 [m/s]

    Returns:
        dict: 轨道六根数 {a, e, i, Omega, omega, nu}
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # 角动量向量
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # 节线向量
    k = np.array([0, 0, 1])
    n_vec = np.cross(k, h_vec)
    n = np.linalg.norm(n_vec)

    # 偏心率向量
    e_vec = ((v**2 - MU/r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / MU
    e = np.linalg.norm(e_vec)

    # 半长轴
    xi = v**2 / 2 - MU / r  # 比轨道能
    if abs(e - 1.0) < 1e-10:
        a = np.inf
    else:
        a = -MU / (2 * xi)

    # 轨道倾角
    i = np.arccos(np.clip(h_vec[2] / h, -1, 1))

    # 升交点赤经
    if n > 1e-10:
        Omega = np.arccos(np.clip(n_vec[0] / n, -1, 1))
        if n_vec[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0.0

    # 近地点幅角
    if n > 1e-10 and e > 1e-10:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0

    # 真近点角
    if e > 1e-10:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0

    return {
        'a': a / 1000,  # km
        'e': e,
        'i': np.degrees(i),
        'Omega': np.degrees(Omega),
        'omega': np.degrees(omega),
        'nu': np.degrees(nu),
    }


def load_terminal_state(filepath):
    """加载轨迹终端状态。"""
    data = np.load(filepath)
    r_eci = data['r_eci'][-1]  # 终端位置 [m]
    v_eci = data['v_eci'][-1]  # 终端速度 [m/s]
    return r_eci, v_eci


def generate_fault_table(fault_key: str):
    """生成单个故障的轨道六根数对比表格。"""
    rows = []

    # 名义轨迹
    nom_file = DATA_DIR / 'nominal.npz'
    r, v = load_terminal_state(nom_file)
    oe = rv_to_orbital_elements(r, v)
    rows.append({
        '工况': '名义入轨',
        '半长轴 a (km)': f'{oe["a"]:.2f}',
        '偏心率 e': f'{oe["e"]:.6f}',
        '倾角 i (°)': f'{oe["i"]:.2f}',
        '升交点赤经 Ω (°)': f'{oe["Omega"]:.2f}',
        '近地点幅角 ω (°)': f'{oe["omega"]:.2f}',
        '真近点角 ν (°)': f'{oe["nu"]:.2f}',
    })

    # 各eta的重规划轨迹
    eta_labels = {0.2: '轻度故障重规划', 0.5: '中度故障重规划', 0.8: '重度故障重规划'}
    for eta in ETAS:
        eta_str = f'{eta:.1f}'.replace('.', '')
        replan_file = DATA_DIR / f'{fault_key}_eta{eta_str}_replan.npz'
        if replan_file.exists():
            r, v = load_terminal_state(replan_file)
            oe = rv_to_orbital_elements(r, v)
            rows.append({
                '工况': eta_labels[eta],
                '半长轴 a (km)': f'{oe["a"]:.2f}',
                '偏心率 e': f'{oe["e"]:.6f}',
                '倾角 i (°)': f'{oe["i"]:.2f}',
                '升交点赤经 Ω (°)': f'{oe["Omega"]:.2f}',
                '近地点幅角 ω (°)': f'{oe["omega"]:.2f}',
                '真近点角 ν (°)': f'{oe["nu"]:.2f}',
            })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    for fault_key in FAULTS:
        print(f"生成 {FAULT_NAMES[fault_key]} 轨道六根数表格...")
        df = generate_fault_table(fault_key)

        # 保存CSV
        csv_path = OUTPUT_DIR / f'{fault_key}_orbital_elements.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 保存Markdown
        md_path = OUTPUT_DIR / f'{fault_key}_orbital_elements.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {FAULT_NAMES[fault_key]} 轨道六根数对比\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")

        print(f"  -> {csv_path}")
        print(df.to_string(index=False))
        print()

    print("=" * 60)
    print("完成！")


if __name__ == '__main__':
    main()
