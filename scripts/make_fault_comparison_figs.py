#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""故障工况对比图: 名义 vs 推力下降30% vs 推力矢量偏置5°"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ch3codev1.1"))
sys.path.insert(0, str(PROJECT_ROOT / "ch4codexv1.1"))
sys.path.insert(0, str(PROJECT_ROOT / "ch4codexv1.1" / "src"))

from plots.plotting import setup_matplotlib, save_figure, DEFAULT_COLORS, LINE_WIDTH
from diagnosis.eso import run_eso
from diagnosis.features import extract_features_from_residual, pwvd, sample_entropy
from src.sim.run_fault import run_fault_scenario
from src.sim.run_nominal import simulate_nominal

# 应用出版级绘图样式
setup_matplotlib()

OUTPUT_DIR = PROJECT_ROOT / "ch3codev1.1" / "exports" / "fault_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 配色方案
COLOR_NOMINAL = DEFAULT_COLORS["nominal"]  # 深蓝
COLOR_THRUST = DEFAULT_COLORS["fault"]     # 鲜红
COLOR_TVC = DEFAULT_COLORS["replan"]       # 鲜绿


def get_residual(states, dt=1.0):
    """从状态轨迹提取ESO残差。"""
    v = states[:, 3:6]
    ax = np.diff(v[:, 0])
    az = np.diff(v[:, 2])
    axz = np.column_stack([ax, az])
    residuals, _, _ = run_eso(axz, dt=dt)
    return np.linalg.norm(residuals, axis=1)


def main():
    print("生成故障工况对比图...")
    dt = 1.0

    # 1. 名义轨迹
    print("[1/3] 名义轨迹仿真...")
    nom_sim = simulate_nominal(dt=dt)
    nom_states = nom_sim.states

    # 2. 推力下降30%
    print("[2/3] 推力下降30%仿真...")
    thrust_sim = run_fault_scenario("F1_thrust_deg15", eta=0.3, dt=dt)
    thrust_states = thrust_sim.states

    # 3. 推力矢量偏置5°
    print("[3/3] 推力矢量偏置5°仿真...")
    tvc_sim = run_fault_scenario("F3_tvc_stuck3deg", eta=0.42, dt=dt)
    tvc_states = tvc_sim.states

    # 计算残差
    print("计算ESO残差...")
    min_len = min(len(nom_states), len(thrust_states), len(tvc_states)) - 1
    nom_res = get_residual(nom_states[:min_len+1], dt)[:min_len]
    thrust_res = get_residual(thrust_states[:min_len+1], dt)[:min_len]
    tvc_res = get_residual(tvc_states[:min_len+1], dt)[:min_len]
    time_res = np.arange(min_len) * dt

    # ========== 图1: 残差对比 ==========
    print("绘制残差对比图...")
    fig, ax = plt.subplots()
    ax.plot(time_res, nom_res, color=COLOR_NOMINAL, linewidth=LINE_WIDTH, label='名义工况')
    ax.plot(time_res, thrust_res, color=COLOR_THRUST, linewidth=LINE_WIDTH, label='推力下降30%')
    ax.plot(time_res, tvc_res, color=COLOR_TVC, linewidth=LINE_WIDTH, label='TVC偏置5°')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('残差幅值')
    ax.set_title('ESO残差对比')
    ax.legend()
    save_figure(fig, OUTPUT_DIR / "fig1_residual_comparison.png")

    # ========== 图2: 时频特征对比 (PWVD) ==========
    print("绘制时频特征对比图...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cases = [
        (nom_res, '名义工况'),
        (thrust_res, '推力下降30%'),
        (tvc_res, 'TVC偏置5°')
    ]
    for idx, (res, title) in enumerate(cases):
        spec = pwvd(res, win_len=128)
        im = axes[idx].imshow(spec, aspect='auto', origin='lower',
                              extent=[0, len(res)*dt, 0, 0.5/dt],
                              cmap='jet', vmin=0, vmax=np.percentile(spec, 95))
        axes[idx].set_xlabel('时间 (s)')
        axes[idx].set_ylabel('频率 (Hz)')
        axes[idx].set_title(f'PWVD: {title}')
    fig.colorbar(im, ax=axes, shrink=0.8, label='幅值')
    save_figure(fig, OUTPUT_DIR / "fig2_pwvd_comparison.png")

    # ========== 图3: 样本熵对比 ==========
    print("计算样本熵...")
    win_size, step = 50, 10

    def sliding_sampen(res, win_size, step):
        sampen_vals, times = [], []
        for i in range(0, len(res) - win_size, step):
            seg = res[i:i+win_size]
            se = sample_entropy(seg, m=2, r=0.2*np.std(seg)+1e-12)
            sampen_vals.append(se)
            times.append((i + win_size//2) * dt)
        return np.array(times), np.array(sampen_vals)

    nom_t, nom_se = sliding_sampen(nom_res, win_size, step)
    thrust_t, thrust_se = sliding_sampen(thrust_res, win_size, step)
    tvc_t, tvc_se = sliding_sampen(tvc_res, win_size, step)

    fig, ax = plt.subplots()
    ax.plot(nom_t, nom_se, color=COLOR_NOMINAL, linewidth=LINE_WIDTH, label='名义工况')
    ax.plot(thrust_t, thrust_se, color=COLOR_THRUST, linewidth=LINE_WIDTH, label='推力下降30%')
    ax.plot(tvc_t, tvc_se, color=COLOR_TVC, linewidth=LINE_WIDTH, label='TVC偏置5°')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('样本熵')
    ax.set_title('样本熵对比 (窗口=50s)')
    ax.legend()
    save_figure(fig, OUTPUT_DIR / "fig3_sample_entropy_comparison.png")

    # ========== 图4: 三维能量分布对比 ==========
    print("绘制三维能量分布图...")
    fig = plt.figure(figsize=(14, 5))
    colors_3d = [COLOR_NOMINAL, COLOR_THRUST, COLOR_TVC]
    titles_3d = ['名义工况', '推力下降30%', 'TVC偏置5°']
    res_list = [nom_res, thrust_res, tvc_res]

    for idx in range(3):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        feats = extract_features_from_residual(res_list[idx], dt=dt, spec_method='pwvd')
        E_low, E_mid, E_high = feats[0], feats[1], feats[2]
        total = E_low + E_mid + E_high + 1e-12
        E_low_n, E_mid_n, E_high_n = E_low/total, E_mid/total, E_high/total

        ax.bar3d([0], [0], [0], [0.8], [0.8], [E_low_n], color='#00468B', alpha=0.8)
        ax.bar3d([1], [0], [0], [0.8], [0.8], [E_mid_n], color='#42B540', alpha=0.8)
        ax.bar3d([2], [0], [0], [0.8], [0.8], [E_high_n], color='#ED0000', alpha=0.8)

        ax.set_xticks([0.4, 1.4, 2.4])
        ax.set_xticklabels(['低频', '中频', '高频'])
        ax.set_zlabel('归一化能量')
        ax.set_title(titles_3d[idx])
        ax.set_zlim(0, 1)

    save_figure(fig, OUTPUT_DIR / "fig4_energy_distribution_3d.png")

    # ========== 图5: 特征雷达图对比 ==========
    print("绘制特征雷达图...")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    labels = ['低频能量', '中频能量', '高频能量', '时频熵', '样本熵', 'DC分量']
    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for res, label, color in zip(res_list, ['名义工况', '推力下降30%', 'TVC偏置5°'],
                                  [COLOR_NOMINAL, COLOR_THRUST, COLOR_TVC]):
        feats = extract_features_from_residual(res, dt=dt, spec_method='pwvd')
        feats_norm = feats / (np.max(np.abs(feats)) + 1e-12)
        values = feats_norm.tolist() + [feats_norm[0]]
        ax.plot(angles, values, 'o-', linewidth=LINE_WIDTH, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('特征雷达图对比', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    save_figure(fig, OUTPUT_DIR / "fig5_feature_radar.png")

    print(f"\n图表已保存到: {OUTPUT_DIR}")
    print("生成完成!")


if __name__ == "__main__":
    main()
