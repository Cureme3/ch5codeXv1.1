#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第5章可视化脚本: 故障诊断与轨迹规划流程图。"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CH4_ROOT = PROJECT_ROOT / "ch4codexv1.1"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ch3codev1.1"))
# 注意顺序：先插入 src，再插入 ch4codexv1.1，这样 ch4codexv1.1 在前面
# 因为 insert(0, ...) 会把新路径放在最前面
sys.path.insert(0, str(CH4_ROOT / "src"))
sys.path.insert(0, str(CH4_ROOT))

from plots.plotting import setup_matplotlib, save_figure, DEFAULT_COLORS, LINE_WIDTH
from diagnosis.eso import run_eso
from diagnosis.features import extract_features_from_residual
from src.sim.run_fault import run_fault_scenario
from src.sim.mission_domains import choose_initial_domain, default_domain_config
from src.learn.weights import compute_adaptive_penalties

# 应用出版级绘图样式
setup_matplotlib()

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "ch5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

R_EARTH = 6.378137e6

# 配色
COLOR_NOMINAL = DEFAULT_COLORS["nominal"]
COLOR_FAULT = DEFAULT_COLORS["fault"]
COLOR_REPLAN = DEFAULT_COLORS["replan"]


def plot_adaptive_weights():
    """绘制自适应权重曲线。"""
    etas = np.linspace(0, 1, 50)
    terminal_weights = []
    slack_weights = []

    for eta in etas:
        weights = compute_adaptive_penalties(eta)
        terminal_weights.append(weights.terminal_state_dev)
        slack_weights.append(weights.q_slack)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(etas, terminal_weights, color=COLOR_NOMINAL, linewidth=LINE_WIDTH, label='终端权重')
    ax1.axvline(x=0.3, color=COLOR_REPLAN, linestyle='--', alpha=0.7)
    ax1.axvline(x=0.7, color=COLOR_FAULT, linestyle='--', alpha=0.7)
    ax1.set_xlabel('故障严重度 $\\eta$')
    ax1.set_ylabel('权重值')
    ax1.set_title('终端状态权重 vs 故障严重度')
    ax1.legend()

    ax2.plot(etas, slack_weights, color=COLOR_FAULT, linewidth=LINE_WIDTH, label='松弛权重')
    ax2.axvline(x=0.3, color=COLOR_REPLAN, linestyle='--', alpha=0.7)
    ax2.axvline(x=0.7, color=COLOR_FAULT, linestyle='--', alpha=0.7)
    ax2.set_xlabel('故障严重度 $\\eta$')
    ax2.set_ylabel('权重值')
    ax2.set_title('约束松弛权重 vs 故障严重度')
    ax2.legend()

    return fig


def plot_mission_domains():
    """绘制任务域划分图。"""
    fig, ax = plt.subplots()

    ax.axvspan(0, 0.3, alpha=0.25, color=COLOR_REPLAN, label='RETAIN (500km)')
    ax.axvspan(0.3, 0.7, alpha=0.25, color='#FFC107', label='DEGRADED (300km)')
    ax.axvspan(0.7, 1.0, alpha=0.25, color=COLOR_FAULT, label='SAFE_AREA (着陆)')

    etas = np.linspace(0, 1, 100)
    targets = []
    for eta in etas:
        domain = choose_initial_domain(eta)
        cfg = default_domain_config(domain)
        targets.append(cfg.terminal_target.target_altitude_km)

    ax.plot(etas, targets, color='black', linewidth=LINE_WIDTH, label='目标高度')
    ax.set_xlabel('故障严重度 $\\eta$')
    ax.set_ylabel('目标高度 (km)')
    ax.set_title('任务域选择策略')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(-50, 550)

    return fig


def plot_comparison_trajectories():
    """绘制故障轨迹与重规划轨迹对比。使用 ch4 的完整 SCvx 规划器。"""
    import os
    from src.sim.run_nominal import simulate_full_mission
    from src.sim.run_fault import plan_recovery_segment_scvx

    # 切换到 ch4codexv1.1 目录以找到配置文件
    old_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT / "ch4codexv1.1")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    scenarios = [
        ("F1_thrust_deg15", 0.2),
        ("F1_thrust_deg15", 0.5),
        ("F1_thrust_deg15", 0.8),
    ]

    # 名义轨迹作为参考
    nominal = simulate_full_mission(dt=1.0)
    nom_states = np.asarray(nominal.states)
    nom_time = np.asarray(nominal.time)
    nom_h = (np.linalg.norm(nom_states[:, 0:3], axis=1) - R_EARTH) / 1000.0
    nom_v = np.linalg.norm(nom_states[:, 3:6], axis=1) / 1000.0

    for idx, (scenario_id, eta) in enumerate(scenarios):
        # 故障轨迹
        fault_sim = run_fault_scenario(scenario_id, eta=eta, dt=1.0)
        fault_states = fault_sim.states
        fault_time = fault_sim.time
        fault_h = (np.linalg.norm(fault_states[:, 0:3], axis=1) - R_EARTH) / 1000.0
        fault_v = np.linalg.norm(fault_states[:, 3:6], axis=1) / 1000.0

        domain = choose_initial_domain(eta)
        cfg = default_domain_config(domain)
        target_h = cfg.terminal_target.target_altitude_km

        # 使用 ch4 的完整 SCvx 重规划
        try:
            recovery = plan_recovery_segment_scvx(
                scenario=fault_sim.scenario,
                fault_sim=fault_sim,
                nominal=nominal,
                nodes=40,
                fault_eta=eta,
                use_adaptive_penalties=True,
                solver_profile="fast",
                mission_domain=domain,
                enable_domain_escalation=True,
            )
            replan_states = recovery.states
            replan_time = recovery.time
            replan_h = (np.linalg.norm(replan_states[:, 0:3], axis=1) - R_EARTH) / 1000.0
            replan_v = np.linalg.norm(replan_states[:, 3:6], axis=1) / 1000.0
            replan_ok = True
        except Exception as e:
            print(f"[WARN] SCvx 重规划失败 ({scenario_id}, eta={eta}): {e}")
            replan_time = np.array([0])
            replan_h = np.array([0])
            replan_v = np.array([0])
            replan_ok = False

        # 绘制高度
        axes[0, idx].plot(nom_time[:len(nom_h)], nom_h, color=COLOR_NOMINAL,
                         linewidth=LINE_WIDTH, alpha=0.5, label='名义')
        axes[0, idx].plot(fault_time, fault_h, color=COLOR_FAULT,
                         linewidth=LINE_WIDTH, label='故障')
        if replan_ok:
            axes[0, idx].plot(replan_time, replan_h, color=COLOR_REPLAN,
                             linewidth=LINE_WIDTH, linestyle='--', label='重规划')
        axes[0, idx].axhline(y=target_h, color='gray', linestyle=':',
                            linewidth=1, label=f'目标: {target_h}km')
        axes[0, idx].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, idx].set_xlabel('时间 (s)')
        axes[0, idx].set_ylabel('高度 (km)')
        axes[0, idx].set_title(f'$\\eta$={eta}, 域={domain.name}')
        axes[0, idx].legend(fontsize=9)
        axes[0, idx].set_ylim(-50, 600)

        # 绘制速度
        axes[1, idx].plot(nom_time[:len(nom_v)], nom_v, color=COLOR_NOMINAL,
                         linewidth=LINE_WIDTH, alpha=0.5, label='名义')
        axes[1, idx].plot(fault_time, fault_v, color=COLOR_FAULT,
                         linewidth=LINE_WIDTH, label='故障')
        if replan_ok:
            axes[1, idx].plot(replan_time, replan_v, color=COLOR_REPLAN,
                             linewidth=LINE_WIDTH, linestyle='--', label='重规划')
        axes[1, idx].set_xlabel('时间 (s)')
        axes[1, idx].set_ylabel('速度 (km/s)')
        axes[1, idx].legend(fontsize=9)

    # 恢复工作目录
    os.chdir(old_cwd)
    return fig


def plot_eso_residual():
    """绘制ESO残差对比：名义 vs 故障。"""
    from src.sim.run_nominal import simulate_nominal

    # 名义轨迹
    nom_sim = simulate_nominal(dt=1.0)
    nom_states = nom_sim.states

    # 故障轨迹
    fault_sim = run_fault_scenario("F1_thrust_deg15", eta=0.5, dt=1.0)
    fault_states = fault_sim.states

    def get_residual(states):
        v = states[:, 3:6]
        ax_acc = np.diff(v[:, 0])
        az_acc = np.diff(v[:, 2])
        axz = np.column_stack([ax_acc, az_acc])
        residuals, _, _ = run_eso(axz, dt=1.0)
        return np.linalg.norm(residuals, axis=1)

    min_len = min(len(nom_states), len(fault_states)) - 1
    nom_res = get_residual(nom_states[:min_len+1])[:min_len]
    fault_res = get_residual(fault_states[:min_len+1])[:min_len]
    time = np.arange(min_len)

    fig, ax = plt.subplots()
    ax.plot(time, nom_res, color=COLOR_NOMINAL, linewidth=LINE_WIDTH, label='名义工况')
    ax.plot(time, fault_res, color=COLOR_FAULT, linewidth=LINE_WIDTH, label='故障工况 ($\\eta$=0.5)')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('残差幅值')
    ax.set_title('ESO残差对比: 名义 vs 推力下降')
    ax.legend()

    return fig


def main():
    print("生成第5章可视化图表...")

    print("[1/4] 自适应权重曲线...")
    fig = plot_adaptive_weights()
    save_figure(fig, OUTPUT_DIR / "fig5_1_adaptive_weights.png")

    print("[2/4] 任务域划分...")
    fig = plot_mission_domains()
    save_figure(fig, OUTPUT_DIR / "fig5_2_mission_domains.png")

    print("[3/4] 轨迹对比...")
    fig = plot_comparison_trajectories()
    save_figure(fig, OUTPUT_DIR / "fig5_3_trajectory_comparison.png")

    print("[4/4] ESO残差...")
    fig = plot_eso_residual()
    save_figure(fig, OUTPUT_DIR / "fig5_4_eso_residual.png")

    print(f"\n图表已保存到: {OUTPUT_DIR}")
    print("生成完成!")


if __name__ == "__main__":
    main()
