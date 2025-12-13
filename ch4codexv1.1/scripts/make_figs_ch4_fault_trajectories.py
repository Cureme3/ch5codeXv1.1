#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四章 4.1：多故障多严重度轨迹对比图生成脚本。

对 F1~F5 故障场景及不同严重度 η 值，生成：
1. 三轨迹对比图（高度/速度/下行程 vs 时间）
2. 路径约束对比图（动压/过载 vs 时间，含约束上限）
3. ECI坐标系3D轨迹图

适配4000s仿真时长（霍曼转移入轨）：
- S4点火: ~287s
- BURN1: 抬升远地点到500km
- COAST: 滑行到远地点 (~2700s)
- BURN2: 远地点圆化 (~3500s)

输出文件:
- outputs/ch4/figures/ch4_fault_trajectories/{fault_id}_eta{eta:.2f}_traj.png/pdf
- outputs/ch4/figures/ch4_fault_trajectories/constraints/{fault_id}_eta{eta:.2f}_q_constraints.png/pdf
- outputs/ch4/figures/ch4_fault_trajectories/constraints/{fault_id}_eta{eta:.2f}_n_constraints.png/pdf
- outputs/ch4/figures/ch4_fault_trajectories/3d/{fault_id}_eta{eta:.2f}_3d.png/pdf

命令行用法:
    python -m scripts.make_figs_ch4_fault_trajectories
    python -m scripts.make_figs_ch4_fault_trajectories --fault-ids F1_thrust_deg15 --etas 0.2,0.8
    python -m scripts.make_figs_ch4_fault_trajectories --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# 在导入matplotlib之前设置字体，确保中文正常显示
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from src.plots.plotting import (
    setup_matplotlib,
    save_figure,
    DEFAULT_COLORS,
    LINE_WIDTH,
    LEGEND_FONTSIZE,
)
from src.sim.viz_trajectories import (
    build_three_trajectories,
    load_three_trajectories_from_npz,
    ThreeTrajectories,
    get_default_fault_ids,
    get_default_etas,
)

# Path constraint limits (与 configs/kz1a_params.yaml 一致)
Q_MAX_KPA = 55.0  # 最大动压 [kPa]
N_MAX_G = 3.5     # 最大法向过载 [g]

# 中文标签映射
LABEL_MAP = {
    'Nominal': '名义轨迹',
    'Fault': '故障轨迹',
    'Time': '时间 t (s)',
    'Altitude_km': '高度 h (km)',
    'Velocity_kms': '速度 v (km/s)',
    'FPA': '弹道倾角 γ (°)',
    'vs_time': ' vs 时间',
    'trajectory': '轨迹',
    'comparison': '对比',
}

def get_label(key: str) -> str:
    """获取中文标签。"""
    return LABEL_MAP.get(key, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第四章多故障多严重度轨迹对比图生成"
    )
    parser.add_argument(
        "--fault-ids",
        type=str,
        default=None,
        help="逗号分隔的故障 ID 列表，默认使用 F1~F5",
    )
    parser.add_argument(
        "--etas",
        type=str,
        default="0.2,0.5,0.8",
        help="逗号分隔的 η 列表，默认 0.2,0.5,0.8",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认 outputs/ch4/figures/ch4_fault_trajectories",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：只对 F1 和单一 η=0.5 生成图",
    )
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="跳过路径约束图生成",
    )
    parser.add_argument(
        "--t-step",
        type=float,
        default=1.0,
        help="时间网格步长 [s]，默认 1.0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="并行线程数，默认 12",
    )
    parser.add_argument(
        "--no-npz",
        action="store_true",
        help="不使用npz快速加载，重新运行仿真",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="跳过3D轨迹图生成",
    )
    return parser.parse_args()


def plot_trajectory_comparison(
    traj: ThreeTrajectories,
    output_dir: Path,
) -> None:
    """生成三行子图：高度/速度/纵向距离 vs 时间。"""
    # 调整画布高度，避免子图太挤
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    t = traj.t
    fault_id = traj.fault_id
    eta = traj.eta

    # --- 子图1: 高度 (带符号 h) ---
    ax1 = axes[0]
    ax1.plot(t, traj.h_nom, color=DEFAULT_COLORS["nominal"], label="标称轨迹", linewidth=LINE_WIDTH)
    ax1.plot(t, traj.h_fault, color=DEFAULT_COLORS["fault"], linestyle="--", label="故障状态", linewidth=LINE_WIDTH)
    ax1.plot(t, traj.h_reconfig, color=DEFAULT_COLORS["replan"], linestyle="-.", label="重构轨迹", linewidth=LINE_WIDTH)
    # 标准学术格式: 物理量名称 + 斜体符号 + 正体单位
    ax1.set_ylabel(r"高度 $h$ (km)")
    ax1.legend(loc="best", frameon=True, edgecolor='black', fancybox=False)  # 图例带框
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- 子图2: 速度 (带符号 V) ---
    ax2 = axes[1]
    ax2.plot(t, traj.v_nom, color=DEFAULT_COLORS["nominal"], label="标称轨迹", linewidth=LINE_WIDTH)
    ax2.plot(t, traj.v_fault, color=DEFAULT_COLORS["fault"], linestyle="--", label="故障状态", linewidth=LINE_WIDTH)
    ax2.plot(t, traj.v_reconfig, color=DEFAULT_COLORS["replan"], linestyle="-.", label="重构轨迹", linewidth=LINE_WIDTH)
    ax2.set_ylabel(r"速度 $V$ (km/s)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- 子图3: 纵向距离 (修正术语，带符号 x) ---
    ax3 = axes[2]
    ax3.plot(t, traj.s_nom, color=DEFAULT_COLORS["nominal"], label="标称轨迹", linewidth=LINE_WIDTH)
    ax3.plot(t, traj.s_fault, color=DEFAULT_COLORS["fault"], linestyle="--", label="故障状态", linewidth=LINE_WIDTH)
    ax3.plot(t, traj.s_reconfig, color=DEFAULT_COLORS["replan"], linestyle="-.", label="重构轨迹", linewidth=LINE_WIDTH)
    ax3.set_xlabel(r"时间 $t$ (s)")       # X轴标签放在最底部
    ax3.set_ylabel(r"纵向距离 $x$ (km)")  # 修正: 下行程 -> 纵向距离
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 移除 suptitle，学术论文通常在图注中说明标题
    fig.tight_layout()

    # 保存
    filename = f"{fault_id}_eta{eta:.2f}_traj.png"
    outpath = output_dir / filename
    save_figure(fig, outpath)
    print(f"  - [中文期刊版] 轨迹图已保存: {outpath}")


def plot_dynamic_pressure_constraint(
    traj: ThreeTrajectories,
    output_dir: Path,
    q_max: float = Q_MAX_KPA,
) -> None:
    """生成动压 vs 时间对比图，含约束上限。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    t = traj.t
    fault_id = traj.fault_id
    eta = traj.eta

    ax.plot(t, traj.q_nom, color=DEFAULT_COLORS["nominal"], label="标称轨迹", linewidth=LINE_WIDTH)
    ax.plot(t, traj.q_fault, color=DEFAULT_COLORS["fault"], linestyle="--", label="故障状态", linewidth=LINE_WIDTH)
    ax.plot(t, traj.q_reconfig, color=DEFAULT_COLORS["replan"], linestyle="-.", label="重构轨迹", linewidth=LINE_WIDTH)

    # 约束上限
    ax.axhline(q_max, color="red", linestyle=":", linewidth=1.5, label=f"$q_{{\\mathrm{{max}}}}$={q_max:.1f} kPa")

    ax.set_xlabel(r"时间 $t$ (s)")
    ax.set_ylabel(r"动压 $q$ (kPa)")
    ax.legend(loc="best", frameon=True, edgecolor='black', fancybox=False)
    ax.grid(True, linestyle="--", alpha=0.5)

    filename = f"{fault_id}_eta{eta:.2f}_q_constraints.png"
    outpath = output_dir / filename
    save_figure(fig, outpath)
    print(f"  - [中文期刊版] 动压约束图已保存: {outpath}")


def plot_normal_load_constraint(
    traj: ThreeTrajectories,
    output_dir: Path,
    n_max: float = N_MAX_G,
) -> None:
    """生成法向过载 vs 时间对比图，含约束上限。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    t = traj.t
    fault_id = traj.fault_id
    eta = traj.eta

    ax.plot(t, traj.n_nom, color=DEFAULT_COLORS["nominal"], label="标称轨迹", linewidth=LINE_WIDTH)
    ax.plot(t, traj.n_fault, color=DEFAULT_COLORS["fault"], linestyle="--", label="故障状态", linewidth=LINE_WIDTH)
    ax.plot(t, traj.n_reconfig, color=DEFAULT_COLORS["replan"], linestyle="-.", label="重构轨迹", linewidth=LINE_WIDTH)

    # 约束上限
    ax.axhline(n_max, color="red", linestyle=":", linewidth=1.5, label=f"$n_{{\\mathrm{{max}}}}$={n_max:.1f} g")

    ax.set_xlabel(r"时间 $t$ (s)")
    ax.set_ylabel(r"法向过载 $n$ (g)")
    ax.legend(loc="best", frameon=True, edgecolor='black', fancybox=False)
    ax.grid(True, linestyle="--", alpha=0.5)

    filename = f"{fault_id}_eta{eta:.2f}_n_constraints.png"
    outpath = output_dir / filename
    save_figure(fig, outpath)
    print(f"  - [中文期刊版] 过载约束图已保存: {outpath}")


def plot_eci_3d_trajectory(
    traj: ThreeTrajectories,
    output_dir: Path,
) -> None:
    """生成ECI坐标系下的3D轨迹图。

    展示名义轨迹、故障轨迹、重构轨迹在ECI坐标系下的三维空间分布。
    包含地球参考球、目标轨��（500km圆轨道/200x400km椭圆轨道）。
    """
    from mpl_toolkits.mplot3d import Axes3D

    # 检查是否有ECI数据
    if traj.r_eci_nom is None:
        print(f"  - [WARN] 无ECI数据，跳过3D轨迹图")
        return

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    fault_id = traj.fault_id
    eta = traj.eta
    domain = traj.mission_domain

    # 地球半径 (km)
    R_EARTH_KM = 6371.0

    # 转换为km
    r_nom = traj.r_eci_nom / 1000.0  # m -> km
    r_fault = traj.r_eci_fault / 1000.0
    r_reconfig = traj.r_eci_reconfig / 1000.0

    # 绘制地球参考球（简化为线框）
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_earth = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    y_earth = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    z_earth = R_EARTH_KM * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3, linewidth=0.5)

    # 绘制轨迹（降采样以提高性能）
    step = max(1, len(traj.t) // 500)

    ax.plot(r_nom[::step, 0], r_nom[::step, 1], r_nom[::step, 2],
            color=DEFAULT_COLORS["nominal"], linewidth=2.0, label="标称轨迹(500km圆轨道)")
    ax.plot(r_fault[::step, 0], r_fault[::step, 1], r_fault[::step, 2],
            color=DEFAULT_COLORS["fault"], linestyle="--", linewidth=2.0, label="故障开环")
    ax.plot(r_reconfig[::step, 0], r_reconfig[::step, 1], r_reconfig[::step, 2],
            color=DEFAULT_COLORS["replan"], linestyle="-.", linewidth=2.0, label=f"重构轨迹({domain.name if domain else 'N/A'})")

    # 标记起点和终点
    ax.scatter(*r_nom[0], color='green', s=150, marker='o', label='发射点', zorder=10)
    ax.scatter(*r_nom[-1], color=DEFAULT_COLORS["nominal"], s=200, marker='*', label='标称入轨点', zorder=10)
    ax.scatter(*r_reconfig[-1], color=DEFAULT_COLORS["replan"], s=150, marker='^', label='重构入轨点', zorder=10)

    # 绘制目标轨道参考圆
    theta = np.linspace(0, 2 * np.pi, 100)

    # 500km圆轨道（RETAIN目标）
    r_500km = R_EARTH_KM + 500
    ax.plot(r_500km * np.cos(theta), r_500km * np.sin(theta), np.zeros_like(theta),
            'g-', alpha=0.6, linewidth=2, label='500km圆轨道(RETAIN)')

    # 200x400km椭圆轨道（DEGRADED目标）- 简化为近地点和远地点圆
    r_200km = R_EARTH_KM + 200
    r_400km = R_EARTH_KM + 400
    ax.plot(r_200km * np.cos(theta), r_200km * np.sin(theta), np.zeros_like(theta),
            'm--', alpha=0.4, linewidth=1.5, label='200km(DEGRADED近地点)')
    ax.plot(r_400km * np.cos(theta), r_400km * np.sin(theta), np.zeros_like(theta),
            'm:', alpha=0.4, linewidth=1.5, label='400km(DEGRADED远地点)')

    # 增大字体
    ax.set_xlabel('X (km)', fontsize=14, labelpad=10)
    ax.set_ylabel('Y (km)', fontsize=14, labelpad=10)
    ax.set_zlabel('Z (km)', fontsize=14, labelpad=10)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(f'{fault_id} (η={eta:.2f}) - ECI 3D轨迹\n任务域: {domain.name if domain else "N/A"}', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # 设置等比例显示
    max_range = np.max([
        np.max(np.abs(r_nom)),
        np.max(np.abs(r_fault)),
        np.max(np.abs(r_reconfig))
    ]) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # 调整视角
    ax.view_init(elev=25, azim=-50)

    plt.tight_layout()

    filename = f"{fault_id}_eta{eta:.2f}_3d.png"
    outpath = output_dir / filename
    save_figure(fig, outpath)
    print(f"  - [中文期刊版] ECI 3D轨迹图已保存: {outpath}")


def _process_single_figure(args):
    """处理单个故障+eta组合的图片生成（用于并行）。"""
    fault_id, eta, output_dir, constraints_dir, eci_3d_dir, generate_constraints, generate_3d, use_npz = args

    try:
        # 从npz快速加载或重新仿真
        if use_npz:
            traj = load_three_trajectories_from_npz(fault_id=fault_id, eta=eta)
        else:
            traj = build_three_trajectories(fault_id=fault_id, eta=eta, t_step=1.0)

        # Trajectory comparison plot
        plot_trajectory_comparison(traj, output_dir)

        # Constraint plots
        if generate_constraints:
            plot_dynamic_pressure_constraint(traj, constraints_dir)
            plot_normal_load_constraint(traj, constraints_dir)

        # ECI 3D trajectory plot
        if generate_3d:
            plot_eci_3d_trajectory(traj, eci_3d_dir)

        return (fault_id, eta, True, None)
    except Exception as e:
        import traceback
        return (fault_id, eta, False, str(e) + "\n" + traceback.format_exc())


def generate_fault_trajectory_figs(
    fault_ids: List[str],
    etas: List[float],
    output_dir: Path,
    t_step: float = 1.0,
    generate_constraints: bool = True,
    generate_3d: bool = True,
    workers: int = 12,
    use_npz: bool = True,
) -> None:
    """生成多故障多严重度轨迹对比图（支持多线程并行）。

    Parameters
    ----------
    fault_ids : List[str]
        故障场景 ID 列表
    etas : List[float]
        故障严重度列表
    output_dir : Path
        输出目录
    t_step : float
        时间网格步长
    generate_constraints : bool
        是否生成路径约束图
    generate_3d : bool
        是否生成ECI 3D轨迹图
    workers : int
        并行线程数，默认12
    use_npz : bool
        是否从npz文件快速加载数据，默认True
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir = output_dir / "constraints"
    if generate_constraints:
        constraints_dir.mkdir(parents=True, exist_ok=True)
    eci_3d_dir = output_dir / "3d"
    if generate_3d:
        eci_3d_dir.mkdir(parents=True, exist_ok=True)

    # 构建任务列表
    tasks = []
    for fault_id in fault_ids:
        for eta in etas:
            tasks.append((fault_id, eta, output_dir, constraints_dir, eci_3d_dir,
                         generate_constraints, generate_3d, use_npz))

    total = len(tasks)
    print(f"总任务数: {total}, 并行线程: {workers}, 数据源: {'npz快速加载' if use_npz else '重新仿真'}")

    # 并行执行
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_single_figure, task): task for task in tasks}

        for future in as_completed(futures):
            completed += 1
            fault_id, eta, success, error = future.result()
            if success:
                print(f"[{completed}/{total}] {fault_id} η={eta:.1f} - 完成")
            else:
                print(f"[{completed}/{total}] {fault_id} η={eta:.1f} - 失败: {error[:80]}...")


def generate_path_constraint_figs(
    fault_ids: List[str],
    eta: float,
    output_dir: Path,
    t_step: float = 1.0,
) -> None:
    """为代表性工况生成路径约束对比图（每种故障选一个重度 η）。

    Parameters
    ----------
    fault_ids : List[str]
        故障场景 ID 列表
    eta : float
        故障严重度（建议用高值如 0.8）
    output_dir : Path
        输出目录
    t_step : float
        时间网格步长
    """
    constraints_dir = output_dir / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)

    for fault_id in fault_ids:
        print(f"\n生成路径约束图: {fault_id}, η={eta:.2f} ...")
        try:
            traj = build_three_trajectories(
                fault_id=fault_id,
                eta=eta,
                t_step=t_step,
            )
            plot_dynamic_pressure_constraint(traj, constraints_dir)
            plot_normal_load_constraint(traj, constraints_dir)
        except Exception as e:
            print(f"  [WARN] 生成失败: {e}")
            continue


def generate_combined_2x2_figure(
    fault_id: str,
    etas: List[float],
    output_dir: Path,
    t_step: float = 1.0,
) -> None:
    """为指定故障场景生成2x2组合图。

    子图布局:
    - (0,0) 高度 vs 时间
    - (0,1) 速度 vs 时间
    - (1,0) 弹道倾角 vs 时间
    - (1,1) 高度-速度相图
    """
    from src.sim.run_nominal import simulate_full_mission
    from src.sim.run_fault import simulate_fault_open_loop
    from src.sim.scenarios import get_scenario, scale_scenario_by_eta

    print(f"\n--- 生成 {fault_id} 2x2组合图 ---")

    # Run nominal simulation
    print("  运行名义轨迹仿真...")
    try:
        nominal = simulate_full_mission(dt=t_step, save_csv=False)
        t_nom = np.array(nominal.time)
        h_nom = np.array(nominal.altitude_km)
        v_nom = np.array(nominal.speed_kms)
        gamma_nom = np.array(nominal.flight_path_deg)
    except Exception as e:
        print(f"  名义仿真出错: {e}")
        return

    # Run fault simulations at different eta
    fault_results = {}
    for eta in etas:
        print(f"  运行故障仿真 (η={eta})...")
        try:
            base_scenario = get_scenario(fault_id)
            scenario = scale_scenario_by_eta(base_scenario, eta)
            fault_sim = simulate_fault_open_loop(scenario, dt=t_step)
            fault_results[eta] = {
                't': np.array(fault_sim.time),
                'h': np.array(fault_sim.altitude_km),
                'v': np.array(fault_sim.speed_kms),
                'gamma': np.array(fault_sim.flight_path_deg),
            }
        except Exception as e:
            print(f"  故障仿真出错 (η={eta}): {e}")
            continue

    if not fault_results:
        print(f"  {fault_id} 没有完成任何故障仿真")
        return

    # Determine time range
    t_max = min(
        t_nom[-1] if len(t_nom) > 0 else 300,
        min(r['t'][-1] for r in fault_results.values() if len(r['t']) > 0)
    )
    t_max = min(t_max, 400)  # Cap at 400s

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(etas)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    mask_nom = t_nom <= t_max

    # (0,0) 高度 vs 时间
    ax = axes[0, 0]
    ax.plot(t_nom[mask_nom], h_nom[mask_nom], 'b-', linewidth=2, label=get_label('Nominal'))
    for (eta, result), color in zip(fault_results.items(), colors):
        mask = result['t'] <= t_max
        ax.plot(result['t'][mask], result['h'][mask], '-', color=color,
                linewidth=1.5, label=f'{get_label("Fault")} (η={eta})')
    ax.set_xlabel(get_label('Time'))
    ax.set_ylabel(get_label('Altitude_km'))
    ax.set_title(f'高度{get_label("vs_time")}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)

    # (0,1) 速度 vs 时间
    ax = axes[0, 1]
    ax.plot(t_nom[mask_nom], v_nom[mask_nom], 'b-', linewidth=2, label=get_label('Nominal'))
    for (eta, result), color in zip(fault_results.items(), colors):
        mask = result['t'] <= t_max
        ax.plot(result['t'][mask], result['v'][mask], '-', color=color,
                linewidth=1.5, label=f'{get_label("Fault")} (η={eta})')
    ax.set_xlabel(get_label('Time'))
    ax.set_ylabel(get_label('Velocity_kms'))
    ax.set_title(f'速度{get_label("vs_time")}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)

    # (1,0) 弹道倾角 vs 时间
    ax = axes[1, 0]
    ax.plot(t_nom[mask_nom], gamma_nom[mask_nom], 'b-', linewidth=2, label=get_label('Nominal'))
    for (eta, result), color in zip(fault_results.items(), colors):
        mask = result['t'] <= t_max
        ax.plot(result['t'][mask], result['gamma'][mask], '-', color=color,
                linewidth=1.5, label=f'{get_label("Fault")} (η={eta})')
    ax.set_xlabel(get_label('Time'))
    ax.set_ylabel(get_label('FPA'))
    ax.set_title(f'弹道倾角{get_label("vs_time")}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)

    # (1,1) 高度-速度相图
    ax = axes[1, 1]
    ax.plot(v_nom[mask_nom], h_nom[mask_nom], 'b-', linewidth=2, label=get_label('Nominal'))
    for (eta, result), color in zip(fault_results.items(), colors):
        mask = result['t'] <= t_max
        ax.plot(result['v'][mask], result['h'][mask], '-', color=color,
                linewidth=1.5, label=f'{get_label("Fault")} (η={eta})')
    ax.set_xlabel(get_label('Velocity_kms'))
    ax.set_ylabel(get_label('Altitude_km'))
    ax.set_title('高度-速度相图')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{fault_id}: 不同故障严重度下的{get_label("trajectory")}{get_label("comparison")}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{fault_id}_combined_2x2.png"
    outpath = combined_dir / filename
    save_figure(fig, outpath)
    print(f"  - 2x2组合图已保存: {outpath}")


def main() -> None:
    """主函数。"""
    args = parse_args()

    # 使用出版级中文样式
    setup_matplotlib()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "ch4" / "figures" / "ch4_fault_trajectories"

    # Determine fault IDs
    if args.quick:
        fault_ids = ["F1_thrust_deg15"]
        etas = [0.5]
        print("=" * 80)
        print("第四章：多故障轨迹对比图生成 (Quick Mode)")
        print("=" * 80)
    elif args.fault_ids:
        fault_ids = [s.strip() for s in args.fault_ids.split(",")]
        etas = [float(e.strip()) for e in args.etas.split(",")]
        print("=" * 80)
        print("第四章：多故障轨迹对比图生成")
        print("=" * 80)
    else:
        fault_ids = get_default_fault_ids()
        etas = [float(e.strip()) for e in args.etas.split(",")]
        print("=" * 80)
        print("第四章：多故障轨迹对比图生成 (Full Mode)")
        print("=" * 80)

    print(f"故障场景: {fault_ids}")
    print(f"严重度 η: {etas}")
    print(f"输出目录: {output_dir}")
    print()

    print(f"并行线程: {args.workers}")
    print(f"数据源: {'重新仿真' if args.no_npz else 'npz快速加载'}")
    print()

    generate_fault_trajectory_figs(
        fault_ids=fault_ids,
        etas=etas,
        output_dir=output_dir,
        t_step=args.t_step,
        generate_constraints=not args.no_constraints,
        generate_3d=not args.no_3d,
        workers=args.workers,
        use_npz=not args.no_npz,
    )

    # 如果非快速模式，也生成2x2组合图
    if not args.quick:
        print("\n生成2x2组合轨迹图...")
        for fault_id in fault_ids:
            generate_combined_2x2_figure(
                fault_id=fault_id,
                etas=[0.2, 0.5, 0.8, 1.0],
                output_dir=output_dir,
                t_step=args.t_step,
            )

    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
