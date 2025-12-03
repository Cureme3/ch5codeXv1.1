#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图4-22, 4-23, 4-24：任务域相关图表。

从真实轨迹数据生成：
- 图4-22：retain 域下若干故障的关键量响应（从真实 replan 结果中选取）
- 图4-23：选择一个代表性故障（F1 severe，η=0.8），对比不同任务域下终端行距或轨迹差异
- 图4-24：五种严重故障的终端行距统计汇总

输出文件：
- outputs/figures/ch4_mission_domains/fig4_22_retain_domain_responses.png/.pdf
- outputs/figures/ch4_mission_domains/fig4_23_f1_severe_downrange.png/.pdf
- outputs/figures/ch4_mission_domains/fig4_24_severe_downrange_summary.png/.pdf

命令行用法：
    python -m scripts.make_figs_ch4_mission_domains
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from src.plots.plotting import setup_matplotlib  # noqa: E402

# 输出目录
OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "ch4_mission_domains"
DATA_DIR = PROJECT_ROOT / "outputs" / "data" / "ch4_trajectories_replan"

# 故障类型
FAULT_IDS = ["F1", "F2", "F3", "F4", "F5"]
FAULT_NAMES = {
    "F1": "推力降级",
    "F2": "TVC速率限制",
    "F3": "TVC卡滞",
    "F4": "传感器偏置",
    "F5": "事件延迟",
}

# eta 值
ETA_VALUES = [0.2, 0.5, 0.8]

# 任务域颜色
DOMAIN_COLORS = {
    "RETAIN": "#2ecc71",      # 绿色
    "DEGRADED": "#f39c12",    # 橙色
    "SAFE_AREA": "#e74c3c",   # 红色
    "retain": "#2ecc71",
    "degraded": "#f39c12",
    "safe_area": "#e74c3c",
}


@dataclass
class TrajectoryData:
    """轨迹数据。"""
    t: np.ndarray
    downrange: np.ndarray
    altitude: np.ndarray
    label: str = ""
    eta: float = 0.0
    mission_domain: str = ""
    fault_type: str = ""
    t_fault: float = 0.0
    t_confirm: float = 0.0


def save_figure(fig: plt.Figure, filepath: Path) -> None:
    """保存图像到 PNG 和 PDF。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, bbox_inches='tight')
    fig.savefig(filepath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def load_trajectory_npz(file_path: Path) -> Optional[TrajectoryData]:
    """加载轨迹数据。"""
    if not file_path.exists():
        return None

    data = np.load(file_path, allow_pickle=True)
    t = data.get('t', np.array([]))
    downrange = data.get('downrange', np.array([]))
    altitude = data.get('altitude', np.array([]))

    if len(t) == 0 or len(downrange) == 0 or len(altitude) == 0:
        return None

    return TrajectoryData(
        t=t,
        downrange=downrange,
        altitude=altitude,
        label=str(data.get('label', '')),
        eta=float(data.get('eta', 0.0)),
        mission_domain=str(data.get('mission_domain', '')),
        fault_type=str(data.get('fault_type', '')),
        t_fault=float(data.get('t_fault', 0.0)),
        t_confirm=float(data.get('t_confirm', 0.0)),
    )


def determine_mission_domain(eta: float) -> str:
    """根据 eta 值判定任务域。

    规则：
    - eta < 0.3: RETAIN
    - 0.3 <= eta < 0.7: DEGRADED
    - eta >= 0.7: SAFE_AREA
    """
    if eta < 0.3:
        return "RETAIN"
    elif eta < 0.7:
        return "DEGRADED"
    else:
        return "SAFE_AREA"


def load_all_trajectories() -> Dict[str, Dict[float, Tuple[Optional[TrajectoryData], Optional[TrajectoryData]]]]:
    """加载所有轨迹数据。

    Returns:
        {fault_id: {eta: (openloop, replan)}}
    """
    all_data = {}

    for fault_id in FAULT_IDS:
        all_data[fault_id] = {}
        for eta in ETA_VALUES:
            eta_str = f"eta{eta:.1f}".replace(".", "")
            openloop_path = DATA_DIR / f"{fault_id}_{eta_str}_openloop.npz"
            replan_path = DATA_DIR / f"{fault_id}_{eta_str}_replan.npz"

            openloop = load_trajectory_npz(openloop_path)
            replan = load_trajectory_npz(replan_path)

            all_data[fault_id][eta] = (openloop, replan)

    # 加载名义轨迹
    nominal_path = DATA_DIR / "nominal.npz"
    nominal = load_trajectory_npz(nominal_path)
    all_data["nominal"] = {0.0: (nominal, nominal)}

    return all_data


def plot_fig4_22_retain_domain_responses(all_data: Dict, out_dir: Path) -> None:
    """图4-22：轻度故障(η=0.2, RETAIN域)下五种典型故障的关键飞行量响应。

    使用真实 replan 结果，展示高度和行距曲线。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 获取名义轨迹
    nominal = all_data.get("nominal", {}).get(0.0, (None, None))[0]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    eta = 0.2  # RETAIN 域对应轻度故障

    # 子图1：高度对比
    ax = axes[0]
    if nominal is not None:
        ax.plot(nominal.t, nominal.altitude, 'k--', linewidth=2, label='名义')
    for i, fid in enumerate(FAULT_IDS):
        _, replan = all_data.get(fid, {}).get(eta, (None, None))
        if replan is not None:
            ax.plot(replan.t, replan.altitude, color=colors[i], linewidth=1.5, label=fid)
    ax.set_xlabel('时间 / s')
    ax.set_ylabel('高度 / km')
    ax.set_title('(a) 高度响应')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 子图2：行距对比
    ax = axes[1]
    if nominal is not None:
        ax.plot(nominal.t, nominal.downrange, 'k--', linewidth=2, label='名义')
    for i, fid in enumerate(FAULT_IDS):
        _, replan = all_data.get(fid, {}).get(eta, (None, None))
        if replan is not None:
            ax.plot(replan.t, replan.downrange, color=colors[i], linewidth=1.5, label=fid)
    ax.set_xlabel('时间 / s')
    ax.set_ylabel('行距 / km')
    ax.set_title('(b) 行距')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 子图3-7：各故障的高度轨迹（不同eta）
    for idx, fid in enumerate(FAULT_IDS):
        if idx >= 4:  # 只画4个子图
            break
        ax = axes[idx + 2]

        if nominal is not None:
            ax.plot(nominal.t, nominal.altitude, 'k--', linewidth=1.5, label='名义', alpha=0.7)

        for e_idx, e in enumerate(ETA_VALUES):
            _, replan = all_data.get(fid, {}).get(e, (None, None))
            if replan is not None:
                domain = determine_mission_domain(e)
                color = DOMAIN_COLORS.get(domain, colors[e_idx])
                linestyle = ['-', '--', '-.'][e_idx]
                ax.plot(replan.t, replan.altitude, color=color, linestyle=linestyle,
                        linewidth=1.5, label=f'η={e}')

        ax.set_xlabel('时间 / s')
        ax.set_ylabel('高度 / km')
        ax.set_title(f'({chr(99+idx)}) {fid}: {FAULT_NAMES[fid]}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    axes[5].axis('off')

    fig.suptitle('RETAIN 域下五种典型故障的关键飞行量响应', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, out_dir / "fig4_22_retain_domain_responses.png")


def plot_fig4_23_f1_severe_downrange(all_data: Dict, out_dir: Path) -> None:
    """图4-23：F1 severe (η=0.8) 故障下三种任务域策略的行距/高度对比。

    由于实际数据中每个 eta 对应一个任务域，这里展示三个不同 eta 值下的轨迹，
    它们分别对应 RETAIN, DEGRADED, SAFE_AREA 三种任务域策略。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 获取名义轨迹
    nominal = all_data.get("nominal", {}).get(0.0, (None, None))[0]

    # 左图：行距对比
    ax = axes[0]
    if nominal is not None:
        ax.plot(nominal.t, nominal.downrange, 'k--', linewidth=2, label='名义')

    # 绘制 F1 故障在三个 eta 下的 replan 轨迹
    for eta in ETA_VALUES:
        _, replan = all_data.get("F1", {}).get(eta, (None, None))
        if replan is not None:
            domain = determine_mission_domain(eta)
            color = DOMAIN_COLORS.get(domain, '#3498db')
            domain_label = {"RETAIN": "RETAIN（保持入轨）",
                           "DEGRADED": "DEGRADED（降级任务）",
                           "SAFE_AREA": "SAFE_AREA（安全区域）"}.get(domain, domain)
            ax.plot(replan.t, replan.downrange, color=color,
                    linewidth=2, label=f'{domain_label} (η={eta})')

    ax.set_xlabel('飞行时间 / s', fontsize=12)
    ax.set_ylabel('行距 / km', fontsize=12)
    ax.set_title('(a) F1 故障下不同任务域策略的行距对比', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 右图：高度对比
    ax = axes[1]
    if nominal is not None:
        ax.plot(nominal.t, nominal.altitude, 'k--', linewidth=2, label='名义')

    for eta in ETA_VALUES:
        _, replan = all_data.get("F1", {}).get(eta, (None, None))
        if replan is not None:
            domain = determine_mission_domain(eta)
            color = DOMAIN_COLORS.get(domain, '#3498db')
            ax.plot(replan.t, replan.altitude, color=color,
                    linewidth=2, label=f'{domain} (η={eta})')

    ax.set_xlabel('飞行时间 / s', fontsize=12)
    ax.set_ylabel('高度 / km', fontsize=12)
    ax.set_title('(b) F1 故障下不同任务域策略的高度对比', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('F1 Severe 故障下三种任务域的轨迹对比', fontsize=14)
    plt.tight_layout()
    save_figure(fig, out_dir / "fig4_23_f1_severe_downrange.png")


def plot_fig4_24_severe_downrange_summary(all_data: Dict, out_dir: Path) -> None:
    """图4-24：五种严重故障(η=0.8)终端行距/高度汇总对比。

    统计五种故障在不同 eta（对应不同任务域）下的终端状态。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 增大图表尺寸以容纳标签
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 收集数据
    terminal_downrange = {eta: [] for eta in ETA_VALUES}
    terminal_altitude = {eta: [] for eta in ETA_VALUES}

    for fault_id in FAULT_IDS:
        for eta in ETA_VALUES:
            _, replan = all_data.get(fault_id, {}).get(eta, (None, None))
            if replan is not None and len(replan.downrange) > 0:
                terminal_downrange[eta].append(replan.downrange[-1])  # km
                terminal_altitude[eta].append(replan.altitude[-1])  # km
            else:
                terminal_downrange[eta].append(np.nan)
                terminal_altitude[eta].append(np.nan)

    # 左图：终端行距柱状图
    ax = axes[0]
    x = np.arange(len(FAULT_IDS))
    width = 0.25

    # 计算Y轴范围以容纳标签
    max_downrange = max(max(terminal_downrange[eta]) for eta in ETA_VALUES if terminal_downrange[eta])

    for i, eta in enumerate(ETA_VALUES):
        domain = determine_mission_domain(eta)
        color = DOMAIN_COLORS.get(domain, '#3498db')
        values = terminal_downrange[eta]
        bars = ax.bar(x + (i - 1) * width, values, width,
                     label=f'{domain} (η={eta})', color=color, alpha=0.8)

        # 添加数值标签
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_downrange * 0.02,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('故障场景', fontsize=12)
    ax.set_ylabel('终端行距 / km', fontsize=12)
    ax.set_title('(a) 五种故障在不同任务域下的终端行距', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(FAULT_IDS)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    # 扩展Y轴上限以容纳标签
    ax.set_ylim(0, max_downrange * 1.15)

    # 右图：终端高度柱状图
    ax = axes[1]

    # 计算Y轴范围以容纳标签
    max_altitude = max(max(terminal_altitude[eta]) for eta in ETA_VALUES if terminal_altitude[eta])

    for i, eta in enumerate(ETA_VALUES):
        domain = determine_mission_domain(eta)
        color = DOMAIN_COLORS.get(domain, '#3498db')
        values = terminal_altitude[eta]
        bars = ax.bar(x + (i - 1) * width, values, width,
                     label=f'{domain} (η={eta})', color=color, alpha=0.8)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_altitude * 0.01,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('故障场景', fontsize=12)
    ax.set_ylabel('终端高度 / km', fontsize=12)
    ax.set_title('(b) 五种故障在不同任务域下的终端高度', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(FAULT_IDS)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    # 扩展Y轴上限以容纳标签
    ax.set_ylim(0, max_altitude * 1.12)

    fig.suptitle('五种严重故障在不同任务域下的终端状态对比', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, out_dir / "fig4_24_severe_downrange_summary.png")


def make_3d_trajectory_plot(
    fault_key: str,
    all_data: Dict,
    out_dir: Path,
) -> None:
    """生成单个故障场景的3D轨迹图。

    使用 (时间, 地面行距, 高度) 三维坐标系展示轨迹。
    包含：名义轨迹、故障开环轨迹、重规划轨迹、轨道高度参考线。
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 轨道高度参考值 (km)
    ORBIT_HEIGHTS = {
        "RETAIN": 500,      # 保持任务轨道高度
        "DEGRADED": 300,    # 降级任务轨道高度
        "SAFE_AREA": 0,     # 安全区（地面）
    }

    # 加载名义轨迹
    nominal = all_data.get("nominal", {}).get(0.0, (None, None))[0]
    if nominal is not None:
        ax.plot(
            nominal.t, nominal.downrange, nominal.altitude,
            'b-', linewidth=2, label="名义轨迹", alpha=0.8
        )
        ax.scatter(
            [nominal.t[-1]], [nominal.downrange[-1]], [nominal.altitude[-1]],
            color='blue', s=100, marker='o'
        )

    # 收集时间和行距范围用于绘制参考平面
    t_min, t_max = 0, 100
    dr_min, dr_max = 0, 500
    openloop_plotted = False  # 标记是否已绘制开环轨迹

    # 加载各eta下的开环和重规划轨迹
    for eta in ETA_VALUES:
        domain = determine_mission_domain(eta)
        color = DOMAIN_COLORS.get(domain, '#3498db')

        openloop, replan = all_data.get(fault_key, {}).get(eta, (None, None))

        # 绘制故障开环轨迹（只绘制一次，因为各eta的开环轨迹相同）
        if openloop is not None and not openloop_plotted:
            ax.plot(
                openloop.t, openloop.downrange, openloop.altitude,
                '--', color='gray', linewidth=2, label="故障开环轨迹", alpha=0.7
            )
            ax.scatter(
                [openloop.t[-1]], [openloop.downrange[-1]], [openloop.altitude[-1]],
                color='gray', s=80, marker='x', zorder=5
            )
            openloop_plotted = True
            # 更新范围
            t_max = max(t_max, openloop.t[-1])
            dr_max = max(dr_max, openloop.downrange[-1])

        # 绘制重规划轨迹
        if replan is not None:
            ax.plot(
                replan.t, replan.downrange, replan.altitude,
                '-', color=color, linewidth=2.5,
                label=f'重规划 {domain} (η={eta})'
            )
            ax.scatter(
                [replan.t[-1]], [replan.downrange[-1]], [replan.altitude[-1]],
                color=color, s=120, marker='*', zorder=10
            )
            # 更新范围
            t_max = max(t_max, replan.t[-1])
            dr_max = max(dr_max, replan.downrange[-1])

    # 绘制轨道高度参考线
    for orbit_name, height in ORBIT_HEIGHTS.items():
        orbit_color = DOMAIN_COLORS.get(orbit_name, 'gray')
        # 绘制水平参考线（在固定时间位置）
        t_ref = t_max * 0.1  # 在图的左侧绘制
        dr_range = np.linspace(dr_min, dr_max * 0.3, 2)
        ax.plot(
            [t_ref, t_ref], dr_range, [height, height],
            '--', color=orbit_color, linewidth=1.5, alpha=0.6
        )
        # 添加高度标签
        ax.text(
            t_ref, dr_max * 0.32, height,
            f'{height}km\n({orbit_name})',
            fontsize=8, color=orbit_color, ha='left', va='bottom'
        )

    # 设置轴标签
    ax.set_xlabel("时间 (s)", fontsize=12, labelpad=10)
    ax.set_ylabel("地面行距 (km)", fontsize=12, labelpad=10)
    ax.set_zlabel("高度 (km)", fontsize=12, labelpad=10)

    # 设置标题
    fault_name = FAULT_NAMES.get(fault_key, fault_key)
    ax.set_title(f"{fault_key}: {fault_name} - 3D轨迹对比", fontsize=13, fontweight='bold')

    # 图例
    ax.legend(loc='upper left', fontsize=9)

    # 调整视角
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()

    # 保存
    out_path = out_dir / f"fig4_3d_{fault_key}_trajectory.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: {out_path.name}")


def make_downrange_altitude_2d_plot(
    all_data: Dict,
    out_dir: Path,
) -> None:
    """生成地面行距-高度2D轨迹图（所有故障场景）。"""
    from matplotlib.patches import Ellipse

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    fault_keys = list(FAULT_IDS)
    colors_eta = ['#2ecc71', '#f39c12', '#e74c3c']  # 对应 0.2, 0.5, 0.8

    for idx, fault_key in enumerate(fault_keys):
        if idx >= 5:
            break
        ax = axes[idx]

        # 加载名义轨迹
        nominal = all_data.get("nominal", {}).get(0.0, (None, None))[0]
        if nominal is not None:
            ax.plot(
                nominal.downrange, nominal.altitude,
                'b-', linewidth=2, label="名义轨迹", alpha=0.8
            )

        # 加载各eta下的重规划轨迹
        for i, eta in enumerate(ETA_VALUES):
            domain = determine_mission_domain(eta)
            color = DOMAIN_COLORS.get(domain, colors_eta[i])

            _, replan = all_data.get(fault_key, {}).get(eta, (None, None))
            if replan is not None:
                ax.plot(
                    replan.downrange, replan.altitude,
                    '-', color=color, linewidth=2.5,
                    label=f'{domain} (η={eta})'
                )
                ax.scatter(
                    [replan.downrange[-1]], [replan.altitude[-1]],
                    color=color, s=80, marker='*', zorder=10
                )

        # 添加目标高度参考线
        ax.axhline(y=500, color='green', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=300, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=0, color='brown', linestyle='-', linewidth=1.5, alpha=0.4)

        # 为SAFE_AREA落地点添加椭圆形安全落区
        _, replan_safe = all_data.get(fault_key, {}).get(0.8, (None, None))
        if replan_safe is not None and len(replan_safe.downrange) > 0:
            landing_dr = replan_safe.downrange[-1]
            safe_ellipse = Ellipse(
                xy=(landing_dr, 0), width=300, height=50,
                facecolor='lightgreen', edgecolor='green',
                alpha=0.4, linewidth=2, linestyle='--'
            )
            ax.add_patch(safe_ellipse)
            ax.text(landing_dr, 35, '安全落区', fontsize=8, color='darkgreen',
                   fontweight='bold', ha='center', va='bottom')

        fault_name = FAULT_NAMES.get(fault_key, fault_key)
        ax.set_title(f'{fault_key}: {fault_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('地面行距 (km)', fontsize=10)
        ax.set_ylabel('高度 (km)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)
        ax.set_ylim(bottom=-20)

    # 隐藏多余子图，添加图例说明
    axes[5].axis('off')
    legend_text = (
        "任务域说明:\n"
        "• RETAIN (η<0.3): 目标500km入轨\n"
        "• DEGRADED (0.3≤η<0.7): 目标300km低轨\n"
        "• SAFE_AREA (η≥0.7): 目标地面安全区"
    )
    axes[5].text(0.5, 0.5, legend_text, transform=axes[5].transAxes,
                 fontsize=11, verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('五种故障场景的轨迹对比（地面行距-高度）', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_dir / 'fig4_2d_all_domains_downrange_altitude.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {out_path.name}")


def main() -> None:
    """主函数：生成任务域相关图表。"""
    setup_matplotlib()
    print("=" * 60)
    print("图4-22 ~ 图4-24：任务域相关图表（使用真实轨迹数据）")
    print("=" * 60)

    # 检查数据目录
    if not DATA_DIR.exists():
        print(f"[ERROR] 数据目录不存在: {DATA_DIR}")
        print("请先运行: python -m scripts.eval_ch4_trajectories_replan")
        return

    print(f"\n[1/6] 加载轨迹数据: {DATA_DIR}")
    all_data = load_all_trajectories()

    # 统计加载情况
    loaded_count = 0
    for fault_id in FAULT_IDS:
        for eta in ETA_VALUES:
            _, replan = all_data.get(fault_id, {}).get(eta, (None, None))
            if replan is not None:
                loaded_count += 1
    print(f"  - 成功加载 {loaded_count}/{len(FAULT_IDS) * len(ETA_VALUES)} 条重规划轨迹")

    print("\n[2/6] 生成图4-22：RETAIN 域下五种故障的飞行量响应...")
    plot_fig4_22_retain_domain_responses(all_data, OUT_DIR)

    print("\n[3/6] 生成图4-23：F1 故障三种任务域策略对比...")
    plot_fig4_23_f1_severe_downrange(all_data, OUT_DIR)

    print("\n[4/6] 生成图4-24：五种严重故障终端状态汇总对比...")
    plot_fig4_24_severe_downrange_summary(all_data, OUT_DIR)

    print("\n[5/6] 生成各故障场景3D轨迹图...")
    for fault_key in FAULT_IDS:
        print(f"  处理 {fault_key}...")
        make_3d_trajectory_plot(fault_key, all_data, OUT_DIR)

    print("\n[6/6] 生成2D地面行距-高度对比图...")
    make_downrange_altitude_2d_plot(all_data, OUT_DIR)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
