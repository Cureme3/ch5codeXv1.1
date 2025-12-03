#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为同一故障场景生成三任务域（RETAIN/DEGRADED/SAFE_AREA）的轨迹对比图（第四章 4.4 节）。

本脚本展示任务域切换对重规划轨迹的影响。

输出文件：
- outputs/ch4/figures/ch4_mission_domains/{fault_id}_eta{eta}_domains_altitude.png/pdf
- outputs/ch4/figures/ch4_mission_domains/{fault_id}_eta{eta}_domains_downrange.png/pdf
- outputs/ch4/figures/ch4_mission_domains/{fault_id}_eta{eta}_domains_qn.png/pdf（可选）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from src.plots.plotting import setup_matplotlib
from src.sim.mission_domains import MissionDomain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成三任务域轨迹对比图（RETAIN/DEGRADED/SAFE_AREA）"
    )
    parser.add_argument(
        "--fault-id",
        type=str,
        default="F1_thrust_deg15",
        help="故障场景 ID，默认 F1_thrust_deg15",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.8,
        help="故障严重度 η，默认 0.8",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认 outputs/ch4/figures/ch4_mission_domains/",
    )
    parser.add_argument(
        "--include-constraints",
        action="store_true",
        help="是否生成 q/n 约束对比图",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run 模式：不实际调用求解器，仅验证脚本导入",
    )
    return parser.parse_args()


def save_figure_with_setup(fig: plt.Figure, outfile: Path) -> None:
    """保存图像到 PDF 和 PNG。"""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    fig.savefig(outfile.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    print(f"  - 图像已保存: {outfile} 和 {outfile.with_suffix('.pdf')}")


def plot_altitude_comparison(
    traj_retain,
    traj_degraded,
    traj_safe,
    out_dir: Path,
    fault_id: str,
    eta: float,
) -> None:
    """生成高度 vs 时间对比图。

    Parameters
    ----------
    traj_retain, traj_degraded, traj_safe : ThreeTrajectories
        三个任务域的轨迹数据
    out_dir : Path
        输出目录
    fault_id : str
        故障场景 ID
    eta : float
        故障严重度
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 名义轨迹（灰色背景线）
    ax.plot(
        traj_retain.t,
        traj_retain.h_nom,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="名义轨迹",
    )

    # 三任务域重规划轨迹
    ax.plot(
        traj_retain.t,
        traj_retain.h_reconfig,
        color="#2ecc71",
        linestyle="-",
        linewidth=2,
        label="RETAIN",
    )
    ax.plot(
        traj_degraded.t,
        traj_degraded.h_reconfig,
        color="#f39c12",
        linestyle="-",
        linewidth=2,
        label="DEGRADED",
    )
    ax.plot(
        traj_safe.t,
        traj_safe.h_reconfig,
        color="#e74c3c",
        linestyle="-",
        linewidth=2,
        label="SAFE_AREA",
    )

    ax.set_xlabel("时间 / s")
    ax.set_ylabel("高度 / km")
    ax.set_title(f"三任务域轨迹对比：高度 ({fault_id}, η={eta:.1f})")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    filename = f"{fault_id}_eta{eta:.1f}_domains_altitude.png"
    save_figure_with_setup(fig, out_dir / filename)


def plot_downrange_comparison(
    traj_retain,
    traj_degraded,
    traj_safe,
    out_dir: Path,
    fault_id: str,
    eta: float,
) -> None:
    """生成下行程 vs 时间对比图。

    Parameters
    ----------
    traj_retain, traj_degraded, traj_safe : ThreeTrajectories
        三个任务域的轨迹数据
    out_dir : Path
        输出目录
    fault_id : str
        故障场景 ID
    eta : float
        故障严重度
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 名义轨迹（灰色背景线）
    ax.plot(
        traj_retain.t,
        traj_retain.s_nom,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="名义轨迹",
    )

    # 三任务域重规划轨迹
    ax.plot(
        traj_retain.t,
        traj_retain.s_reconfig,
        color="#2ecc71",
        linestyle="-",
        linewidth=2,
        label="RETAIN",
    )
    ax.plot(
        traj_degraded.t,
        traj_degraded.s_reconfig,
        color="#f39c12",
        linestyle="-",
        linewidth=2,
        label="DEGRADED",
    )
    ax.plot(
        traj_safe.t,
        traj_safe.s_reconfig,
        color="#e74c3c",
        linestyle="-",
        linewidth=2,
        label="SAFE_AREA",
    )

    ax.set_xlabel("时间 / s")
    ax.set_ylabel("下行程 / km")
    ax.set_title(f"三任务域轨迹对比：下行程 ({fault_id}, η={eta:.1f})")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    filename = f"{fault_id}_eta{eta:.1f}_domains_downrange.png"
    save_figure_with_setup(fig, out_dir / filename)


def plot_qn_constraints_comparison(
    traj_retain,
    traj_degraded,
    traj_safe,
    out_dir: Path,
    fault_id: str,
    eta: float,
    q_max: float = 55.0,
    n_max: float = 3.5,
) -> None:
    """生成动压/过载约束对比图（可选）。

    Parameters
    ----------
    traj_retain, traj_degraded, traj_safe : ThreeTrajectories
        三个任务域的轨迹数据
    out_dir : Path
        输出目录
    fault_id : str
        故障场景 ID
    eta : float
        故障严重度
    q_max : float
        动压上限 [kPa]，默认 55.0
    n_max : float
        过载上限 [g]，默认 3.5
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 子图 1：动压
    ax1.axhline(q_max, color="red", linestyle=":", linewidth=1.5, label=f"$q_{{max}}$={q_max} kPa")
    ax1.plot(
        traj_retain.t, traj_retain.q_reconfig, color="#2ecc71", linewidth=2, label="RETAIN"
    )
    ax1.plot(
        traj_degraded.t, traj_degraded.q_reconfig, color="#f39c12", linewidth=2, label="DEGRADED"
    )
    ax1.plot(
        traj_safe.t, traj_safe.q_reconfig, color="#e74c3c", linewidth=2, label="SAFE_AREA"
    )
    ax1.set_ylabel("动压 / kPa")
    ax1.set_title(f"三任务域约束对比：动压 ({fault_id}, η={eta:.1f})")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 子图 2：法向过载
    ax2.axhline(n_max, color="red", linestyle=":", linewidth=1.5, label=f"$n_{{max}}$={n_max} g")
    ax2.plot(
        traj_retain.t, traj_retain.n_reconfig, color="#2ecc71", linewidth=2, label="RETAIN"
    )
    ax2.plot(
        traj_degraded.t, traj_degraded.n_reconfig, color="#f39c12", linewidth=2, label="DEGRADED"
    )
    ax2.plot(
        traj_safe.t, traj_safe.n_reconfig, color="#e74c3c", linewidth=2, label="SAFE_AREA"
    )
    ax2.set_xlabel("时间 / s")
    ax2.set_ylabel("法向过载 / g")
    ax2.set_title(f"三任务域约束对比：法向过载 ({fault_id}, η={eta:.1f})")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    filename = f"{fault_id}_eta{eta:.1f}_domains_qn.png"
    save_figure_with_setup(fig, out_dir / filename)


def main() -> None:
    """主函数。"""
    args = parse_args()

    setup_matplotlib()

    # 确定输出目录
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "outputs" / "ch4" / "figures" / "ch4_mission_domains"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("第四章 4.4：三任务域轨迹对比图生成")
    print("=" * 80)
    print(f"故障场景: {args.fault_id}")
    print(f"严重度 η: {args.eta}")
    print(f"输出目录: {out_dir}")
    print(f"Dry-run: {args.dry_run}")
    print()

    if args.dry_run:
        print("[INFO] Dry-run 模式：跳过轨迹生成，仅验证脚本导入")
        print("       若要真正运行，请去掉 --dry-run 参数")
        print()
        print("=" * 80)
        print("脚本导入验证通过！")
        print("=" * 80)
        return

    # 检查 cvxpy 是否可用
    try:
        import cvxpy
    except ImportError:
        print("[ERROR] cvxpy 未安装，无法运行轨迹生成")
        print("        请安装 cvxpy 或使用 --dry-run 模式")
        return

    # 延迟导入，避免在 dry-run 模式下因缺少 cvxpy 而失败
    from src.sim.viz_trajectories import build_three_trajectories

    # 生成三个任务域的轨迹
    print("[1/4] 生成 RETAIN 域轨迹...")
    traj_retain = build_three_trajectories(
        fault_id=args.fault_id,
        eta=args.eta,
        mission_domain=MissionDomain.RETAIN,
        t_step=1.0,
        nodes=40,
        solver_profile="fast",
    )
    print()

    print("[2/4] 生成 DEGRADED 域轨迹...")
    traj_degraded = build_three_trajectories(
        fault_id=args.fault_id,
        eta=args.eta,
        mission_domain=MissionDomain.DEGRADED,
        t_step=1.0,
        nodes=40,
        solver_profile="fast",
    )
    print()

    print("[3/4] 生成 SAFE_AREA 域轨迹...")
    traj_safe = build_three_trajectories(
        fault_id=args.fault_id,
        eta=args.eta,
        mission_domain=MissionDomain.SAFE_AREA,
        t_step=1.0,
        nodes=40,
        solver_profile="fast",
    )
    print()

    # 生成对比图
    print("[4/4] 生成对比图...")
    print("  - 高度对比图")
    plot_altitude_comparison(
        traj_retain, traj_degraded, traj_safe, out_dir, args.fault_id, args.eta
    )

    print("  - 下行程对比图")
    plot_downrange_comparison(
        traj_retain, traj_degraded, traj_safe, out_dir, args.fault_id, args.eta
    )

    if args.include_constraints:
        print("  - 约束对比图（q/n）")
        plot_qn_constraints_comparison(
            traj_retain, traj_degraded, traj_safe, out_dir, args.fault_id, args.eta
        )

    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
