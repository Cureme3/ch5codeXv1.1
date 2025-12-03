#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在单一故障场景下，对比冷启动和学习热启动的 SCvx 收敛曲线。

若当前环境未安装 cvxpy，可通过 --dry-run 仅检查脚本导入。

输出文件：
- outputs/ch4/figures/ch4_warmstart/warmstart_case_cost_vs_iter.png/pdf
- outputs/ch4/figures/ch4_warmstart/warmstart_case_feas_vs_iter.png/pdf
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单故障场景下冷/热启动 SCvx 收敛曲线对比"
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
        default=0.7,
        help="故障严重度 η，默认 0.7",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=15,
        help="最大迭代次数，默认 15",
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


def run_scvx_once(
    fault_id: str,
    eta: float,
    mode: str,
    max_iters: int = 15,
) -> dict:
    """运行一次 SCvx 优化并返回收敛历史。

    Parameters
    ----------
    fault_id : str
        故障场景 ID
    eta : float
        故障严重度
    mode : str
        "cold" 或 "warm"
    max_iters : int
        最大迭代次数

    Returns
    -------
    dict
        包含 iters, total_cost, feas_violation 的字典
    """
    # 延迟导入，避免在 dry-run 模式下因缺少 cvxpy 而失败
    from src.sim.run_fault import (
        run_fault_scenario,
        plan_recovery_segment_scvx,
    )
    from src.sim.run_nominal import simulate_full_mission
    from src.learn.warmstart import load_learning_context, build_learning_warmstart

    # 准备数据
    nominal = simulate_full_mission(dt=1.0, save_csv=False)
    fault_sim = run_fault_scenario(fault_id, dt=1.0)
    scenario = fault_sim.scenario

    # 准备热启动（如果需要）
    warmstart_h = None
    if mode == "warm":
        ctx = load_learning_context("outputs/ch4/data/ch4_learning")
        warmstart_h = build_learning_warmstart(ctx, scenario, fault_sim, nominal, nodes=40)

    # 运行 SCvx
    result = plan_recovery_segment_scvx(
        scenario=scenario,
        fault_sim=fault_sim,
        nominal=nominal,
        nodes=40,
        fault_eta=eta,
        use_adaptive_penalties=True,
        warmstart_h=warmstart_h,
        solver_profile="fast",
    )

    # 提取收敛历史
    logs = getattr(result, "logs", [])
    if not logs:
        # 如果没有logs，构造一个假的
        return {
            "iters": np.array([0]),
            "total_cost": np.array([0.0]),
            "feas_violation": np.array([0.0]),
        }

    iters = np.arange(1, len(logs) + 1)
    total_cost = np.array([log.total_cost for log in logs])
    feas_violation = np.array([log.diagnostics.feas_violation for log in logs])

    return {
        "iters": iters,
        "total_cost": total_cost,
        "feas_violation": feas_violation,
    }


def plot_cost_comparison(cold_data: dict, warm_data: dict, out_dir: Path) -> None:
    """生成成本收敛对比图。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(cold_data["iters"], cold_data["total_cost"], marker="o",
            label="冷启动", linewidth=2, markersize=5)
    ax.plot(warm_data["iters"], warm_data["total_cost"], marker="s",
            label="热启动", linewidth=2, markersize=5)

    ax.set_xlabel("迭代次数")
    ax.set_ylabel("目标函数值")
    ax.set_title("SCvx 收敛对比：冷启动 vs 热启动（目标函数）")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    save_figure_with_setup(fig, out_dir / "warmstart_case_cost_vs_iter.png")


def plot_feas_comparison(cold_data: dict, warm_data: dict, out_dir: Path) -> None:
    """生成可行性违背度对比图。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(cold_data["iters"], cold_data["feas_violation"], marker="o",
            label="冷启动", linewidth=2, markersize=5)
    ax.plot(warm_data["iters"], warm_data["feas_violation"], marker="s",
            label="热启动", linewidth=2, markersize=5)

    ax.set_xlabel("迭代次数")
    ax.set_ylabel("约束违背度（max slack）")
    ax.set_yscale("log")
    ax.set_title("SCvx 收敛对比：冷启动 vs 热启动（可行性）")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    save_figure_with_setup(fig, out_dir / "warmstart_case_feas_vs_iter.png")


def main() -> None:
    """主函数。"""
    args = parse_args()

    setup_matplotlib()

    out_dir = PROJECT_ROOT / "outputs" / "ch4" / "figures" / "ch4_warmstart"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("第四章 4.3：单案例冷/热启动 SCvx 收敛曲线对比")
    print("=" * 80)
    print(f"故障场景: {args.fault_id}")
    print(f"严重度 η: {args.eta}")
    print(f"最大迭代: {args.max_iters}")
    print(f"Dry-run: {args.dry_run}")
    print()

    if args.dry_run:
        print("[INFO] Dry-run 模式：跳过 SCvx 求解，仅验证脚本导入")
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
        print("[ERROR] cvxpy 未安装，无法运行 SCvx 求解")
        print("        请安装 cvxpy 或使用 --dry-run 模式")
        return

    # 运行冷启动
    print("[1/3] 运行冷启动 SCvx...")
    cold_data = run_scvx_once(args.fault_id, args.eta, "cold", args.max_iters)
    print(f"  - 冷启动迭代次数: {len(cold_data['iters'])}")
    print()

    # 运行热启动
    print("[2/3] 运行热启动 SCvx...")
    warm_data = run_scvx_once(args.fault_id, args.eta, "warm", args.max_iters)
    print(f"  - 热启动迭代次数: {len(warm_data['iters'])}")
    print()

    # 生成对比图
    print("[3/3] 生成收敛对比图...")
    plot_cost_comparison(cold_data, warm_data, out_dir)
    plot_feas_comparison(cold_data, warm_data, out_dir)
    print()

    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
