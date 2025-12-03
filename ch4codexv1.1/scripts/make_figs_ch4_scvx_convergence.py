#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""根据 run_scvx_demo 导出的收敛日志 CSV，生成第四章 4.2 中的 SCvx 收敛图与统计表。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.plots.plotting import setup_matplotlib  # noqa: E402


def load_log_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """读取收敛日志 CSV，返回 (iters, total_cost, feas_violation, trust_radius, cost_decomp)。

    cost_decomp 字典包含可选的成本分解列（若存在）：
    - cost_state, cost_control, cost_terminal, cost_slack
    """

    if not path.exists():
        raise FileNotFoundError(f"Log CSV not found: {path}")

    iters = []
    total_cost = []
    feas = []
    trust_radius = []
    cost_decomp = {"cost_state": [], "cost_control": [], "cost_terminal": [], "cost_slack": []}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_decomp = all(k in reader.fieldnames for k in cost_decomp.keys()) if reader.fieldnames else False
        has_trust_radius = "trust_radius" in reader.fieldnames if reader.fieldnames else False

        for row in reader:
            iters.append(int(row["iter_idx"]))
            total_cost.append(float(row["total_cost"]))
            feas.append(float(row["feas_violation"]))
            if has_trust_radius:
                trust_radius.append(float(row["trust_radius"]))

            # 读取成本分解列（如果存在）
            if has_decomp:
                for key in cost_decomp.keys():
                    cost_decomp[key].append(float(row[key]))

    # 转换为numpy数组（如果有数据）
    for key in cost_decomp.keys():
        if cost_decomp[key]:
            cost_decomp[key] = np.array(cost_decomp[key])
        else:
            cost_decomp[key] = None

    trust_radius_arr = np.array(trust_radius) if trust_radius else None
    return np.array(iters), np.array(total_cost), np.array(feas), trust_radius_arr, cost_decomp


def save_figure_with_setup(fig: plt.Figure, outfile: Path) -> None:
    """保存图像到 PDF 和 PNG。"""

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    fig.savefig(outfile.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    print(f"图像已保存: {outfile} 和 {outfile.with_suffix('.pdf')}")


def plot_cost_convergence(iters: np.ndarray, total_cost: np.ndarray, figs_dir: Path) -> None:
    """绘制目标函数收敛曲线。"""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, total_cost, marker="o", linestyle="-", linewidth=2, markersize=5, color="#00468B")
    ax.set_xlabel(r"迭代次数 $k$")
    ax.set_ylabel(r"目标函数值 $J$")
    ax.set_title("SCvx 收敛曲线：目标函数")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend([r"$J(k)$"], loc="best", frameon=True, edgecolor='black', fancybox=False)
    save_figure_with_setup(fig, figs_dir / "fig4_07_scvx_convergence_cost.png")


def plot_feasibility_convergence(iters: np.ndarray, feas: np.ndarray, figs_dir: Path) -> None:
    """绘制约束可行性收敛曲线。"""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, feas, marker="s", linestyle="-", linewidth=2, markersize=5, color="#ED0000")
    ax.set_xlabel(r"迭代次数 $k$")
    ax.set_ylabel(r"约束违背度 $\|\xi\|_{\infty}$")
    ax.set_title("SCvx 收敛曲线：约束可行性")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend([r"$\|\xi(k)\|_{\infty}$"], loc="best", frameon=True, edgecolor='black', fancybox=False)
    save_figure_with_setup(fig, figs_dir / "fig4_07_scvx_convergence_feas.png")


def plot_cost_decomposition(iters: np.ndarray, cost_decomp: dict, figs_dir: Path) -> None:
    """绘制成本分解随迭代变化曲线（用于论文第四章 4.2 小节）。

    参数:
        iters: 迭代索引数组
        cost_decomp: 成本分解字典，包含 cost_state, cost_control, cost_terminal, cost_slack
        figs_dir: 输出目录
    """
    # 检查是否有成本分解数据
    if not cost_decomp or all(v is None for v in cost_decomp.values()):
        print("  [WARN] CSV 中无成本分解列，跳过成本分解图生成")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # 绘制各成本项（如果存在）
    if cost_decomp["cost_state"] is not None:
        ax.plot(iters, cost_decomp["cost_state"], marker="o", label="状态偏差代价", linewidth=2, markersize=4)
    if cost_decomp["cost_control"] is not None:
        ax.plot(iters, cost_decomp["cost_control"], marker="s", label="控制偏差代价", linewidth=2, markersize=4)
    if cost_decomp["cost_terminal"] is not None:
        ax.plot(iters, cost_decomp["cost_terminal"], marker="^", label="终端误差代价", linewidth=2, markersize=4)
    if cost_decomp["cost_slack"] is not None:
        ax.plot(iters, cost_decomp["cost_slack"], marker="v", label="松弛变量代价", linewidth=2, markersize=4)

    ax.set_xlabel("迭代次数")
    ax.set_ylabel("代价值")
    ax.set_yscale("log")
    ax.set_title("SCvx 收敛曲线：成本分解")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    save_figure_with_setup(fig, figs_dir / "fig4_07_scvx_convergence_decomp.png")


def plot_trust_region(iters: np.ndarray, trust_radius: np.ndarray, figs_dir: Path) -> None:
    """绘制信赖域半径随迭代变化曲线。"""
    if trust_radius is None or len(trust_radius) == 0:
        print("  [WARN] 无信赖域数据，跳过信赖域图生成")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, trust_radius, marker="^", linestyle="-", linewidth=2, markersize=5, color="#2ca02c")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("信赖域半径")
    ax.set_title("SCvx 收敛曲线：信赖域")
    ax.grid(True, linestyle="--", alpha=0.4)
    save_figure_with_setup(fig, figs_dir / "fig4_07_scvx_convergence_trustregion.png")


def generate_stats_table(iters: np.ndarray, total_cost: np.ndarray, feas: np.ndarray, tables_dir: Path) -> None:
    """生成 SCvx 统计表（Markdown 格式）。"""

    num_iters = len(iters)
    initial_cost = total_cost[0] if len(total_cost) > 0 else float("nan")
    final_cost = total_cost[-1] if len(total_cost) > 0 else float("nan")
    final_feas = feas[-1] if len(feas) > 0 else float("nan")
    max_feas = np.max(feas) if len(feas) > 0 else float("nan")

    # Cost reduction
    if not np.isnan(initial_cost) and not np.isnan(final_cost) and initial_cost != 0:
        cost_reduction = (initial_cost - final_cost) / initial_cost * 100
    else:
        cost_reduction = float("nan")

    markdown_content = f"""# SCvx 收敛统计表

| 指标                     | 数值              |
| ------------------------ | ----------------- |
| 迭代次数                 | {num_iters}       |
| 初始目标函数值           | {initial_cost:.6e} |
| 最终目标函数值           | {final_cost:.6e}   |
| 成本降幅 (%)             | {cost_reduction:.2f} |
| 最大约束违背度           | {max_feas:.6e}     |
| 最终约束违背度           | {final_feas:.6e}   |
"""

    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "table_scvx_stats.md"
    table_path.write_text(markdown_content, encoding="utf-8")
    print(f"统计表已保存: {table_path}")


def main() -> None:
    """主函数：读取日志，生成图表和统计表。"""

    setup_matplotlib()

    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figs_dir = PROJECT_ROOT / "outputs" / "figures" / "ch4_scvx_convergence"
    figs_dir.mkdir(parents=True, exist_ok=True)

    log_path = PROJECT_ROOT / "outputs" / "data" / "scvx_convergence_log.csv"

    print("=" * 80)
    print("第四章 4.2 SCvx 收敛图表生成")
    print("=" * 80)
    print(f"读取日志: {log_path}")

    iters, total_cost, feas, trust_radius, cost_decomp = load_log_csv(log_path)

    print(f"迭代次数: {len(iters)}")
    print()

    # Generate plots
    print("[1/5] 生成目标函数收敛曲线...")
    plot_cost_convergence(iters, total_cost, figs_dir)

    print("[2/5] 生成约束可行性收敛曲线...")
    plot_feasibility_convergence(iters, feas, figs_dir)

    print("[3/5] 生成成本分解收敛曲线...")
    plot_cost_decomposition(iters, cost_decomp, figs_dir)

    print("[4/5] 生成信赖域收敛曲线...")
    plot_trust_region(iters, trust_radius, figs_dir)

    print("[5/5] 生成统计表...")
    generate_stats_table(iters, total_cost, feas, tables_dir)

    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"图表输出目录: {figs_dir}")
    print(f"统计表输出目录: {tables_dir}")


if __name__ == "__main__":
    main()
