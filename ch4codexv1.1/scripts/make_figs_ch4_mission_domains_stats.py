#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 ch4_mission_domains.csv 生成任务域分布统计图（第四章 4.4 节）。

输出文件：
- outputs/ch4/figures/ch4_mission_domains/mission_domain_distribution.png/pdf
- outputs/ch4/figures/ch4_mission_domains/domain_attempts_bar.png/pdf
- outputs/ch4/figures/ch4_mission_domains/final_feas_vs_eta.png/pdf（可选）
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from src.plots.plotting import setup_matplotlib


def save_figure_with_setup(fig: plt.Figure, outfile: Path) -> None:
    """保存图像到 PDF 和 PNG。"""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    fig.savefig(outfile.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    print(f"  - 图像已保存: {outfile} 和 {outfile.with_suffix('.pdf')}")


def plot_mission_domain_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """生成任务域分布堆叠柱状图。

    Parameters
    ----------
    df : pd.DataFrame
        包含 eta 和 mission_domain 列的数据
    out_dir : Path
        输出目录
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # 按 eta 分组，统计每个 mission_domain 的数量
    etas_sorted = sorted(df["eta"].unique())
    domains = ["RETAIN", "DEGRADED", "SAFE_AREA"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # 绿色/橙色/红色

    # 构造堆叠数据
    domain_counts = {domain: [] for domain in domains}
    for eta in etas_sorted:
        subset = df[df["eta"] == eta]
        total = len(subset)
        for domain in domains:
            count = len(subset[subset["mission_domain"] == domain])
            domain_counts[domain].append(count / total * 100 if total > 0 else 0)

    # 绘制堆叠柱状图
    x = np.arange(len(etas_sorted))
    width = 0.5
    bottom = np.zeros(len(etas_sorted))

    for idx, domain in enumerate(domains):
        ax.bar(
            x,
            domain_counts[domain],
            width,
            label=domain,
            color=colors[idx],
            bottom=bottom,
        )
        bottom += np.array(domain_counts[domain])

    ax.set_xlabel("故障严重度 η")
    ax.set_ylabel("任务域分布比例 / %")
    ax.set_title("不同故障严重度 η 下的任务域分布")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{eta:.1f}" for eta in etas_sorted])
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    save_figure_with_setup(fig, out_dir / "mission_domain_distribution.png")


def plot_domain_attempts_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """生成域升级次数统计柱状图。

    Parameters
    ----------
    df : pd.DataFrame
        包含 eta, fault_id, domain_attempts 列的数据
    out_dir : Path
        输出目录
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # 按 eta 分组，计算平均 domain_attempts
    etas_sorted = sorted(df["eta"].unique())
    fault_ids = sorted(df["fault_id"].unique())

    # 使用分组条形图展示不同 fault_id
    x = np.arange(len(etas_sorted))
    width = 0.15
    colors_faults = plt.cm.tab10(np.linspace(0, 1, len(fault_ids)))

    for idx, fid in enumerate(fault_ids):
        attempts = []
        for eta in etas_sorted:
            subset = df[(df["eta"] == eta) & (df["fault_id"] == fid)]
            avg_attempts = subset["domain_attempts"].mean() if len(subset) > 0 else 0
            attempts.append(avg_attempts)

        offset = (idx - len(fault_ids) / 2) * width
        ax.bar(
            x + offset,
            attempts,
            width,
            label=fid,
            color=colors_faults[idx],
            alpha=0.8,
        )

    ax.set_xlabel("故障严重度 η")
    ax.set_ylabel("平均域升级次数")
    ax.set_title("任务域升级次数统计（域升级机制触发频率）")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{eta:.1f}" for eta in etas_sorted])
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    save_figure_with_setup(fig, out_dir / "domain_attempts_bar.png")


def plot_feas_vs_eta(df: pd.DataFrame, out_dir: Path) -> None:
    """生成可行性违背度 vs η 散点图（可选）。

    Parameters
    ----------
    df : pd.DataFrame
        包含 eta, final_feas_violation, mission_domain 列的数据
    out_dir : Path
        输出目录
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    domains = ["RETAIN", "DEGRADED", "SAFE_AREA"]
    colors = {"RETAIN": "#2ecc71", "DEGRADED": "#f39c12", "SAFE_AREA": "#e74c3c"}
    markers = {"RETAIN": "o", "DEGRADED": "s", "SAFE_AREA": "^"}

    for domain in domains:
        subset = df[df["mission_domain"] == domain]
        ax.scatter(
            subset["eta"],
            subset["final_feas_violation"],
            label=domain,
            color=colors[domain],
            marker=markers[domain],
            s=80,
            alpha=0.7,
        )

    ax.set_xlabel("故障严重度 η")
    ax.set_ylabel("最终约束违背度")
    ax.set_yscale("log")
    ax.set_title("任务域升级对可行性的改善效果")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    save_figure_with_setup(fig, out_dir / "final_feas_vs_eta.png")


def main() -> None:
    """主函数：读取 CSV，生成三类统计图。"""
    setup_matplotlib()

    tables_dir = PROJECT_ROOT / "outputs" / "ch4" / "tables"
    out_dir = PROJECT_ROOT / "outputs" / "ch4" / "figures" / "ch4_mission_domains"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "ch4_mission_domains.csv"

    print("=" * 80)
    print("第四章 4.4：任务域分布统计图生成")
    print("=" * 80)
    print(f"读取数据: {csv_path}")

    if not csv_path.exists():
        print(f"[ERROR] 文件不存在: {csv_path}")
        print("        请先运行 python -m scripts.eval_ch4_mission_domains")
        return

    # 读取 CSV
    df = pd.read_csv(csv_path)
    print(f"  - 共 {len(df)} 条记录")
    print()

    # 生成三类统计图
    print("[1/3] 生成任务域分布图...")
    plot_mission_domain_distribution(df, out_dir)

    print("[2/3] 生成域升级次数统计图...")
    plot_domain_attempts_bar(df, out_dir)

    print("[3/3] 生成可行性 vs η 散点图（可选）...")
    plot_feas_vs_eta(df, out_dir)

    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
