#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成第四章 4.3 中自适应罚权重随故障严重度 eta 变化的图表和表格。

输出文件：
- outputs/figures/ch4_adaptive_weights/adaptive_weights_vs_eta.pdf/png
- outputs/tables/table_ch4_adaptive_weights.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from learn.weights import compute_adaptive_penalties  # noqa: E402
from plots.plotting import save_figure, setup_matplotlib  # noqa: E402


def main() -> None:
    print("=" * 80)
    print("第四章 4.3：自适应罚权重随故障严重度变化图表生成")
    print("=" * 80)
    print()

    setup_matplotlib()

    figs_dir = PROJECT_ROOT / "outputs" / "figures" / "ch4_adaptive_weights"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 计算权重随 eta 的变化
    print("[1/3] 计算自适应权重...")
    etas = np.linspace(0.0, 1.0, 21)
    w_terminal = []
    w_state = []
    w_control = []
    w_q_slack = []
    w_n_slack = []
    w_cone_slack = []

    for eta in etas:
        pw = compute_adaptive_penalties(float(eta))
        w_terminal.append(pw.terminal_state_dev)
        w_state.append(pw.state_dev)
        w_control.append(pw.control_dev)
        w_q_slack.append(pw.q_slack)
        w_n_slack.append(pw.n_slack)
        w_cone_slack.append(pw.cone_slack)

    # 绘图：各权重随 eta 的变化
    print("[2/3] 生成权重变化曲线...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(etas, w_terminal, label="终端误差权重", linewidth=2, marker="o", markersize=4)
    ax.plot(etas, w_state, label="状态偏差权重", linewidth=2, marker="s", markersize=4)
    ax.plot(etas, w_control, label="控制偏差权重", linewidth=2, marker="^", markersize=4)
    ax.plot(etas, w_q_slack, label="动压松弛权重", linewidth=2, marker="d", markersize=4, linestyle="--")
    ax.plot(etas, w_n_slack, label="过载松弛权重", linewidth=2, marker="v", markersize=4, linestyle="--")
    ax.plot(etas, w_cone_slack, label="推力锥松弛权重", linewidth=2, marker="<", markersize=4, linestyle="--")

    ax.set_xlabel(r"故障严重度 $\eta$")
    ax.set_ylabel("罚权重")
    ax.set_title("自适应罚权重随故障严重度变化")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    save_figure(fig, figs_dir / "fig4_15_adaptive_weights_vs_eta.png")
    print(f"  - 图表已保存: {figs_dir / 'fig4_15_adaptive_weights_vs_eta.png'}")
    print(f"  - 图表已保存: {figs_dir / 'fig4_15_adaptive_weights_vs_eta.pdf'}")

    # 典型 eta 取值下的配置表
    print("[3/3] 生成典型权重配置表...")
    sample_etas = [0.0, 0.3, 0.6, 1.0]
    rows = []
    for eta in sample_etas:
        pw = compute_adaptive_penalties(float(eta))
        rows.append(
            (
                eta,
                pw.terminal_state_dev,
                pw.state_dev,
                pw.control_dev,
                pw.q_slack,
                pw.n_slack,
                pw.cone_slack,
            )
        )

    md_path = tables_dir / "table_ch4_adaptive_weights.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# 自适应罚权重典型配置\n\n")
        f.write("| η | 终端误差 | 状态偏差 | 控制偏差 | 动压松弛 | 过载松弛 | 推力锥松弛 |\n")
        f.write("| --- | -------- | -------- | -------- | -------- | -------- | ---------- |\n")
        for eta, wt, ws, wc, wq, wn, wcone in rows:
            f.write(f"| {eta:.1f} | {wt:.2f} | {ws:.2f} | {wc:.2f} | {wq:.2f} | {wn:.2f} | {wcone:.2f} |\n")

    print(f"  - 表格已保存: {md_path}")
    print()

    print("=" * 80)
    print("全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
