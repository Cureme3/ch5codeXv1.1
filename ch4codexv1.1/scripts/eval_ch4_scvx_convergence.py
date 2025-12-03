#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四章：SCvx 收敛性能评估脚本。

对若干代表性工况（故障场景 + eta 组合）执行 SCvx 规划，
导出详细的迭代收敛日志到 CSV，用于后续绘制收敛曲线图。

输出文件：
- outputs/data/ch4_scvx_convergence_<scenario>_<eta>.csv
- outputs/data/scvx_convergence_log.csv (合并日志，用于默认绘图脚本)

用法:
    python -m scripts.eval_ch4_scvx_convergence [--scenarios F1,F2] [--etas 0.3,0.5,0.7]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from sim.run_fault import simulate_fault_and_solve  # noqa: E402


# 默认测试工况
DEFAULT_SCENARIOS = ["F1_thrust_deg15", "F2_tvc_rate4"]
DEFAULT_ETAS = [0.3, 0.5, 0.7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 SCvx 收敛性能")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="F1_thrust_deg15,F2_tvc_rate4",
        help="逗号分隔的故障场景 ID 列表",
    )
    parser.add_argument(
        "--etas",
        type=str,
        default="0.3,0.5,0.7",
        help="逗号分隔的 eta 值列表",
    )
    parser.add_argument(
        "--solver-profile",
        type=str,
        default="convergence",
        help="求解器配置 (fast/convergence)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=40,
        help="SCvx 离散节点数",
    )
    return parser.parse_args()


def export_iteration_logs_to_csv(
    logs: List,
    scenario_id: str,
    eta: float,
    out_dir: Path,
) -> Path:
    """将 IterationLog 列表导出为 CSV 文件。"""
    eta_str = f"eta{eta:.2f}"
    filename = f"ch4_scvx_convergence_{scenario_id}_{eta_str}.csv"
    out_path = out_dir / filename
    out_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # CSV header（与 make_figs_ch4_scvx_convergence.py 期望格式一致）
        writer.writerow([
            "iter_idx",
            "total_cost",
            "cost_state",
            "cost_control",
            "cost_slack",
            "cost_terminal",
            "feas_violation",
            "max_slack_q",
            "max_slack_n",
            "max_slack_cone",
            "term_error_norm",
            "trust_radius",
            "solver_status",
            "rho",
            "cost_nominal",
            "cost_candidate",
        ])
        for entry in logs:
            writer.writerow([
                getattr(entry, "iter_idx", 0),
                getattr(entry, "total_cost", 0.0),
                getattr(entry, "cost_state", 0.0),
                getattr(entry, "cost_control", 0.0),
                getattr(entry, "cost_slack", 0.0),
                getattr(entry, "cost_terminal", 0.0),
                getattr(entry, "feasibility_violation", 0.0),
                getattr(entry, "max_slack_q", 0.0),
                getattr(entry, "max_slack_n", 0.0),
                getattr(entry, "max_slack_cone", 0.0),
                getattr(entry, "term_error_norm", 0.0),
                getattr(entry, "trust_radius", 0.0),
                getattr(entry, "solver_status", "unknown"),
                getattr(entry, "rho", 0.0),
                getattr(entry, "cost_nominal", 0.0),
                getattr(entry, "cost_candidate", 0.0),
            ])

    return out_path


def main() -> None:
    args = parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",")]
    etas = [float(e.strip()) for e in args.etas.split(",")]

    print("=" * 80)
    print("第四章 4.2：SCvx 收敛性能评估")
    print("=" * 80)
    print(f"故障场景: {scenarios}")
    print(f"Eta 值: {etas}")
    print(f"求解器配置: {args.solver_profile}")
    print(f"节点数: {args.nodes}")
    print()

    out_dir = Path("outputs/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_logs = []  # 收集所有日志用于合并文件
    generated_files = []

    for scenario_id in scenarios:
        for eta in etas:
            print(f"[RUN] 场景={scenario_id}, eta={eta:.2f}")

            try:
                result = simulate_fault_and_solve(
                    scenario_id,
                    eta=eta,
                    nodes=args.nodes,
                    use_adaptive_penalties=True,
                    solver_profile=args.solver_profile,
                )

                # 提取 SCvx 迭代日志
                logs = result.diagnostics.get("scvx_logs", [])
                num_iters = result.diagnostics.get("num_iterations", len(logs))
                solver_status = result.diagnostics.get("solver_status", "unknown")

                print(f"      迭代次数: {num_iters}, 状态: {solver_status}")

                if logs:
                    # 导出单个场景的收敛日志
                    csv_path = export_iteration_logs_to_csv(logs, scenario_id, eta, out_dir)
                    generated_files.append(csv_path)
                    print(f"      已导出: {csv_path}")

                    # 添加到合并日志
                    for log in logs:
                        all_logs.append((scenario_id, eta, log))
                else:
                    print("      [WARN] 无迭代日志可导出")

            except Exception as e:
                print(f"      [ERROR] {e}")
                import traceback
                traceback.print_exc()

    # 生成合并日志文件（用于默认绘图脚本）
    if all_logs:
        # 取第一个场景的日志作为默认日志
        first_scenario, first_eta, _ = all_logs[0]
        default_logs = [log for s, e, log in all_logs if s == first_scenario and e == first_eta]

        default_log_path = out_dir / "scvx_convergence_log.csv"
        with default_log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iter_idx",
                "total_cost",
                "cost_state",
                "cost_control",
                "cost_slack",
                "cost_terminal",
                "feas_violation",
                "max_slack_q",
                "max_slack_n",
                "max_slack_cone",
                "term_error_norm",
                "trust_radius",
                "solver_status",
                "rho",
                "cost_nominal",
                "cost_candidate",
            ])
            for entry in default_logs:
                writer.writerow([
                    getattr(entry, "iter_idx", 0),
                    getattr(entry, "total_cost", 0.0),
                    getattr(entry, "cost_state", 0.0),
                    getattr(entry, "cost_control", 0.0),
                    getattr(entry, "cost_slack", 0.0),
                    getattr(entry, "cost_terminal", 0.0),
                    getattr(entry, "feasibility_violation", 0.0),
                    getattr(entry, "max_slack_q", 0.0),
                    getattr(entry, "max_slack_n", 0.0),
                    getattr(entry, "max_slack_cone", 0.0),
                    getattr(entry, "term_error_norm", 0.0),
                    getattr(entry, "trust_radius", 0.0),
                    getattr(entry, "solver_status", "unknown"),
                    getattr(entry, "rho", 0.0),
                    getattr(entry, "cost_nominal", 0.0),
                    getattr(entry, "cost_candidate", 0.0),
                ])
        generated_files.append(default_log_path)
        print(f"\n[INFO] 默认收敛日志已导出: {default_log_path}")

    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"生成的文件:")
    for f in generated_files:
        print(f"  - {f}")
    print()
    print("下一步：运行 make_figs_ch4_scvx_convergence.py 生成收敛曲线图")


if __name__ == "__main__":
    main()
