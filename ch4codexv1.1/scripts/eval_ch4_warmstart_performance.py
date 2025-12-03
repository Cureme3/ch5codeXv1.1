#!/usr/bin/env python
"""第四章：SCvx 冷/热启动性能评估脚本。

对 F1-F5 故障场景分别执行冷启动（无热启动）和热启动（使用学习预测）的 SCvx 规划，
记录迭代次数、CPU 时间、求解状态等指标，输出 CSV 和 Markdown 表格。

输出文件：
- outputs/data/ch4_warmstart_performance.csv
- outputs/tables/table_ch4_warmstart_performance.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from sim.run_fault import plan_recovery_segment_scvx, run_fault_scenario  # noqa: E402
from sim.run_nominal import simulate_full_mission  # noqa: E402
from learn.warmstart import build_learning_warmstart, load_learning_context  # noqa: E402


FAULT_ID_MAP: Dict[str, str] = {
    "F1": "F1_thrust_deg15",
    "F2": "F2_tvc_rate4",
    "F3": "F3_tvc_stuck3deg",
    "F4": "F4_sensor_bias2deg",
    "F5": "F5_event_delay5s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 SCvx 冷/热启动性能")
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="每个故障的重复次数",
    )
    parser.add_argument(
        "--solver-profile",
        type=str,
        default="fast",
        help="SCvx solver profile (e.g. 'fast', 'convergence').",
    )
    return parser.parse_args()


def prepare_fault_context(fault_id: str):
    """运行单个故障场景，返回 scenario、fault_sim、nominal。"""

    scenario_id = FAULT_ID_MAP[fault_id]
    nominal = simulate_full_mission(dt=1.0)
    fault_sim = run_fault_scenario(scenario_id, dt=1.0)
    scenario = fault_sim.scenario
    return scenario, fault_sim, nominal


def run_single_replan(
    ctx,
    scenario,
    fault_sim,
    nominal,
    mode: str,
    *,
    nodes: int = 40,
    use_adaptive_penalties: bool = True,
    solver_profile: str = "fast",
) -> dict:
    """运行一次 SCvx 重规划并返回记录行。"""

    warm_l2 = 0.0
    warmstart = None
    if mode == "warm":
        warm_h = build_learning_warmstart(ctx, scenario, fault_sim, nominal, nodes)
        if warm_h.shape[0] != nodes:
            raise ValueError(f"warmstart length {warm_h.shape[0]} != nodes {nodes}")
        warm_l2 = float(np.linalg.norm(warm_h))
        warmstart = warm_h
    t0 = time.perf_counter()
    result = plan_recovery_segment_scvx(
        scenario=scenario,
        fault_sim=fault_sim,
        nominal=nominal,
        nodes=nodes,
        eta=None,
        use_adaptive_penalties=use_adaptive_penalties,
        warmstart_h=warmstart,
        solver_profile=solver_profile,
    )
    t1 = time.perf_counter()
    diagnostics = result.diagnostics
    num_iters = getattr(result, "num_iterations", None)
    if num_iters is None:
        logs_attr = getattr(result, "logs", None)
        if logs_attr is not None:
            num_iters = len(logs_attr)
        else:
            num_iters = diagnostics.get("num_iterations")
    record = {
        "fault_id": scenario.id,
        "mode": mode,
        "iters": num_iters,
        "cpu_ms": (t1 - t0) * 1000.0,
        "status": diagnostics.get("solver_status", ""),
        "eta": diagnostics.get("eta"),
        "warmstart_l2": warm_l2,
        "solver_profile": solver_profile,
    }
    return record


def generate_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    """
    从 CSV 数据生成 Markdown 格式的性能对比表。

    表格包含：场景、冷启动/热启动的迭代次数和CPU时间、以及减少比例。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 按 fault_id 和 mode 分组计算平均值
    grouped = df.groupby(["fault_id", "mode"]).agg({
        "iters": "mean",
        "cpu_ms": "mean",
    }).reset_index()

    # 分离冷启动和热启动数据
    cold_df = grouped[grouped["mode"] == "cold"].set_index("fault_id")
    warm_df = grouped[grouped["mode"] == "warm"].set_index("fault_id")

    # 合并并计算减少比例
    fault_ids = sorted(set(grouped["fault_id"]))
    rows = []
    for fid in fault_ids:
        if fid in cold_df.index and fid in warm_df.index:
            cold_iters = cold_df.loc[fid, "iters"]
            warm_iters = warm_df.loc[fid, "iters"]
            cold_time = cold_df.loc[fid, "cpu_ms"] / 1000.0  # 转换为秒
            warm_time = warm_df.loc[fid, "cpu_ms"] / 1000.0

            iter_reduction = (cold_iters - warm_iters) / cold_iters * 100 if cold_iters > 0 else 0.0
            time_reduction = (cold_time - warm_time) / cold_time * 100 if cold_time > 0 else 0.0

            rows.append({
                "scenario": fid,
                "cold_iters": f"{cold_iters:.1f}",
                "warm_iters": f"{warm_iters:.1f}",
                "iter_reduction": f"{iter_reduction:.1f}%",
                "cold_time": f"{cold_time:.3f}",
                "warm_time": f"{warm_time:.3f}",
                "time_reduction": f"{time_reduction:.1f}%",
            })

    # 写入 Markdown 文件
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# SCvx 冷启动 vs 热启动性能对比\n\n")
        f.write("| 场景 | 冷启动迭代数 | 热启动迭代数 | 迭代减少比例 | 冷启动CPU时间[s] | 热启动CPU时间[s] | 时间减少比例 |\n")
        f.write("| ---- | ------------ | ------------ | ------------ | ---------------- | ---------------- | ------------ |\n")
        for row in rows:
            f.write(
                f"| {row['scenario']} | {row['cold_iters']} | {row['warm_iters']} | "
                f"{row['iter_reduction']} | {row['cold_time']} | {row['warm_time']} | "
                f"{row['time_reduction']} |\n"
            )

    print(f"Markdown table saved to {out_path}")


def main() -> None:
    args = parse_args()
    fault_ids = ["F1", "F2", "F3", "F4", "F5"]
    repeat = int(args.repeat)
    solver_profile = args.solver_profile
    records: List[dict] = []

    print("=" * 80)
    print("第四章 4.3：SCvx 冷启动 vs 热启动性能评估")
    print("=" * 80)
    print(f"重复次数: {repeat}")
    print(f"求解器配置: {solver_profile}")
    print()

    # 加载学习模型
    print("[1/3] 加载学习模型...")
    ctx = load_learning_context("outputs/data/ch4_learning")
    print(f"  - 模型节点数: {ctx.nodes}")
    print()

    nodes = 40

    # 运行所有场景的冷/热启动测试
    print("[2/3] 运行冷启动 vs 热启动测试...")
    for fid in fault_ids:
        print(f"  处理故障场景: {fid}")
        scenario, fault_sim, nominal = prepare_fault_context(fid)
        for rep in range(repeat):
            for mode in ("cold", "warm"):
                record = run_single_replan(
                    ctx,
                    scenario,
                    fault_sim,
                    nominal,
                    mode,
                    nodes=nodes,
                    use_adaptive_penalties=True,
                    solver_profile=solver_profile,
                )
                record["rep"] = rep
                records.append(record)
                print(f"    - {mode:4s} | rep={rep} | iters={record['iters']} | cpu_ms={record['cpu_ms']:.1f}")
    print()

    # 保存 CSV
    print("[3/3] 保存结果...")
    df = pd.DataFrame(records)

    # 保存 CSV 到 data 目录
    data_dir = Path("outputs/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_file = data_dir / "ch4_warmstart_performance.csv"
    df.to_csv(csv_file, index=False)
    print(f"  - CSV 已保存: {csv_file}")

    # 生成 Markdown 表到 tables 目录
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    md_file = tables_dir / "table_ch4_warmstart_performance.md"
    generate_markdown_table(df, md_file)
    print(f"  - Markdown 表已保存: {md_file}")
    print()

    print("=" * 80)
    print("全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
