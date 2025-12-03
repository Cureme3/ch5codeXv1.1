#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四章图表统一生成入口。

本脚本统一调度第四章所有图表/表格的生成，包括：
- 4.1 名义轨迹仿真曲线
- 4.2 SCvx 收敛曲线与统计表（含成本分解图）
- 4.3 自适应罚权重随故障严重度变化图
- 4.3 学习热启动性能统计对比图（迭代次数、CPU时间柱状图）
- 4.3 学习热启动单案例收敛曲线对比图（冷启动 vs 热启动）
- 4.4 多故障多严重度轨迹对比图（名义/故障开环/重规划）
- 4.4 路径约束对比图（动压/过载 vs 时间）
- 4.4 任务域分布统计图（RETAIN/DEGRADED/SAFE_AREA）
- 4.4 三任务域轨迹对比图（同一故障不同任务域）

命令行用法:
    python -m scripts.make_figs_ch4          # 完整模式：生成所有图表
    python -m scripts.make_figs_ch4 --quick  # 快速模式：只生成代表性图表

输出目录:
    outputs/ch4/figures/
    outputs/ch4/tables/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

# Import plotting setup
from src.plots.plotting import setup_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第四章图表统一生成入口"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：只生成代表性图表，减少运行时间",
    )
    parser.add_argument(
        "--skip-nominal",
        action="store_true",
        help="跳过名义轨迹仿真",
    )
    parser.add_argument(
        "--skip-scvx",
        action="store_true",
        help="跳过 SCvx 收敛图（需要先运行 run_scvx_demo 生成日志）",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="跳过自适应权重图",
    )
    parser.add_argument(
        "--skip-trajectories",
        action="store_true",
        help="跳过多故障轨迹对比图",
    )
    parser.add_argument(
        "--etas",
        type=str,
        default="0.2,0.5,0.8",
        help="逗号分隔的 η 列表，默认 0.2,0.5,0.8",
    )
    return parser.parse_args()


def generate_nominal_trajectory_figs() -> None:
    """生成名义轨迹仿真图（4.1节）。"""
    print("\n" + "=" * 60)
    print("[1] 生成名义轨迹仿真图 (4.1)")
    print("=" * 60)

    from src.sim.run_nominal import simulate_full_mission, NominalResult
    from src.plots.plotting import plot_time_series, save_figure
    import matplotlib.pyplot as plt

    figs_dir = PROJECT_ROOT / "outputs" / "ch4" / "figures" / "ch4_nominal"
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("运行名义仿真...")
    nominal: NominalResult = simulate_full_mission(dt=1.0, save_csv=True)

    # 高度曲线
    plot_time_series(
        nominal.time, nominal.altitude_km,
        title="名义轨迹：高度 vs 时间",
        ylabel="高度 / km",
        outfile=figs_dir / "altitude_vs_time.png",
    )
    print(f"  - 高度曲线已保存: {figs_dir / 'altitude_vs_time.png'}")

    # 速度曲线
    plot_time_series(
        nominal.time, nominal.speed_kms,
        title="名义轨迹：速度 vs 时间",
        ylabel="速度 / km/s",
        outfile=figs_dir / "speed_vs_time.png",
    )
    print(f"  - 速度曲线已保存: {figs_dir / 'speed_vs_time.png'}")

    # 动压曲线
    plot_time_series(
        nominal.time, nominal.dynamic_pressure_kpa,
        title="名义轨迹：动压 vs 时间",
        ylabel="动压 / kPa",
        outfile=figs_dir / "dynamic_pressure_vs_time.png",
        constraint=(55.0, "$q_{max}$=55 kPa"),
    )
    print(f"  - 动压曲线已保存: {figs_dir / 'dynamic_pressure_vs_time.png'}")

    # 过载曲线
    plot_time_series(
        nominal.time, nominal.normal_load_g,
        title="名义轨迹：法向过载 vs 时间",
        ylabel="法向过载 / g",
        outfile=figs_dir / "normal_load_vs_time.png",
        constraint=(3.5, "$n_{max}$=3.5 g"),
    )
    print(f"  - 过载曲线已保存: {figs_dir / 'normal_load_vs_time.png'}")

    print("名义轨迹图生成完成。")


def generate_scvx_convergence_figs() -> None:
    """生成 SCvx 收敛图（4.2节）。"""
    print("\n" + "=" * 60)
    print("[2] 生成 SCvx 收敛图 (4.2)")
    print("=" * 60)

    # 检查日志文件是否存在
    log_path = PROJECT_ROOT / "outputs" / "ch4" / "tables" / "scvx_convergence_log.csv"
    if not log_path.exists():
        print(f"[WARN] SCvx 收敛日志不存在: {log_path}")
        print("       请先运行 python -m scripts.run_scvx_demo --scvx --save-log outputs/ch4/tables/scvx_convergence_log.csv")
        print("       跳过 SCvx 收敛图生成。")
        return

    # 调用现有脚本
    try:
        from scripts.make_figs_ch4_scvx_convergence import main as scvx_main
        scvx_main()
    except Exception as e:
        print(f"[WARN] SCvx 收敛图生成失败: {e}")


def generate_adaptive_weights_figs() -> None:
    """生成自适应权重图（4.3节）。"""
    print("\n" + "=" * 60)
    print("[3] 生成自适应权重图 (4.3)")
    print("=" * 60)

    try:
        from scripts.make_figs_ch4_adaptive_weights import main as weights_main
        weights_main()
    except Exception as e:
        print(f"[WARN] 自适应权重图生成失败: {e}")


def generate_warmstart_stats_figs() -> None:
    """生成学习热启动统计对比图（4.3节）。"""
    print("\n" + "=" * 60)
    print("[3+] 生成学习热启动统计对比图 (4.3)")
    print("=" * 60)

    # 检查性能数据文件是否存在
    perf_csv = PROJECT_ROOT / "outputs" / "ch4" / "tables" / "ch4_warmstart_performance.csv"
    if not perf_csv.exists():
        print(f"[WARN] 热启动性能数据不存在: {perf_csv}")
        print("       请先运行 python -m scripts.eval_ch4_warmstart_performance")
        print("       跳过热启动统计图生成。")
        return

    try:
        from scripts.make_figs_ch4_warmstart_stats import main as warmstart_stats_main
        warmstart_stats_main()
    except Exception as e:
        print(f"[WARN] 热启动统计图生成失败: {e}")


def generate_warmstart_case_figs(dry_run: bool = False) -> None:
    """生成学习热启动单案例对比图（4.3节）。

    Parameters
    ----------
    dry_run : bool
        若为 True，则以 dry-run 模式运行，仅验证脚本导入而不执行 SCvx 求解
    """
    print("\n" + "=" * 60)
    print("[3++] 生成学习热启动单案例对比图 (4.3)")
    print("=" * 60)

    try:
        # 动态修改 sys.argv 以传递 --dry-run 参数
        import sys
        original_argv = sys.argv.copy()

        if dry_run:
            print("[INFO] 使用 dry-run 模式（仅验证脚本导入）")
            sys.argv = ["make_figs_ch4_warmstart_case", "--dry-run"]
        else:
            sys.argv = ["make_figs_ch4_warmstart_case"]

        from scripts.make_figs_ch4_warmstart_case import main as warmstart_case_main
        warmstart_case_main()

        # 恢复原始 argv
        sys.argv = original_argv
    except Exception as e:
        print(f"[WARN] 热启动单案例对比图生成失败: {e}")
        # 恢复原始 argv（异常情况下也要恢复）
        sys.argv = original_argv


def generate_fault_trajectory_figs(
    fault_ids: List[str],
    etas: List[float],
    quick: bool = False,
) -> None:
    """生成多故障多严重度轨迹对比图（4.4节）。"""
    print("\n" + "=" * 60)
    print("[4] 生成多故障轨迹对比图 (4.4)")
    print("=" * 60)

    from scripts.make_figs_ch4_fault_trajectories import (
        generate_fault_trajectory_figs as _gen_figs,
    )

    output_dir = PROJECT_ROOT / "outputs" / "ch4" / "figures" / "ch4_fault_trajectories"

    if quick:
        # 快速模式：只生成 F1 + 单一 η
        print("Quick 模式：只生成 F1_thrust_deg15, η=0.5")
        _gen_figs(
            fault_ids=["F1_thrust_deg15"],
            etas=[0.5],
            output_dir=output_dir,
            t_step=1.0,
            generate_constraints=True,
        )
    else:
        # 完整模式
        _gen_figs(
            fault_ids=fault_ids,
            etas=etas,
            output_dir=output_dir,
            t_step=1.0,
            generate_constraints=True,
        )


def generate_path_constraint_figs(
    fault_ids: List[str],
    eta: float = 0.8,
) -> None:
    """为代表性工况生成路径约束对比图（4.4节补充）。"""
    print("\n" + "=" * 60)
    print("[5] 生成路径约束对比图 (4.4 补充)")
    print("=" * 60)

    from scripts.make_figs_ch4_fault_trajectories import (
        generate_path_constraint_figs as _gen_constraints,
    )

    output_dir = PROJECT_ROOT / "outputs" / "ch4" / "figures" / "ch4_fault_trajectories"

    print(f"为所有故障场景生成 η={eta:.2f} 的约束图...")
    _gen_constraints(
        fault_ids=fault_ids,
        eta=eta,
        output_dir=output_dir,
        t_step=1.0,
    )


def generate_mission_domain_stats_figs() -> None:
    """生成任务域分布统计图（4.4节）。"""
    print("\n" + "=" * 60)
    print("[6] 生成任务域分布统计图 (4.4)")
    print("=" * 60)

    # 检查数据文件是否存在
    csv_path = PROJECT_ROOT / "outputs" / "ch4" / "tables" / "ch4_mission_domains.csv"
    if not csv_path.exists():
        print(f"[WARN] 任务域数据不存在: {csv_path}")
        print("       请先运行 python -m scripts.eval_ch4_mission_domains")
        print("       跳过任务域统计图生成。")
        return

    try:
        from scripts.make_figs_ch4_mission_domains_stats import main as domain_stats_main
        domain_stats_main()
    except Exception as e:
        print(f"[WARN] 任务域统计图生成失败: {e}")


def generate_mission_domain_trajectory_figs(dry_run: bool = False) -> None:
    """生成三任务域轨迹对比图（4.4节）。

    Parameters
    ----------
    dry_run : bool
        若为 True，则以 dry-run 模式运行，仅验证脚本导入而不执行 SCvx 求解
    """
    print("\n" + "=" * 60)
    print("[7] 生成三任务域轨迹对比图 (4.4)")
    print("=" * 60)

    try:
        # 动态修改 sys.argv 以传递 --dry-run 参数
        import sys
        original_argv = sys.argv.copy()

        if dry_run:
            print("[INFO] 使用 dry-run 模式（仅验证脚本导入）")
            sys.argv = ["make_figs_ch4_mission_domain_trajectories", "--dry-run"]
        else:
            # 默认参数：F1_thrust_deg15, eta=0.8, 包含约束图
            sys.argv = [
                "make_figs_ch4_mission_domain_trajectories",
                "--fault-id", "F1_thrust_deg15",
                "--eta", "0.8",
                "--include-constraints",
            ]

        from scripts.make_figs_ch4_mission_domain_trajectories import main as domain_traj_main
        domain_traj_main()

        # 恢复原始 argv
        sys.argv = original_argv
    except Exception as e:
        print(f"[WARN] 三任务域轨迹对比图生成失败: {e}")
        # 恢复原始 argv（异常情况下也要恢复）
        sys.argv = original_argv


def main() -> None:
    """主函数。"""
    args = parse_args()

    setup_matplotlib()

    print("#" * 80)
    print("# 第四章图表统一生成")
    print("# 模式:", "Quick" if args.quick else "Full")
    print("#" * 80)

    # 默认故障场景
    from src.sim.viz_trajectories import get_default_fault_ids
    fault_ids = get_default_fault_ids()
    etas = [float(e.strip()) for e in args.etas.split(",")]

    # 1. 名义轨迹图
    if not args.skip_nominal:
        generate_nominal_trajectory_figs()

    # 2. SCvx 收敛图
    if not args.skip_scvx:
        generate_scvx_convergence_figs()

    # 3. 自适应权重图
    if not args.skip_weights:
        generate_adaptive_weights_figs()

    # 3+. 学习热启动统计图（完整模式和快速模式都生成）
    generate_warmstart_stats_figs()

    # 3++. 学习热启动单案例对比图（快速模式下用 dry-run，完整模式下真实求解）
    if args.quick:
        generate_warmstart_case_figs(dry_run=True)
    else:
        generate_warmstart_case_figs(dry_run=False)

    # 4. 多故障轨迹对比图
    if not args.skip_trajectories:
        generate_fault_trajectory_figs(
            fault_ids=fault_ids,
            etas=etas,
            quick=args.quick,
        )

        # 5. 路径约束对比图（重度工况）
        if not args.quick:
            generate_path_constraint_figs(
                fault_ids=fault_ids,
                eta=0.8,
            )

    # 6. 任务域分布统计图（完整模式和快速模式都生成）
    generate_mission_domain_stats_figs()

    # 7. 三任务域轨迹对比图（快速模式下用 dry-run，完整模式下真实求解）
    if args.quick:
        generate_mission_domain_trajectory_figs(dry_run=True)
    else:
        generate_mission_domain_trajectory_figs(dry_run=False)

    print("\n" + "#" * 80)
    print("# 第四章图表生成完成！")
    print("#" * 80)
    print(f"输出目录: {PROJECT_ROOT / 'outputs' / 'ch4'}")


if __name__ == "__main__":
    main()
