#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一生成论文第四章所需的所有数据表（CSV + Markdown）。

本脚本封装调用各个评估脚本，生成：
- 学习热启动性能对比表（CSV + Markdown）
- 任务域评估表（CSV + Markdown）
- 自适应权重配置表（Markdown）
- SCvx 收敛统计表（Markdown）

命令行用法：
    python -m scripts.make_tables_ch4                    # 生成所有表格
    python -m scripts.make_tables_ch4 --skip-warmstart   # 跳过热启动表
    python -m scripts.make_tables_ch4 --skip-domains     # 跳过任务域表

输出目录：
    outputs/tables/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一生成第四章所有数据表"
    )
    parser.add_argument(
        "--skip-warmstart",
        action="store_true",
        help="跳过学习热启动性能评估表",
    )
    parser.add_argument(
        "--skip-domains",
        action="store_true",
        help="跳过任务域评估表",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="跳过自适应权重配置表",
    )
    parser.add_argument(
        "--skip-scvx",
        action="store_true",
        help="跳过 SCvx 收敛统计表",
    )
    return parser.parse_args()


def generate_warmstart_tables() -> None:
    """生成学习热启动性能对比表。

    调用 eval_ch4_warmstart_performance.py 主函数。

    输出文件：
    - ch4_warmstart_performance.csv
    - table_ch4_warmstart_performance.md
    """
    print("\n" + "=" * 60)
    print("[1] 生成学习热启动性能对比表")
    print("=" * 60)

    try:
        from scripts.eval_ch4_warmstart_performance import main as warmstart_main

        # 修改 sys.argv 以传递默认参数（如 --repeat 3）
        original_argv = sys.argv.copy()
        sys.argv = ["eval_ch4_warmstart_performance", "--repeat", "3"]

        warmstart_main()

        # 恢复原始 argv
        sys.argv = original_argv
    except Exception as e:
        print(f"[WARN] 学习热启动性能表生成失败: {e}")
        # 恢复原始 argv（异常情况下也要恢复）
        sys.argv = original_argv


def generate_mission_domain_tables() -> None:
    """生成任务域评估表。

    调用 eval_ch4_mission_domains.py 主函数。

    输出文件：
    - ch4_mission_domains.csv
    - table_ch4_mission_domains.md
    """
    print("\n" + "=" * 60)
    print("[2] 生成任务域评估表")
    print("=" * 60)

    try:
        from scripts.eval_ch4_mission_domains import main as domains_main

        # 修改 sys.argv 以传递默认参数
        original_argv = sys.argv.copy()
        sys.argv = [
            "eval_ch4_mission_domains",
            "--eta-values", "0.2,0.5,0.8",
            "--enable-escalation",
        ]

        domains_main()

        # 恢复原始 argv
        sys.argv = original_argv
    except Exception as e:
        print(f"[WARN] 任务域评估表生成失败: {e}")
        # 恢复原始 argv（异常情况下也要恢复）
        sys.argv = original_argv


def generate_adaptive_weights_table() -> None:
    """生成自适应权重配置表。

    调用 make_figs_ch4_adaptive_weights.py 主函数（它会同时生成图和表）。

    输出文件：
    - table_ch4_adaptive_weights.md
    """
    print("\n" + "=" * 60)
    print("[3] 生成自适应权重配置表")
    print("=" * 60)

    try:
        from scripts.make_figs_ch4_adaptive_weights import main as weights_main
        weights_main()
    except Exception as e:
        print(f"[WARN] 自适应权重配置表生成失败: {e}")


def generate_scvx_stats_table() -> None:
    """生成 SCvx 收敛统计表。

    调用 make_figs_ch4_scvx_convergence.py 主函数（它会同时生成图和表）。

    输出文件：
    - table_scvx_stats.md
    """
    print("\n" + "=" * 60)
    print("[4] 生成 SCvx 收敛统计表")
    print("=" * 60)

    # 检查日志文件是否存在
    log_path = TABLES_DIR / "scvx_convergence_log.csv"
    if not log_path.exists():
        print(f"[WARN] SCvx 收敛日志不存在: {log_path}")
        print("       请先运行 python -m scripts.run_scvx_demo --scvx --save-log outputs/ch4/tables/scvx_convergence_log.csv")
        print("       跳过 SCvx 收敛统计表生成。")
        return

    try:
        from scripts.make_figs_ch4_scvx_convergence import main as scvx_main
        scvx_main()
    except Exception as e:
        print(f"[WARN] SCvx 收敛统计表生成失败: {e}")


def main() -> None:
    """主函数：统一生成所有第四章数据表。"""
    args = parse_args()

    print("#" * 80)
    print("# 第四章数据表统一生成")
    print("#" * 80)
    print(f"输出目录: {TABLES_DIR}")
    print()

    # 1. 学习热启动性能表
    if not args.skip_warmstart:
        generate_warmstart_tables()

    # 2. 任务域评估表
    if not args.skip_domains:
        generate_mission_domain_tables()

    # 3. 自适应权重配置表
    if not args.skip_weights:
        generate_adaptive_weights_table()

    # 4. SCvx 收敛统计表
    if not args.skip_scvx:
        generate_scvx_stats_table()

    print("\n" + "#" * 80)
    print("# 第四章数据表生成完成！")
    print("#" * 80)
    print(f"输出目录: {TABLES_DIR}")
    print()
    print("生成的表格文件：")
    print("  CSV:")
    print("    - ch4_warmstart_performance.csv")
    print("    - ch4_mission_domains.csv")
    print("    - scvx_convergence_log.csv（需先运行 run_scvx_demo）")
    print("  Markdown:")
    print("    - table_ch4_warmstart_performance.md")
    print("    - table_ch4_mission_domains.md")
    print("    - table_ch4_adaptive_weights.md")
    print("    - table_scvx_stats.md")
    print()
    print("提示：可运行 python -m scripts.make_index_ch4_figs_tables 生成图表索引")


if __name__ == "__main__":
    main()
