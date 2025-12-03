#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四章：一键生成所有图表和数据的总入口脚本。

本脚本按照论文第四章的组织结构，统一调度以下子脚本：
1. 学习模型训练（可选，耗时较长）
2. 学习曲线与预测效果图（图4-10~4-14）
3. 热启动性能评估表
4. 任务域评估表
5. 自适应权重配置表与图
6. SCvx 收敛分析图与表
7. 故障轨迹对比图（可选，需要 cvxpy）
8. 图表索引文件

输出目录结构：
    outputs/
    ├── data/           # 中间数据（CSV, NPZ, JSON）
    │   └── ch4_learning/   # 学习模型相关数据
    ├── figures/        # 所有图像（PNG + PDF）
    │   ├── ch4_learning/   # 学习曲线与预测效果图
    │   ├── eta_scaling/    # Eta 缩放分析图
    │   ├── domain_comparison/  # 任务域对比图
    │   └── trajectories/   # 轨迹相关图
    ├── tables/         # 所有 Markdown 表格
    └── INDEX.md        # 图表索引文件

命令行用法：
    python -m scripts.generate_ch4_outputs           # 生成所有输出
    python -m scripts.generate_ch4_outputs --skip-train  # 跳过模型训练
    python -m scripts.generate_ch4_outputs --skip-scvx   # 跳过 SCvx 相关（需 cvxpy）
    python -m scripts.generate_ch4_outputs --quick       # 快速模式（跳过耗时步骤）
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# 路径设置
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# 输出目录
OUTPUT_BASE = ROOT / "outputs"
DATA_DIR = OUTPUT_BASE / "data"
FIG_DIR = OUTPUT_BASE / "figures"
TABLE_DIR = OUTPUT_BASE / "tables"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="第四章：一键生成所有图表和数据"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过学习模型训练（假设已有训练好的模型）",
    )
    parser.add_argument(
        "--skip-scvx",
        action="store_true",
        help="跳过需要 cvxpy/SCvx 的步骤",
    )
    parser.add_argument(
        "--skip-warmstart",
        action="store_true",
        help="跳过热启动性能评估",
    )
    parser.add_argument(
        "--skip-domains",
        action="store_true",
        help="跳过任务域评估",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：跳过训练和 SCvx 相关步骤",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default=None,
        help="运行模式：quick=跳过耗时步骤, full=运行所有步骤包括新增脚本",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    """确保输出目录存在。"""
    for d in [DATA_DIR, FIG_DIR, TABLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def step_train_learning_model() -> bool:
    """步骤1：训练学习热启动模型。

    分为两个子步骤：
    1a. 生成数据集（如果不存在）：调用 gen_ch4_learning_dataset.py
    1b. 训练模型（如果模型不存在）：调用 train_ch4_learning.py

    输出：
    - outputs/data/ch4_learning/dataset.npz
    - outputs/data/ch4_learning/model.pt
    - outputs/data/ch4_learning/train_log.json
    - outputs/data/ch4_learning/feature_stats.json
    """
    print("\n" + "=" * 60)
    print("步骤 1: 训练学习热启动模型")
    print("=" * 60)

    dataset_path = DATA_DIR / "ch4_learning" / "dataset.npz"
    model_path = DATA_DIR / "ch4_learning" / "model.pt"
    train_log_path = DATA_DIR / "ch4_learning" / "train_log.json"

    # 1a. 检查数据集是否存在，不存在则生成
    if not dataset_path.exists():
        print("\n[1a] 数据集不存在，正在生成...")
        try:
            from scripts.gen_ch4_learning_dataset import main as gen_dataset_main
            original_argv = sys.argv.copy()
            sys.argv = [
                "gen_ch4_learning_dataset",
                "--samples", "200",
                "--output", str(dataset_path),
            ]
            gen_dataset_main()
            sys.argv = original_argv
        except Exception as e:
            print(f"[WARN] 数据集生成失败: {e}")
            return False
    else:
        print(f"\n[1a] 数据集已存在: {dataset_path}")

    # 1b. 检查模型是否存在，不存在则训练
    if not model_path.exists() or not train_log_path.exists():
        print("\n[1b] 模型不存在，正在训练...")
        try:
            from scripts.train_ch4_learning import main as train_main
            original_argv = sys.argv.copy()
            sys.argv = [
                "train_ch4_learning",
                "--dataset", str(dataset_path),
                "--output-dir", str(DATA_DIR / "ch4_learning"),
                "--epochs", "200",
            ]
            train_main()
            sys.argv = original_argv
        except Exception as e:
            print(f"[WARN] 模型训练失败: {e}")
            return False
    else:
        print(f"\n[1b] 模型已存在: {model_path}")

    return True


def step_generate_learning_curves() -> bool:
    """步骤2：生成学习曲线图（图4-10）。

    输出：
    - outputs/figures/ch4_learning/fig4_10_learning_curves.png/.pdf
    """
    print("\n" + "=" * 60)
    print("步骤 2: 生成学习曲线图 (图4-10)")
    print("=" * 60)

    try:
        from scripts.make_figs_ch4_learning_curves import main as curves_main
        curves_main()
        return True
    except Exception as e:
        print(f"[WARN] 学习曲线图生成失败: {e}")
        return False


def step_generate_altitude_prediction_figs() -> bool:
    """步骤3：生成高度预测效果图（图4-11~4-14）。

    输出：
    - outputs/figures/ch4_learning/fig4_11_single_sample_altitude.png/.pdf
    - outputs/figures/ch4_learning/fig4_12_fault_cases_altitude.png/.pdf
    - outputs/figures/ch4_learning/fig4_13_mean_node_error.png/.pdf
    - outputs/figures/ch4_learning/fig4_14_rmse_hist.png/.pdf
    """
    print("\n" + "=" * 60)
    print("步骤 3: 生成高度预测效果图 (图4-11~4-14)")
    print("=" * 60)

    try:
        from scripts.make_figs_ch4_learning_altitude import main as altitude_main
        altitude_main()
        return True
    except Exception as e:
        print(f"[WARN] 高度预测效果图生成失败: {e}")
        return False


def step_generate_warmstart_tables() -> bool:
    """步骤4：生成热启动性能评估表。

    输出：
    - outputs/data/ch4_warmstart_performance.csv
    - outputs/tables/table_ch4_warmstart_performance.md
    """
    print("\n" + "=" * 60)
    print("步骤 4: 生成热启动性能评估表")
    print("=" * 60)

    try:
        from scripts.eval_ch4_warmstart_performance import main as warmstart_main

        # 保存原始 argv 并设置参数
        original_argv = sys.argv.copy()
        sys.argv = ["eval_ch4_warmstart_performance", "--repeat", "3"]

        warmstart_main()

        sys.argv = original_argv
        return True
    except Exception as e:
        print(f"[WARN] 热启动性能评估失败: {e}")
        # 恢复 argv
        if 'original_argv' in locals():
            sys.argv = original_argv
        return False


def step_generate_domain_tables() -> bool:
    """步骤5：生成任务域评估表。

    从已有轨迹数据读取，根据 eta 值判定任务域。

    输出：
    - outputs/ch4/tables/ch4_mission_domains.csv
    - outputs/ch4/tables/table_ch4_mission_domains.md
    """
    print("\n" + "=" * 60)
    print("步骤 5: 生成任务域评估表")
    print("=" * 60)

    try:
        from scripts.eval_ch4_mission_domains import main as domains_main

        original_argv = sys.argv.copy()
        sys.argv = [
            "eval_ch4_mission_domains",
            "--eta-values", "0.2,0.5,0.8",
        ]

        domains_main()

        sys.argv = original_argv
        return True
    except Exception as e:
        print(f"[WARN] 任务域评估失败: {e}")
        if 'original_argv' in locals():
            sys.argv = original_argv
        return False


def step_generate_adaptive_weights() -> bool:
    """步骤6：生成自适应权重配置表与图。

    输出：
    - outputs/tables/table_ch4_adaptive_weights.md
    - outputs/figures/eta_scaling/adaptive_weights_vs_eta.png
    """
    print("\n" + "=" * 60)
    print("步骤 6: 生成自适应权重配置表与图")
    print("=" * 60)

    try:
        from scripts.make_figs_ch4_adaptive_weights import main as weights_main
        weights_main()
        return True
    except Exception as e:
        print(f"[WARN] 自适应权重配置生成失败: {e}")
        return False


def step_generate_scvx_convergence() -> bool:
    """步骤7：生成 SCvx 收敛分析图与表。

    输出：
    - outputs/figures/ch4_scvx_convergence/...
    - outputs/tables/table_scvx_stats.md
    """
    print("\n" + "=" * 60)
    print("步骤 7: 生成 SCvx 收敛分析图与表")
    print("=" * 60)

    # 检查日志文件是否存在，如果不存在则先生成
    log_path = DATA_DIR / "scvx_convergence_log.csv"
    if not log_path.exists():
        print(f"[INFO] SCvx 收敛日志不存在: {log_path}")
        print("       正在生成收敛日志...")

        try:
            # 尝试运行收敛评估脚本生成日志
            from scripts.eval_ch4_scvx_convergence import main as eval_scvx_main
            original_argv = sys.argv.copy()
            sys.argv = [
                "eval_ch4_scvx_convergence",
                "--scenarios", "F1_thrust_deg15",
                "--etas", "0.5",
                "--solver-profile", "fast",
            ]
            eval_scvx_main()
            sys.argv = original_argv
        except Exception as e:
            print(f"[WARN] 无法生成 SCvx 收敛日志: {e}")
            print("       跳过此步骤。")
            return False

    if not log_path.exists():
        print(f"[WARN] 收敛日志仍然不存在，跳过绘图步骤")
        return False

    try:
        from scripts.make_figs_ch4_scvx_convergence import main as scvx_main
        scvx_main()
        return True
    except Exception as e:
        print(f"[WARN] SCvx 收敛分析生成失败: {e}")
        return False


def step_generate_eta_scaling_figs() -> bool:
    """步骤8：生成 Eta 缩放分析图与表。

    输出：
    - outputs/figures/eta_scaling/eta_scaling_all_faults.png
    - outputs/tables/eta_scaling_parameters.md
    """
    print("\n" + "=" * 60)
    print("步骤 8: 生成 Eta 缩放分析图与表")
    print("=" * 60)

    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 配置 matplotlib
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150

        from src.sim.scenarios import get_scenario, scale_scenario_by_eta, SCENARIO_CATALOG

        fig_dir = FIG_DIR / "eta_scaling"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 生成图
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle('故障严重度 eta 对故障参数的���响', fontsize=14)

        etas = np.linspace(0, 1, 21)

        # F1-F5 各故障参数
        plot_configs = [
            ("F1_thrust_deg15", "degrade_frac", "推力降级比例", "F1: 推力降级"),
            ("F2_tvc_rate4", "tvc_rate_deg_s", "TVC速率限制 (°/s)", "F2: TVC速率限制"),
            ("F2_tvc_rate4", "angle_bias_deg", "角度偏置 (°)", "F2: 角度偏置"),
            ("F3_tvc_stuck3deg", "stuck_angle_deg", "卡滞角度 (°)", "F3: TVC卡滞"),
            ("F4_sensor_bias2deg", "sensor_bias_deg", "传感器偏置 (°)", "F4: 传感器偏置"),
            ("F5_event_delay5s", "event_delay_s", "事件延迟 (s)", "F5: 事件延迟"),
        ]

        colors = ['b', 'r', 'g', 'm', 'c', 'orange']

        for idx, (scenario_id, param, ylabel, title) in enumerate(plot_configs):
            ax = axes.flatten()[idx]
            base = get_scenario(scenario_id)
            values = [scale_scenario_by_eta(base, e).params.get(param, 0) for e in etas]
            ax.plot(etas, values, color=colors[idx], linestyle='-', linewidth=2)
            ax.set_xlabel('eta (故障严重度)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)

        plt.tight_layout()

        fig_path = fig_dir / "eta_scaling_all_faults.png"
        fig.savefig(fig_path, bbox_inches='tight')
        fig.savefig(fig_dir / "eta_scaling_all_faults.pdf", bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {fig_path}")

        # 生成表格
        rows = []
        rows.append("# Eta 缩放参数表\n")
        rows.append("| 故障场景 | 参数 | eta=0.0 | eta=0.25 | eta=0.5 | eta=0.75 | eta=1.0 |")
        rows.append("|----------|------|---------|----------|---------|----------|---------|")

        eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        for scenario_id, param, _, _ in plot_configs:
            base = get_scenario(scenario_id)
            values = [f"{scale_scenario_by_eta(base, e).params.get(param, 0):.3f}" for e in eta_values]
            rows.append(f"| {scenario_id} | {param} | {' | '.join(values)} |")

        table_path = TABLE_DIR / "eta_scaling_parameters.md"
        table_path.write_text("\n".join(rows), encoding='utf-8')
        print(f"  Saved: {table_path}")

        return True
    except Exception as e:
        print(f"[WARN] Eta 缩放分析生成失败: {e}")
        return False


def step_generate_domain_comparison_figs() -> bool:
    """步骤9：生成任务域对比图。

    输出：
    - outputs/figures/domain_comparison/domain_configuration_comparison.png
    - outputs/tables/mission_domain_config.md
    """
    print("\n" + "=" * 60)
    print("步骤 9: 生成任务域对比图")
    print("=" * 60)

    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 150

        from src.sim.mission_domains import MissionDomain, default_domain_config

        fig_dir = FIG_DIR / "domain_comparison"
        fig_dir.mkdir(parents=True, exist_ok=True)

        domains = [MissionDomain.RETAIN, MissionDomain.DEGRADED, MissionDomain.SAFE_AREA]
        domain_names_cn = ['保持入轨', '降级任务', '安全区域']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左图：权重缩放
        ax = axes[0]
        x = np.arange(len(domains))
        width = 0.2

        configs = [default_domain_config(d) for d in domains]
        terminal_scales = [c.terminal_weight_scale for c in configs]
        state_scales = [c.state_weight_scale for c in configs]
        control_scales = [c.control_weight_scale for c in configs]
        slack_scales = [c.slack_weight_scale for c in configs]

        ax.bar(x - 1.5*width, terminal_scales, width, label='终端权重', color='blue')
        ax.bar(x - 0.5*width, state_scales, width, label='状态权重', color='green')
        ax.bar(x + 0.5*width, control_scales, width, label='控制权重', color='orange')
        ax.bar(x + 1.5*width, slack_scales, width, label='松弛权重', color='red')

        ax.set_xlabel('任务域')
        ax.set_ylabel('权重缩放系数')
        ax.set_title('不同任务域的权重缩放')
        ax.set_xticks(x)
        ax.set_xticklabels(domain_names_cn)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 右图：是否追求入轨
        ax = axes[1]
        require_orbit = [1 if c.require_orbit else 0 for c in configs]
        colors = ['green' if r else 'red' for r in require_orbit]
        bars = ax.bar(domain_names_cn, require_orbit, color=colors)

        ax.set_xlabel('任务域')
        ax.set_ylabel('是否追求入轨')
        ax.set_title('终端目标: 是否追求轨道插入')
        ax.set_ylim(0, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['否', '是'])

        for bar, r in zip(bars, require_orbit):
            label = '追求入轨' if r else '放弃入轨'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    label, ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        fig_path = fig_dir / "domain_configuration_comparison.png"
        fig.savefig(fig_path, bbox_inches='tight')
        fig.savefig(fig_dir / "domain_configuration_comparison.pdf", bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {fig_path}")

        # 生成表格
        rows = []
        rows.append("# 任务域配置表\n")
        rows.append("| 任务域 | 追求入轨 | 终端权重缩放 | 状态权重缩放 | 控制权重缩放 | 松弛权重缩放 |")
        rows.append("|--------|----------|--------------|--------------|--------------|--------------|")

        for domain, cfg in zip(domains, configs):
            require = "是" if cfg.require_orbit else "否"
            rows.append(f"| {domain.name} | {require} | {cfg.terminal_weight_scale:.1f} | {cfg.state_weight_scale:.1f} | {cfg.control_weight_scale:.1f} | {cfg.slack_weight_scale:.1f} |")

        rows.append("\n## 基于 eta 的任务域选择\n")
        rows.append("| eta 范围 | 选择的任务域 |")
        rows.append("|----------|-------------|")
        rows.append("| 0.0 - 0.3 | RETAIN (保持入轨) |")
        rows.append("| 0.3 - 0.7 | DEGRADED (降级任务) |")
        rows.append("| 0.7 - 1.0 | SAFE_AREA (安全区域) |")

        table_path = TABLE_DIR / "mission_domain_config.md"
        table_path.write_text("\n".join(rows), encoding='utf-8')
        print(f"  Saved: {table_path}")

        return True
    except Exception as e:
        print(f"[WARN] 任务域对比图生成失败: {e}")
        return False


# ============================================================================
# 新增步骤 (full 模式专用)
# ============================================================================

def step_generate_core_tables() -> bool:
    """步骤11 [Full]: 生成核心表格（表4-2/3/4）。

    输出：
    - outputs/tables/table4_02_scvx_dimension_stats.md
    - outputs/tables/table4_03_network_hyperparams.md
    - outputs/tables/table4_04_mission_domain_mapping.md
    """
    print("\n" + "=" * 60)
    print("步骤 11 [Full]: 生成核心表格 (表4-2/3/4)")
    print("=" * 60)

    try:
        from scripts.make_tables_ch4_core import main as core_tables_main
        core_tables_main()
        return True
    except Exception as e:
        print(f"[WARN] 核心表格生成失败: {e}")
        return False


def step_generate_warmstart_performance_figs() -> bool:
    """步骤12 [Full]: 生成热启动性能图（图4-16/17/18）。

    输出：
    - outputs/figures/ch4_warmstart/fig4_16_warmstart_iterations.png
    - outputs/figures/ch4_warmstart/fig4_17_warmstart_time.png
    - outputs/figures/ch4_warmstart/fig4_18_terminal_rmse_distribution.png
    """
    print("\n" + "=" * 60)
    print("步骤 12 [Full]: 生成热启动性能图 (图4-16/17/18)")
    print("=" * 60)

    try:
        from scripts.make_figs_ch4_warmstart_performance import main as warmstart_figs_main
        warmstart_figs_main()
        return True
    except Exception as e:
        print(f"[WARN] 热启动性能图生成失败: {e}")
        return False


def step_generate_mission_domain_figs() -> bool:
    """步骤13 [Full]: 生成任务域响应图（图4-22/23/24）。

    输出：
    - outputs/figures/ch4_mission_domains/fig4_22_retain_domain_responses.png
    - outputs/figures/ch4_mission_domains/fig4_23_f1_severe_downrange.png
    - outputs/figures/ch4_mission_domains/fig4_24_severe_downrange_summary.png
    """
    print("\n" + "=" * 60)
    print("步骤 13 [Full]: 生成任务域响应图 (图4-22/23/24)")
    print("=" * 60)

    try:
        from scripts.make_figs_ch4_mission_domains import main as domain_figs_main
        domain_figs_main()
        return True
    except Exception as e:
        print(f"[WARN] 任务域响应图生成失败: {e}")
        return False


def step_generate_trajectory_replan_figs() -> bool:
    """步骤13b [Full]: 生成轨迹重规划对比图（图4-25~30）。

    输出：
    - outputs/figures/ch4_trajectories_replan/fig4_25_F1_trajectory_replan.png
    - outputs/figures/ch4_trajectories_replan/fig4_26_F2_trajectory_replan.png
    - outputs/figures/ch4_trajectories_replan/fig4_27_F3_trajectory_replan.png
    - outputs/figures/ch4_trajectories_replan/fig4_28_F4_trajectory_replan.png
    - outputs/figures/ch4_trajectories_replan/fig4_29_F5_trajectory_replan.png
    - outputs/figures/ch4_trajectories_replan/fig4_30_all_faults_summary.png
    """
    print("\n" + "=" * 60)
    print("步骤 13b [Full]: 生成轨迹重规划对比图 (图4-25~30)")
    print("=" * 60)

    try:
        from scripts.make_figs_ch4_trajectories_replan import main as replan_figs_main
        replan_figs_main()
        return True
    except Exception as e:
        print(f"[WARN] 轨迹重规划对比图生成失败: {e}")
        return False


def step_generate_spec_manifest() -> bool:
    """步骤14 [Full]: 生成图表规范清单。

    输出：
    - outputs/tables/ch4_figure_table_spec.md
    """
    print("\n" + "=" * 60)
    print("步骤 14 [Full]: 生成图表规范清单")
    print("=" * 60)

    try:
        from scripts.spec_ch4_fig_table_manifest import main as spec_main
        spec_main()
        return True
    except Exception as e:
        print(f"[WARN] 图表规范清单生成失败: {e}")
        return False


def step_generate_coverage_report() -> bool:
    """步骤15 [Full]: 生成覆盖率检查报告。

    输出：
    - outputs/tables/ch4_figure_table_coverage_report.md
    """
    print("\n" + "=" * 60)
    print("步骤 15 [Full]: 生成覆盖率检查报告")
    print("=" * 60)

    try:
        from scripts.check_ch4_fig_table_coverage import main as coverage_main
        coverage_main()
        return True
    except Exception as e:
        print(f"[WARN] 覆盖率检查报告生成失败: {e}")
        return False


def step_generate_final_status_report() -> bool:
    """步骤16 [Full]: 生成最终状态报告。

    输出：
    - outputs/tables/ch4_figure_table_final_status.md
    """
    print("\n" + "=" * 60)
    print("步骤 16 [Full]: 生成最终状态报告")
    print("=" * 60)

    try:
        from scripts.make_final_status_report import main as status_main
        status_main()
        return True
    except Exception as e:
        print(f"[WARN] 最终状态报告生成失败: {e}")
        return False


def step_generate_index() -> bool:
    """步骤10：生成图表索引文件。

    输出：
    - outputs/INDEX.md
    """
    print("\n" + "=" * 60)
    print("步骤 10: 生成图表索引文件")
    print("=" * 60)

    try:
        from scripts.make_index_ch4_figs_tables import main as index_main
        index_main()
        return True
    except Exception as e:
        print(f"[WARN] 索引文件生成失败: {e}")
        return False


def main() -> None:
    """主函数：一键生成第四章所有图表和数据。"""
    args = parse_args()

    # 处理 --mode 参数
    full_mode = False
    if args.mode == "quick":
        args.quick = True
    elif args.mode == "full":
        full_mode = True

    # 快速模式：跳过训练和 SCvx
    if args.quick:
        args.skip_train = True
        args.skip_scvx = True
        args.skip_warmstart = True
        args.skip_domains = True

    print("#" * 80)
    print("# 第四章：一键生成所有图表和数据")
    print("#" * 80)
    mode_str = "full (完整)" if full_mode else ("quick (快速)" if args.quick else "default (默认)")
    print(f"运行模式: {mode_str}")
    print(f"输出目录: {OUTPUT_BASE}")
    print(f"跳过训练: {args.skip_train}")
    print(f"跳过 SCvx: {args.skip_scvx}")
    print(f"跳过热启动评估: {args.skip_warmstart}")
    print(f"跳过任务域评估: {args.skip_domains}")
    print()

    # 确保目录存在
    ensure_directories()

    # 统计结果
    results = {}

    # 步骤1：训练学习模型
    if not args.skip_train:
        results["训练学习模型"] = step_train_learning_model()
    else:
        print("\n[跳过] 步骤 1: 训练学习模型")
        results["训练学习模型"] = "跳过"

    # 步骤2：学习曲线图
    results["学习曲线图"] = step_generate_learning_curves()

    # 步骤3：高度预测效果图
    results["高度预测效果图"] = step_generate_altitude_prediction_figs()

    # 步骤4：热启动性能评估表
    if not args.skip_warmstart and not args.skip_scvx:
        results["热启动性能表"] = step_generate_warmstart_tables()
    else:
        print("\n[跳过] 步骤 4: 热启动性能评估表")
        results["热启动性能表"] = "跳过"

    # 步骤5：任务域评估表
    if not args.skip_domains and not args.skip_scvx:
        results["任务域评估表"] = step_generate_domain_tables()
    else:
        print("\n[跳过] 步骤 5: 任务域评估表")
        results["任务域评估表"] = "跳过"

    # 步骤6：自适应权重配置
    results["自适应权重配置"] = step_generate_adaptive_weights()

    # 步骤7：SCvx 收敛分析
    if not args.skip_scvx:
        results["SCvx收敛分析"] = step_generate_scvx_convergence()
    else:
        print("\n[跳过] 步骤 7: SCvx 收敛分析")
        results["SCvx收敛分析"] = "跳过"

    # 步骤8：Eta 缩放分析
    results["Eta缩放分析"] = step_generate_eta_scaling_figs()

    # 步骤9：任务域对比图
    results["任务域对比图"] = step_generate_domain_comparison_figs()

    # ============================================================
    # Full 模式专用步骤 (步骤 11-15)
    # ============================================================
    if full_mode:
        print("\n" + "#" * 80)
        print("# [Full 模式] 执行扩展步骤")
        print("#" * 80)

        # 步骤11：核心表格
        results["核心表格 (表4-2/3/4)"] = step_generate_core_tables()

        # 步骤12：热启动性能图
        results["热启动性能图 (图4-16/17/18)"] = step_generate_warmstart_performance_figs()

        # 步骤13：任务域响应图
        results["任务域响应图 (图4-22/23/24)"] = step_generate_mission_domain_figs()

        # 步骤13b：轨迹重规划对比图
        results["轨迹重规划图 (图4-25~30)"] = step_generate_trajectory_replan_figs()

        # 步骤14：图表规范清单
        results["图表规范清单"] = step_generate_spec_manifest()

        # 步骤15：覆盖率检查报告
        results["覆盖率检查报告"] = step_generate_coverage_report()

        # 步骤16：最终状态报告
        results["最终状态报告"] = step_generate_final_status_report()

    # 步骤10：索引文件 (始终最后运行)
    results["索引文件"] = step_generate_index()

    # 打印总结
    print("\n" + "#" * 80)
    print("# 第四章图表生成完成！")
    print("#" * 80)
    print()
    print("生成结果汇总：")
    print("-" * 40)
    for step_name, result in results.items():
        if result == "跳过":
            status = "[跳过]"
        elif result is True:
            status = "[成功]"
        else:
            status = "[失败]"
        print(f"  {step_name}: {status}")

    print()
    print("输出目录结构：")
    print(f"  {OUTPUT_BASE}/")
    print(f"  ├── data/           # 中间数据")
    print(f"  ├── figures/        # 图像文件")
    print(f"  ├── tables/         # 表格文件")
    print(f"  └── INDEX.md        # 图表索引")
    print()
    print("提示：")
    print("  - 查看索引: cat outputs/INDEX.md")
    print("  - 重新生成: python -m scripts.generate_ch4_outputs")
    print("  - 快速模式: python -m scripts.generate_ch4_outputs --quick")
    print("  - 完整模式: python -m scripts.generate_ch4_outputs --mode full")


if __name__ == "__main__":
    main()
