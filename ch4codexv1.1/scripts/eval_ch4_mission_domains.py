#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四章 4.4：任务域选择与自动升级评估脚本。

从已生成的轨迹数据中读取结果，根据 eta 值判定任务域，统计选择情况。

任务域判定规则：
- eta < 0.3: RETAIN（保持入轨）
- 0.3 <= eta < 0.7: DEGRADED（降级任务）
- eta >= 0.7: SAFE_AREA（安全区域）

输出文件：
- outputs/ch4/tables/ch4_mission_domains.csv
- outputs/ch4/tables/table_ch4_mission_domains.md

命令行用法：
    python -m scripts.eval_ch4_mission_domains
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

# 数据目录
DATA_DIR = PROJECT_ROOT / "outputs" / "data" / "ch4_trajectories_replan"

# 故障类型映射
FAULT_ID_MAP: Dict[str, str] = {
    "F1": "F1_thrust_deg15",
    "F2": "F2_tvc_rate4",
    "F3": "F3_tvc_stuck3deg",
    "F4": "F4_sensor_bias2deg",
    "F5": "F5_event_delay5s",
}

# 故障名称
FAULT_NAMES: Dict[str, str] = {
    "F1": "推力降级",
    "F2": "TVC速率限制",
    "F3": "TVC卡滞",
    "F4": "传感器偏置",
    "F5": "事件延迟",
}

# 默认 eta 值
DEFAULT_ETA_VALUES = [0.2, 0.5, 0.8]


@dataclass
class TrajectoryData:
    """轨迹数据。"""
    t: np.ndarray
    downrange: np.ndarray
    altitude: np.ndarray
    eta: float = 0.0
    mission_domain: str = ""
    fault_type: str = ""
    converged: bool = True
    num_iterations: int = 0
    final_violation: float = 0.0


def determine_mission_domain(eta: float) -> str:
    """根据 eta 值判定任务域。

    规则：
    - eta < 0.3: RETAIN（保持入轨）
    - 0.3 <= eta < 0.7: DEGRADED（降级任务）
    - eta >= 0.7: SAFE_AREA（安全区域）
    """
    if eta < 0.3:
        return "RETAIN"
    elif eta < 0.7:
        return "DEGRADED"
    else:
        return "SAFE_AREA"


def get_domain_chinese_name(domain: str) -> str:
    """获取任务域的中文名称。"""
    mapping = {
        "RETAIN": "保持入轨",
        "DEGRADED": "降级任务",
        "SAFE_AREA": "安全区域",
    }
    return mapping.get(domain, domain)


def load_trajectory_npz(file_path: Path) -> Optional[TrajectoryData]:
    """加载轨迹数据。"""
    if not file_path.exists():
        return None

    try:
        data = np.load(file_path, allow_pickle=True)
        t = data.get('t', np.array([]))
        downrange = data.get('downrange', np.array([]))
        altitude = data.get('altitude', np.array([]))

        if len(t) == 0 or len(downrange) == 0 or len(altitude) == 0:
            return None

        return TrajectoryData(
            t=t,
            downrange=downrange,
            altitude=altitude,
            eta=float(data.get('eta', 0.0)),
            mission_domain=str(data.get('mission_domain', '')),
            fault_type=str(data.get('fault_type', '')),
            converged=bool(data.get('converged', True)),
            num_iterations=int(data.get('num_iterations', 0)),
            final_violation=float(data.get('final_violation', 0.0)),
        )
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估任务域选择策略（从已有数据）")
    parser.add_argument(
        "--eta-values",
        type=str,
        default="0.2,0.5,0.8",
        help="逗号分隔的 eta 测试值",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="轨迹数据目录",
    )
    return parser.parse_args()


def generate_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    """生成 Markdown 格式的任务域选择统计表。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# 表4-3：任务域选择与自动升级统计\n\n")
        f.write("基于故障严重度 η 的任务域自动判定结果。\n\n")
        f.write("**判定规则**：\n")
        f.write("- η < 0.3 → RETAIN（保持入轨）\n")
        f.write("- 0.3 ≤ η < 0.7 → DEGRADED（降级任务）\n")
        f.write("- η ≥ 0.7 → SAFE_AREA（安全区域）\n\n")
        f.write("| 故障场景 | 故障类型 | η | 任务域 | 终端行距(km) | 终端高度(km) | 状态 |\n")
        f.write("|:--------:|:--------:|:---:|:------:|:------------:|:------------:|:----:|\n")

        for _, row in df.iterrows():
            status = "✓" if row['status'] == "OK" else "—"
            f.write(
                f"| {row['fault_id']} | {row['fault_name']} | {row['eta']:.1f} | "
                f"{row['mission_domain']} | {row['terminal_downrange']:.1f} | "
                f"{row['terminal_altitude']:.1f} | {status} |\n"
            )

        # 添加统计汇总
        f.write("\n## 统计汇总\n\n")

        # 按任务域统计
        domain_counts = df['mission_domain'].value_counts()
        f.write("**各任务域分布**：\n")
        for domain, count in domain_counts.items():
            f.write(f"- {domain}（{get_domain_chinese_name(domain)}）: {count} 条轨迹\n")

        # 按故障类型统计
        f.write("\n**各故障类型终端行距统计**：\n")
        for fault_id in ["F1", "F2", "F3", "F4", "F5"]:
            fault_data = df[df['fault_id'] == fault_id]
            if len(fault_data) > 0:
                mean_dr = fault_data['terminal_downrange'].mean()
                f.write(f"- {fault_id}（{FAULT_NAMES[fault_id]}）: 平均终端行距 {mean_dr:.1f} km\n")

    print(f"  Markdown table saved to {out_path}")


def main() -> None:
    args = parse_args()
    fault_ids = ["F1", "F2", "F3", "F4", "F5"]
    eta_values = [float(x.strip()) for x in args.eta_values.split(",")]
    data_dir = Path(args.data_dir)

    records: List[dict] = []

    print("=" * 80)
    print("第四章 4.4：任务域选择与自动升级评估")
    print("=" * 80)
    print(f"数据目录: {data_dir}")
    print(f"eta 测试值: {eta_values}")
    print()

    # 检查数据目录
    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        print("请先运行: python -m scripts.eval_ch4_trajectories_replan")
        return

    print("[1/2] 读取轨迹数据并判定任务域...")
    loaded_count = 0

    for fault_id in fault_ids:
        print(f"  处理故障场景: {fault_id} ({FAULT_NAMES[fault_id]})")
        for eta in eta_values:
            # 构建文件名
            eta_str = f"eta{eta:.1f}".replace(".", "")
            replan_path = data_dir / f"{fault_id}_{eta_str}_replan.npz"

            # 尝试加载轨迹数据
            traj = load_trajectory_npz(replan_path)

            if traj is not None:
                loaded_count += 1
                # 使用 npz 中保存的任务域，若无则回退到 eta 判定
                if traj.mission_domain:
                    domain = str(traj.mission_domain).upper()
                else:
                    domain = determine_mission_domain(eta)
                # 数据已为 km，不再除以 1000
                terminal_downrange = traj.downrange[-1] if len(traj.downrange) > 0 else 0.0
                terminal_altitude = traj.altitude[-1] if len(traj.altitude) > 0 else 0.0
                status = "OK"
            else:
                domain = determine_mission_domain(eta)
                terminal_downrange = 0.0
                terminal_altitude = 0.0
                status = "NO_DATA"

            record = {
                "fault_id": fault_id,
                "fault_name": FAULT_NAMES[fault_id],
                "scenario_id": FAULT_ID_MAP[fault_id],
                "eta": eta,
                "mission_domain": domain,
                "mission_domain_cn": get_domain_chinese_name(domain),
                "terminal_downrange": terminal_downrange,
                "terminal_altitude": terminal_altitude,
                "status": status,
            }
            records.append(record)

            print(
                f"    - eta={eta:.1f} | domain={domain:10s} | "
                f"downrange={terminal_downrange:7.1f} km | alt={terminal_altitude:6.1f} km | {status}"
            )

    print(f"\n  成功加载 {loaded_count}/{len(fault_ids) * len(eta_values)} 条轨迹数据")
    print()

    # Save CSV
    print("[2/2] 保存结果...")
    df = pd.DataFrame(records)
    outdir = PROJECT_ROOT / "outputs" / "ch4" / "tables"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_file = outdir / "ch4_mission_domains.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"  - CSV 已保存: {csv_file}")

    # Generate Markdown table
    md_file = outdir / "table_ch4_mission_domains.md"
    generate_markdown_table(df, md_file)
    print()

    # 打印汇总
    print("=" * 80)
    print("任务域统计汇总")
    print("=" * 80)
    domain_counts = df['mission_domain'].value_counts()
    for domain, count in domain_counts.items():
        print(f"  {domain:12s} ({get_domain_chinese_name(domain):8s}): {count} 条")
    print()
    print("全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
