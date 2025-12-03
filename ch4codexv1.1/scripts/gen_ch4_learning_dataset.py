#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四章：生成学习热启动训练数据集。

此脚本仅负责数据集生成，不进行模型训练。
训练请使用 train_ch4_learning.py。

输出文件：
- outputs/data/ch4_learning/dataset.npz

数据集格式：
- X: 特征矩阵 (N, 7)
    [0] 故障类别索引 (0-4 对应 F1-F5)
    [1] t_confirm [s] - 故障确认时间
    [2] h_confirm [km] - 确认时刻高度
    [3] v_confirm [km/s] - 确认时刻速度
    [4] q_confirm [kPa] - 确认时刻动压
    [5] n_confirm [g] - 确认时刻法向过载
    [6] thrust_kN [kN] - 当前阶段推力
- Y: 标签矩阵 (N, nodes) - 高度序列 [km]

用法:
    python -m scripts.gen_ch4_learning_dataset [--samples N] [--nodes N] [--augment]
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from learn.dataset import build_offline_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="生成第四章学习热启动训练数据集")
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="生成的样本数量（默认: 200，含数据增强）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/data/ch4_learning/dataset.npz",
        help="输出数据集路径",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("第四章 4.3：学习热启动数据集生成")
    print("=" * 80)
    print()

    data_dir = Path(args.output).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.output)

    print(f"[INFO] 目标路径: {dataset_path}")
    print(f"[INFO] 样本数量: {args.samples}")
    print()

    build_offline_dataset(dataset_path, num_samples=args.samples)

    print()
    print("=" * 80)
    print("数据集生成完成！")
    print("=" * 80)
    print("\n下一步：运行 train_ch4_learning.py 进行模型训练")


if __name__ == "__main__":
    main()
