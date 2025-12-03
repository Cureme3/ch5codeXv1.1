#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四章：从数据集训练学习热启动模型。

此脚本仅负责模型训练，不生成数据集。
数据集生成请使用 gen_ch4_learning_dataset.py。

输入文件：
- outputs/data/ch4_learning/dataset.npz (由 gen_ch4_learning_dataset.py 生成)

输出文件：
- outputs/data/ch4_learning/model.pt
- outputs/data/ch4_learning/train_log.json
- outputs/data/ch4_learning/feature_stats.json

用法:
    python -m scripts.train_ch4_learning [--epochs N] [--batch-size N] [--lr LR]
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from learn.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="训练第四章学习热启动模型")
    parser.add_argument(
        "--dataset",
        type=str,
        default="outputs/data/ch4_learning/dataset.npz",
        help="输入数据集路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/data/ch4_learning",
        help="输出目录",
    )
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument(
        "--gen-dataset",
        action="store_true",
        help="如果数据集不存在，先生成数据集（向后兼容）",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("第四章 4.3：学习热启动模型训练")
    print("=" * 80)
    print()

    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset)

    # 检查数据集是否存在
    if not dataset_path.exists():
        if args.gen_dataset:
            print(f"[INFO] 数据集不存在，正在生成: {dataset_path}")
            from learn.dataset import build_offline_dataset
            build_offline_dataset(dataset_path, num_samples=200)
            print()
        else:
            print(f"[ERROR] 数据集不存在: {dataset_path}")
            print("请先运行: python -m scripts.gen_ch4_learning_dataset")
            print("或添加 --gen-dataset 参数自动生成")
            sys.exit(1)

    print(f"[INFO] 数据集路径: {dataset_path}")
    print(f"[INFO] 输出目录: {data_dir}")
    print(f"[INFO] 训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print()

    train_model(
        dataset_path,
        out_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print()
    print("=" * 80)
    print("训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
