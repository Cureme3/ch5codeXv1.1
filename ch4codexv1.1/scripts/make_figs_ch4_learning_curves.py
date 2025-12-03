#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图4-10：学习热启动模型的训练/验证损失曲线。

从 train_log.json 生成训练与验证损失随 epoch 变化的曲线。

输出文件：
- outputs/figures/ch4_learning/fig4_10_learning_curves.png
- outputs/figures/ch4_learning/fig4_10_learning_curves.pdf

命令行用法：
    python -m scripts.make_figs_ch4_learning_curves
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 路径设置
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plots.plotting import setup_matplotlib  # noqa: E402

# 输入输出路径
DATA_DIR = ROOT / "outputs" / "data" / "ch4_learning"
OUT_DIR = ROOT / "outputs" / "figures" / "ch4_learning"


def load_train_log(log_path: Path) -> dict:
    """加载训练日志文件。"""
    if not log_path.exists():
        raise FileNotFoundError(f"训练日志不存在: {log_path}")
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_learning_curves(log_data: dict, out_dir: Path) -> None:
    """绘制训练/验证损失曲线。

    Parameters
    ----------
    log_data : dict
        训练日志数据，包含 train_loss 和 val_loss 列表
    out_dir : Path
        输出目录
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loss = log_data.get('train_loss', [])
    val_loss = log_data.get('val_loss', [])

    if not train_loss or not val_loss:
        print("[WARN] 训练日志中缺少 train_loss 或 val_loss 数据")
        return

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='训练损失 (Train Loss)')
    ax.plot(epochs, val_loss, 'r--', linewidth=2, label='验证损失 (Val Loss)')

    ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax.set_ylabel('均方误差损失 (MSE Loss)', fontsize=12)
    ax.set_title('学习热启动模型训练曲线', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1, len(epochs))

    # 设置对数坐标（如果损失跨越多个数量级）
    if max(train_loss) / min(train_loss) > 100:
        ax.set_yscale('log')

    plt.tight_layout()

    # 保存 PNG 和 PDF
    png_path = out_dir / "fig4_10_learning_curves.png"
    pdf_path = out_dir / "fig4_10_learning_curves.pdf"

    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def main() -> None:
    """主函数。"""
    setup_matplotlib()
    print("=" * 60)
    print("图4-10：学习热启动模型训练曲线")
    print("=" * 60)

    log_path = DATA_DIR / "train_log.json"
    print(f"读取训练日志: {log_path}")

    try:
        log_data = load_train_log(log_path)
        print(f"  - 训练轮次: {len(log_data.get('train_loss', []))}")

        hyper = log_data.get('hyperparameters', {})
        if hyper:
            print(f"  - 学习率: {hyper.get('lr', 'N/A')}")
            print(f"  - 批大小: {hyper.get('batch_size', 'N/A')}")
        print()

        plot_learning_curves(log_data, OUT_DIR)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("请先运行 python -m scripts.train_ch4_learning 生成训练日志")
        return

    print()
    print("=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
