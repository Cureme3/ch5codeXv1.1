# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import build_model

# 固定随机种子，确保可复现性
RANDOM_SEED = 42


def _train_val_split(
    X: np.ndarray,
    Y: np.ndarray,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = X.shape[0]
    if N < 2:
        return X, X.copy(), Y, Y.copy()
    n_val = max(1, int(N * val_ratio))
    idx = np.arange(N)
    np.random.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]


def train_model(
    dataset_path: Path,
    out_dir: Path,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    """
    训练 MLP 回归模型，预测高度序列。

    保存文件：
    - model.pt: 模型权重
    - train_log.json: 训练/验证损失曲线及超参数
    - feature_stats.json: 特征归一化统计（mean, std, output_len）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 固定随机种子
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    data = np.load(dataset_path)
    X = np.asarray(data["X"], dtype=np.float32)
    Y = np.asarray(data["Y"], dtype=np.float32)

    if X.shape[0] == 0:
        print(f"[WARN] dataset {dataset_path} is empty, skip training.")
        return

    X_train, X_val, Y_train, Y_val = _train_val_split(X, Y, val_ratio=0.2)

    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    model = build_model(input_dim, output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    log: Dict[str, Any] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_ds))

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"[EPOCH {epoch + 1:03d}] train_loss={train_loss:.4e}, val_loss={val_loss:.4e}")

    # 保存训练日志
    log_data = {
        "train_loss": log["train_loss"],
        "val_loss": log["val_loss"],
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "random_seed": RANDOM_SEED,
        },
        "data_shapes": {
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "input_dim": input_dim,
            "output_dim": output_dim,
        },
    }
    (out_dir / "train_log.json").write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    # 保存模型权重
    torch.save(model.state_dict(), out_dir / "model.pt")

    # 保存特征归一化统计
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0) + 1e-6
    feature_stats = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "output_len": int(output_dim),
    }
    (out_dir / "feature_stats.json").write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")

    # 打印训练摘要
    final_train_loss = log["train_loss"][-1]
    final_val_loss = log["val_loss"][-1]
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(val_ds)}")
    print(f"最终训练 MSE: {final_train_loss:.6e}")
    print(f"最终验证 MSE: {final_val_loss:.6e}")
    print(f"\n模型存储路径: {out_dir / 'model.pt'}")
    print(f"训练日志路径: {out_dir / 'train_log.json'}")
    print(f"特征统计路径: {out_dir / 'feature_stats.json'}")
    print("=" * 60)
