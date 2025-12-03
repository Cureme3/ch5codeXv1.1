#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学习热启动工具模块：加载训练模型并根据故障场景生成高度预测。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from learn.model import build_model
from sim.run_fault import FaultScenario, FaultSimResult, NominalResult, _load_config, sample_state_at_time
from sim.run_nominal import R_EARTH


@dataclass
class LearningContext:
    """封装学习热启动所需的模型及元信息。"""

    model: torch.nn.Module
    device: torch.device
    nodes: int
    feature_mean: np.ndarray
    feature_std: np.ndarray


def load_learning_context(model_dir: str | Path = "outputs/data/ch4_learning") -> LearningContext:
    """
    加载离线训练得到的学习模型及其特征统计。

    特征归一化统计（mean/std）从 dataset.npz 中实时计算，
    确保与训练时的归一化方式一致。
    """

    model_dir = Path(model_dir)
    dataset_path = model_dir / "dataset.npz"
    model_path = model_dir / "model.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found at {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model checkpoint not found at {model_path}")

    data = np.load(dataset_path)
    X = np.asarray(data["X"], dtype=np.float32)
    Y = np.asarray(data["Y"], dtype=np.float32)

    # 优先从 feature_stats.json 读取归一化统计，以确保与训练时完全一致；
    # 若不存在则退化为从 dataset 计算。
    stats_path = model_dir / "feature_stats.json"
    feature_mean: np.ndarray
    feature_std: np.ndarray
    output_dim: int
    if stats_path.exists():
        import json
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        feature_mean = np.asarray(stats.get("feature_mean", X.mean(axis=0)), dtype=np.float32)
        feature_std = np.asarray(stats.get("feature_std", X.std(axis=0) + 1e-6), dtype=np.float32)
        output_dim = int(stats.get("output_len", int(Y.shape[1])))
    else:
        feature_mean = X.mean(axis=0)
        feature_std = X.std(axis=0) + 1e-6
        output_dim = int(Y.shape[1])

    input_dim = X.shape[1]
    model = build_model(input_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return LearningContext(
        model=model,
        device=device,
        nodes=output_dim,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def _stage_thrust_kN(t_conf: float) -> float:
    cfg = _load_config()
    stages = cfg.get("stages", [])
    timeline = cfg.get("timeline", {})
    t_12_sep = float(timeline.get("t_12_sep_s", timeline.get("t_stage1_end_s", 0.0)))
    t_23_sep = float(timeline.get("t_23_sep_s", t_12_sep))
    t_34_sep = float(timeline.get("t_34_sep_s", t_23_sep))
    if t_conf < t_12_sep:
        stage_idx = 1
    elif t_conf < t_23_sep:
        stage_idx = 2
    elif t_conf < t_34_sep:
        stage_idx = 3
    else:
        stage_idx = 4
    for stage in stages:
        if int(stage.get("index", -1)) == stage_idx:
            return float(stage.get("thrust_kN", 0.0))
    return 0.0


def _fault_index(scenario_id: str) -> int:
    catalog = [
        "F1_thrust_deg15",
        "F2_tvc_rate4",
        "F3_tvc_stuck3deg",
        "F4_sensor_bias2deg",
        "F5_event_delay5s",
    ]
    try:
        return catalog.index(scenario_id)
    except ValueError:
        return 0


def _extract_feature_vector(
    scenario: FaultScenario,
    fault_sim: FaultSimResult,
    nominal: NominalResult,
) -> np.ndarray:
    """
    构造特征向量，与 dataset.py 中 build_offline_dataset 的 X 定义严格一致。

    特征向量（7维）：
    - [0] 故障类别索引 (0-4 对应 F1-F5)
    - [1] t_confirm [s] - 故障确认时间
    - [2] h_confirm [km] - 确认时刻高度
    - [3] v_confirm [km/s] - 确认时刻速度
    - [4] q_confirm [kPa] - 确认时刻动压
    - [5] n_confirm [g] - 确认时刻法向过载
    - [6] thrust_kN [kN] - 当前阶段推力
    """
    t_conf = float(scenario.t_confirm_s)
    _, state_conf, idx = sample_state_at_time(fault_sim, t_conf)
    pos = state_conf[0:3]
    vel = state_conf[3:6]
    altitude_km = (np.linalg.norm(pos) - R_EARTH) / 1000.0
    speed_kms = np.linalg.norm(vel) / 1000.0
    q_series = getattr(fault_sim, "dynamic_pressure_kpa", None)
    n_series = getattr(fault_sim, "normal_load_g", None)
    q_conf = float(q_series[idx]) if q_series is not None and len(q_series) > idx else 0.0
    n_conf = float(n_series[idx]) if n_series is not None and len(n_series) > idx else 0.0

    # 构造特征向量（7维，顺序与 dataset.py 严格一致）
    feature = np.array(
        [
            float(_fault_index(scenario.id)),  # [0] 故障类别索引
            t_conf,                            # [1] 确认时间 [s]
            altitude_km,                       # [2] 确认时刻高度 [km]
            speed_kms,                         # [3] 确认时刻速度 [km/s]
            q_conf,                            # [4] 确认时刻动压 [kPa]
            n_conf,                            # [5] 确认时刻法向过载 [g]
            _stage_thrust_kN(t_conf),          # [6] 当前阶段推力 [kN]
        ],
        dtype=np.float32,
    )
    return feature


def build_learning_warmstart(
    ctx: LearningContext,
    scenario: FaultScenario,
    fault_sim: FaultSimResult,
    nominal: NominalResult,
    nodes: int,
) -> np.ndarray:
    """根据当前故障场景预测一条高度轨迹，用作学习热启动。"""

    feature = _extract_feature_vector(scenario, fault_sim, nominal)
    feature_norm = (feature - ctx.feature_mean) / ctx.feature_std
    x_tensor = torch.from_numpy(feature_norm.astype(np.float32)).unsqueeze(0).to(ctx.device)
    with torch.no_grad():
        pred = ctx.model(x_tensor).cpu().numpy()[0]
    base_nodes = ctx.nodes
    if base_nodes == nodes:
        return pred.copy()
    base_grid = np.linspace(0.0, 1.0, base_nodes)
    target_grid = np.linspace(0.0, 1.0, nodes)
    interp = np.interp(target_grid, base_grid, pred)
    return interp.astype(np.float32)
