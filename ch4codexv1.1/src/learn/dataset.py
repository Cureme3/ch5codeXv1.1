# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from sim.run_nominal import R_EARTH, simulate_full_mission
from sim.run_fault import plan_recovery_segment_scvx, run_fault_scenario
from sim.run_fault import _load_config  # type: ignore[attr-defined]

SCENARIOS = [
    "F1_thrust_deg15",
    "F2_tvc_rate4",
    "F3_tvc_stuck3deg",
    "F4_sensor_bias2deg",
    "F5_event_delay5s",
]


def build_offline_dataset(out_path: Path, num_samples: int = 200) -> None:
    """
    从 F1–F5 故障 + SCvx 重规划结果构造离线数据集：

    特征向量 X（7维，与 warmstart.py 中 _extract_feature_vector 严格一致）：
    - [0] 故障类别索引 (0-4 对应 F1-F5)
    - [1] t_confirm [s] - 故障确认时间
    - [2] h_confirm [km] - 确认时刻高度
    - [3] v_confirm [km/s] - 确认时刻速度
    - [4] q_confirm [kPa] - 确认时刻动压
    - [5] n_confirm [g] - 确认时刻法向过载
    - [6] thrust_kN [kN] - 当前阶段推力

    标签向量 Y：
    - 重规划段离散节点上的高度序列 [km]，长度 = nodes（默认40）
    - 来源于 plan_recovery_segment_scvx 返回的状态轨迹
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    cfg = _load_config()
    timeline = cfg.get("timeline", {})
    t_12_sep = float(timeline.get("t_12_sep_s", timeline.get("t_stage1_end_s", 0.0)))
    t_23_sep = float(timeline.get("t_23_sep_s", t_12_sep))
    t_34_sep = float(timeline.get("t_34_sep_s", t_23_sep))

    def _stage_thrust_kN(t_conf: float) -> float:
        if t_conf < t_12_sep:
            stage_idx = 1
        elif t_conf < t_23_sep:
            stage_idx = 2
        elif t_conf < t_34_sep:
            stage_idx = 3
        else:
            stage_idx = 4
        for stage in cfg.get("stages", []):
            if int(stage.get("index", -1)) == stage_idx:
                return float(stage.get("thrust_kN", 0.0))
        return 0.0

    for idx, scenario_id in enumerate(SCENARIOS):
        nominal = simulate_full_mission(dt=1.0)
        fault_sim = run_fault_scenario(scenario_id, dt=1.0)
        scenario = fault_sim.scenario

        seg = plan_recovery_segment_scvx(scenario, fault_sim, nominal, nodes=40)
        states = seg.states
        pos = states[:, 0:3]
        vel = states[:, 3:6]

        altitude_km = (np.linalg.norm(pos, axis=1) - R_EARTH) / 1000.0
        speed_kms = np.linalg.norm(vel, axis=1) / 1000.0

        h_conf_km = float(altitude_km[0])
        v_conf_kms = float(speed_kms[0])

        q_conf_kpa = 0.0
        n_conf_g = 0.0

        t_conf = float(scenario.t_confirm_s)
        thrust_kN = _stage_thrust_kN(t_conf)

        # 构造特征向量 X（7维，顺序与 warmstart.py 严格一致）
        feature = np.array(
            [
                float(idx),       # [0] 故障类别索引
                t_conf,           # [1] 确认时间 [s]
                h_conf_km,        # [2] 确认时刻高度 [km]
                v_conf_kms,       # [3] 确认时刻速度 [km/s]
                q_conf_kpa,       # [4] 确认时刻动压 [kPa]
                n_conf_g,         # [5] 确认时刻法向过载 [g]
                thrust_kN,        # [6] 当前阶段推力 [kN]
            ],
            dtype=float,
        )
        # 标签向量 Y: 高度序列 [km]
        label = altitude_km.astype(float)

        X_list.append(feature)
        Y_list.append(label)

    if not X_list:
        X = np.zeros((0, 7), dtype=float)
        Y = np.zeros((0, 40), dtype=float)
    else:
        base_X = np.vstack(X_list)
        base_Y = np.vstack(Y_list)

        repeats = 20
        rng = np.random.default_rng(seed=42)
        X_aug = []
        Y_aug = []
        for x, y in zip(base_X, base_Y):
            for _ in range(repeats):
                x_noisy = x.copy()
                x_noisy[1] += rng.normal(0.0, 1.0)
                x_noisy[2] += rng.normal(0.0, 2.0)
                x_noisy[3] += rng.normal(0.0, 0.05)
                x_noisy[4] += rng.normal(0.0, 2.0)
                x_noisy[5] += rng.normal(0.0, 0.05)
                x_noisy[6] += rng.normal(0.0, 20.0)
                X_aug.append(x_noisy)
                Y_aug.append(y.copy())

        X = np.vstack(X_aug)
        Y = np.vstack(Y_aug)

    np.savez(out_path, X=X, Y=Y)
    print(f"[INFO] Saved offline dataset to {out_path} with shape X={X.shape}, Y={Y.shape}")
