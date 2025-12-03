#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三章诊断 → 第四章重规划接口层。

该模块提供统一的 DiagnosisResult 数据类，用于承接第三章故障诊断输出，
并将其映射为第四章 SCvx 重规划所需的 FaultScenario 与故障严重度 η。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Dict, Tuple

from .scenarios import FaultScenario, get_scenario


@dataclass
class DiagnosisResult:
    """
    第三章诊断输出在第四章视角下的统一接口。

    说明：
    - fault_class_idx: 第三章分类器输出的类别索引（例如 0~5）
    - fault_label: 可读的故障标签（例如 "nominal", "thrust_drop", ...），
                   用于映射到 F1~F5 等场景 ID
    - confidence: 飞行级诊断置信度（0~1）
    - severity_raw: 来自第三章的"严重度指标"（例如残差幅值归一化、votes margin 等）
    - severity_eta: 若第三章已给出归一化的 η∈[0,1]，可直接填入；否则留空，由第四章统一映射
    """

    fault_class_idx: int
    fault_label: str
    confidence: float

    severity_raw: Optional[float] = None
    severity_eta: Optional[float] = None

    # 以下为可选的等效参数与时间信息，方便后续扩展
    thrust_drop_est: Optional[float] = None
    tvc_rate_limit_est: Optional[float] = None
    tvc_stuck_deg_est: Optional[float] = None
    sensor_bias_deg_est: Optional[float] = None
    event_delay_s_est: Optional[float] = None

    t_fault_s: Optional[float] = None
    t_detect_s: Optional[float] = None
    t_confirm_s: Optional[float] = None

    extra: Optional[Dict[str, float]] = None


def map_fault_class_to_scenario_id(fault_class_idx: int, fault_label: str) -> str:
    """
    根据第三章输出的类别索引和标签，映射到第四章的场景 ID。

    参数：
    - fault_class_idx: 第三章分类器输出的类别索引（0~5）
    - fault_label: 故障标签字符串

    返回：
    - scenario_id: 第四章场景 ID（如 "F1_thrust_deg15"）

    映射规则：
    - 0 或 label 含 "nominal" → "nominal"
    - 1 或 label 含 "thrust" → "F1_thrust_deg15"
    - 2 或 label 含 "tvc_rate" → "F2_tvc_rate4"
    - 3 或 label 含 "tvc_stick" / "tvc_stuck" → "F3_tvc_stuck3deg"
    - 4 或 label 含 "sensor" → "F4_sensor_bias2deg"
    - 5 或 label 含 "delay" → "F5_event_delay5s"

    若无法识别，抛出 ValueError。
    """
    label_lower = fault_label.lower()

    # 优先按类别索引映射
    if fault_class_idx == 0 or "nominal" in label_lower:
        return "nominal"
    elif fault_class_idx == 1 or "thrust" in label_lower:
        return "F1_thrust_deg15"
    elif fault_class_idx == 2 or "tvc_rate" in label_lower:
        return "F2_tvc_rate4"
    elif fault_class_idx == 3 or "tvc_stick" in label_lower or "tvc_stuck" in label_lower:
        return "F3_tvc_stuck3deg"
    elif fault_class_idx == 4 or "sensor" in label_lower:
        return "F4_sensor_bias2deg"
    elif fault_class_idx == 5 or "delay" in label_lower:
        return "F5_event_delay5s"
    else:
        raise ValueError(
            f"无法识别的故障类别: idx={fault_class_idx}, label='{fault_label}'"
        )


def build_fault_scenario_from_diag(diag: DiagnosisResult) -> FaultScenario:
    """
    从 DiagnosisResult 构造一个 FaultScenario。

    逻辑：
    1. 调用 map_fault_class_to_scenario_id 得到 scenario_id
    2. 用 get_scenario(scenario_id) 取得基准 FaultScenario
    3. 若 diag 中提供了等效参数估计（如 thrust_drop_est），则覆盖 params
    4. 若 diag 中提供了时间信息（如 t_fault_s），则覆盖时间字段
    5. 返回新的 FaultScenario 实例

    参数：
    - diag: 诊断结果

    返回：
    - 构造后的 FaultScenario
    """
    # 1. 获取基准场景
    scenario_id = map_fault_class_to_scenario_id(diag.fault_class_idx, diag.fault_label)
    base = get_scenario(scenario_id)

    # 2. 拷贝 params 并覆盖估计值
    params = dict(base.params)

    if diag.thrust_drop_est is not None and "degrade_frac" in params:
        params["degrade_frac"] = float(diag.thrust_drop_est)

    if diag.tvc_rate_limit_est is not None and "tvc_rate_deg_s" in params:
        params["tvc_rate_deg_s"] = float(diag.tvc_rate_limit_est)

    if diag.tvc_stuck_deg_est is not None and "stuck_angle_deg" in params:
        params["stuck_angle_deg"] = float(diag.tvc_stuck_deg_est)

    if diag.sensor_bias_deg_est is not None and "sensor_bias_deg" in params:
        params["sensor_bias_deg"] = float(diag.sensor_bias_deg_est)

    if diag.event_delay_s_est is not None and "event_delay_s" in params:
        params["event_delay_s"] = float(diag.event_delay_s_est)

    # 3. 覆盖时间信息
    t_fault = diag.t_fault_s if diag.t_fault_s is not None else base.t_fault_s
    t_detect = diag.t_detect_s if diag.t_detect_s is not None else base.t_detect_s
    t_confirm = diag.t_confirm_s if diag.t_confirm_s is not None else base.t_confirm_s

    # 4. 构造新的 FaultScenario
    return FaultScenario(
        id=base.id,
        description=base.description,
        fault_type=base.fault_type,
        t_fault_s=t_fault,
        t_detect_s=t_detect,
        t_confirm_s=t_confirm,
        t_plan_horizon_s=base.t_plan_horizon_s,
        params=params,
    )


def compute_eta_from_diag(diag: DiagnosisResult, scenario: FaultScenario) -> float:
    """
    统一将诊断输出映射为 η∈[0,1]，供自适应罚权重使用。

    映射逻辑（优先级从高到低）：
    1. 若 diag.severity_eta 非空且在 [0,1] 内，则直接返回
    2. 若 diag.severity_raw 非空，则做简单归一化（假设 raw 已经是 0~1 尺度）
    3. 若 scenario.params 中存在故障强度参数，则简单映射
    4. 否则返回保守默认值 0.5

    注意：后续可以由第三章的详细指标（如残差幅值、置信度、特征量等）
    替换这里的占位归一化规则。

    参数：
    - diag: 诊断结果
    - scenario: 对应的故障场景

    返回：
    - eta: 故障严重度，范围 [0, 1]
    """
    # 1. 优先使用 severity_eta
    if diag.severity_eta is not None:
        eta = float(diag.severity_eta)
        if 0.0 <= eta <= 1.0:
            return eta

    # 2. 尝试使用 severity_raw
    if diag.severity_raw is not None:
        # 假设 raw 已经是 0~1 尺度，做简单 clamp
        eta = max(0.0, min(1.0, float(diag.severity_raw)))
        return eta

    # 3. 根据 scenario.params 中的故障强度参数估计
    params = scenario.params

    # 推力降级：degrade_frac ∈ [0, 1]
    if "degrade_frac" in params:
        eta = max(0.0, min(1.0, abs(float(params["degrade_frac"]))))
        return eta

    # TVC 速率限制：tvc_rate_deg_s 越小越严重（正常约 10 deg/s，故障约 0.5 deg/s）
    if "tvc_rate_deg_s" in params:
        rate = float(params["tvc_rate_deg_s"])
        # 映射：10 deg/s → 0.0, 0.5 deg/s → 0.95
        eta = max(0.0, min(1.0, 1.0 - rate / 10.0))
        return eta

    # TVC 卡滞：stuck_angle_deg 越大越严重（假设正常偏差 < 2 deg，故障 > 5 deg）
    if "stuck_angle_deg" in params:
        angle = abs(float(params["stuck_angle_deg"]))
        # 映射：5 deg → 0.5, 10 deg → 1.0
        eta = max(0.0, min(1.0, angle / 10.0))
        return eta

    # 传感器偏置：sensor_bias_deg 越大越严重
    if "sensor_bias_deg" in params:
        bias = abs(float(params["sensor_bias_deg"]))
        # 映射：5 deg → 0.5, 10 deg → 1.0
        eta = max(0.0, min(1.0, bias / 10.0))
        return eta

    # 事件延迟：event_delay_s 越大越严重
    if "event_delay_s" in params:
        delay = abs(float(params["event_delay_s"]))
        # 映射：15 s → 0.6, 25 s → 1.0
        eta = max(0.0, min(1.0, delay / 25.0))
        return eta

    # 4. 保守默认值
    return 0.5


def diagnosis_to_scenario_and_eta(diag: DiagnosisResult) -> Tuple[FaultScenario, float]:
    """
    综合调用 build_fault_scenario_from_diag 和 compute_eta_from_diag，
    一次性得到 FaultScenario 和对应的故障严重度 eta。

    参数：
    - diag: 诊断结果

    返回：
    - (scenario, eta): 故障场景与严重度
    """
    scenario = build_fault_scenario_from_diag(diag)
    eta = compute_eta_from_diag(diag, scenario)
    return scenario, eta
