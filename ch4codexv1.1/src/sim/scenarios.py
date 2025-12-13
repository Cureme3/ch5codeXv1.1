"""第四章故障场景定义与场景库。

该模块按照论文第 4 章（以及《要求 4.docx》）中的符号定义统一的数据结构：

- t_fault_s (t_f): 故障真实发生时间（感知滞后之前）
- t_detect_s (t_det): 诊断系统首次检测到异常
- t_confirm_s (t_conf): 故障被确认、允许进入重构阶段的时间
- t_plan_horizon_s (ΔT): 单次 SCvx/重构规划窗口长度
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class FaultScenario:
    """故障场景描述。

    Attributes
    ----------
    id : str
        场景唯一 ID（如 “F1_thrust_deg15”），对应论文中的 F1~F5 编号。
    description : str
        中文描述，可直接放入表格或 CLI 输出。
    fault_type : str
        故障类别（thrust_degradation / tvc_rate_limit / tvc_stuck /
        sensor_bias / event_delay）。
    t_fault_s : float
        故障真实发生时刻 t_f。
    t_detect_s : float
        诊断系统首次检测时间 t_det。
    t_confirm_s : float
        诊断确认时间 t_conf / 重构窗口起点。
    t_plan_horizon_s : float
        重构规划窗口长度 ΔT。
    params : Dict[str, float]
        故障参数，如 degrade_frac、theta_max_deg_after、tvc_rate_deg_s、
        stuck_angle_deg、sensor_bias_deg、event_delay_s 等。
    """

    id: str
    description: str
    fault_type: str
    t_fault_s: float
    t_detect_s: float
    t_confirm_s: float
    t_plan_horizon_s: float
    params: Dict[str, float]


# ---- 典型场景（论文示例工况） ----------------------------------------------
# 故障参数设计原则：
# 1. 基础参数（degrade_frac等）定义为 eta=1.0 时的最大故障强度
# 2. eta=0.2/0.5/0.8 通过 scale_scenario_by_eta() 缩放得到
# 3. 参数值经过调整，确保：
#    - 开环轨迹有明显偏离但不会太快坠毁
#    - 不同eta值之间有明显区分
#    - 故障时间点与飞行阶段匹配（推力非零时）
#
# KZ-1A 推力裕度分析（基于3D霍曼转移制导仿真验证）：
# - 名义轨迹：500km SSO圆轨道，霍曼转移制导
# - 0-6.5% 推力损失：仍可成功入轨（RETAIN）
# - 7%+ 推力损失：需要S4轨迹重构（SCvx）
# - 15%+ 推力损失：需要降级到SAFE_AREA
# 因此 F1 最大推力损失设为 20%，使得：
# - eta=0.2 → 4% 损失 → RETAIN 可行
# - eta=0.5 → 10% 损失 → DEGRADED（需S4重构）
# - eta=0.8 → 16% 损失 → SAFE_AREA（亚轨道/安全落区）
#
# 重要约束：
# - 前三级(S1/S2/S3)是固体火箭，点燃后无法停止
# - 只有S4(液体火箭)可以多次点火，是轨迹重构的关键
# - 俯仰调整只能在S4阶段通过SCvx优化实现

SCENARIO_CATALOG: Dict[str, FaultScenario] = {
    # F1：推力降级
    # t_fault=85s 是第二级点火初期，推力 ~354 kN
    # 最大推力损失 20%（需要S4 SCvx轨迹重构）
    # eta=0.2 → 4% 损失 → RETAIN
    # eta=0.5 → 10% 损失 → DEGRADED（需S4重构）
    # eta=0.8 → 16% 损失 → SAFE_AREA
    "F1_thrust_deg15": FaultScenario(
        id="F1_thrust_deg15",
        description="第二级推力降级 20%，需S4轨迹重构",
        fault_type="thrust_degradation",
        t_fault_s=85.0,  # Stage 2 burn phase
        t_detect_s=95.0,
        t_confirm_s=105.0,
        t_plan_horizon_s=150.0,
        params={
            "degrade_frac": 0.20,  # 20% thrust loss at eta=1.0
            "theta_max_deg_after": 7.0,
        },
    ),
    "F1_severe": FaultScenario(
        id="F1_severe",
        description="F1 severe thrust degradation (5% drop)",
        fault_type="thrust_degradation",
        t_fault_s=85.0,
        t_detect_s=95.0,
        t_confirm_s=105.0,
        t_plan_horizon_s=150.0,
        params={
            "degrade_frac": 0.05,  # 5% thrust loss (definitely suborbital)
            "theta_max_deg_after": 6.0,
        },
    ),
    # F2：TVC 卡滞偏置
    # TVC固定偏置导致推力方向持续偏离
    # eta=0.2有偏离不坠毁，eta=0.5/0.8坠毁
    "F2_tvc_rate4": FaultScenario(
        id="F2_tvc_rate4",
        description="TVC 卡滞偏置 (最大35度)",
        fault_type="tvc_stuck",
        t_fault_s=85.0,  # S2阶段发生故障
        t_detect_s=95.0,
        t_confirm_s=105.0,
        t_plan_horizon_s=140.0,
        params={
            "stuck_angle_deg": 35.0,  # eta=0.2时7度，eta=0.5时17.5度，eta=0.8时28度
            "stuck_duration_s": 200.0,
        },
    ),
    "F2_severe": FaultScenario(
        id="F2_severe",
        description="F2 severe: TVC速率限制 0.5 deg/s",
        fault_type="tvc_rate_limit",
        t_fault_s=100.0,
        t_detect_s=110.0,
        t_confirm_s=120.0,
        t_plan_horizon_s=140.0,
        params={
            "tvc_rate_deg_s": 0.25,  # 更严重的速率限制
        },
    ),
    # F3：TVC 卡滞
    # TVC在某一角度卡住，导致推力方向固定偏离
    # eta=0.2有偏离不坠毁，eta=0.5/0.8坠毁
    "F3_tvc_stuck3deg": FaultScenario(
        id="F3_tvc_stuck3deg",
        description="TVC 卡滞 (最大50度, 150秒)",
        fault_type="tvc_stuck",
        t_fault_s=85.0,  # S2阶段初期，有推力
        t_detect_s=90.0,
        t_confirm_s=95.0,
        t_plan_horizon_s=140.0,
        params={
            "stuck_angle_deg": 50.0,  # eta=0.2时10度，eta=0.5时25度，eta=0.8时40度
            "stuck_duration_s": 150.0,
        },
    ),
    "F3_severe": FaultScenario(
        id="F3_severe",
        description="F3 severe: TVC卡滞 20度",
        fault_type="tvc_stuck",
        t_fault_s=200.0,
        t_detect_s=210.0,
        t_confirm_s=220.0,
        t_plan_horizon_s=140.0,
        params={
            "stuck_angle_deg": 20.0,
            "stuck_duration_s": 150.0,
        },
    ),
    # F4：姿态传感器偏置
    # 加速度计/陀螺仪偏置导致姿态估计误差，进而影响制导
    # 传感器偏置效果很强，需要较小参数
    # eta=0.2有偏离不坠毁，eta=0.5/0.8坠毁
    "F4_sensor_bias2deg": FaultScenario(
        id="F4_sensor_bias2deg",
        description="传感器偏置 (最大8度)",
        fault_type="sensor_bias",
        t_fault_s=85.0,  # S2阶段
        t_detect_s=95.0,
        t_confirm_s=105.0,
        t_plan_horizon_s=140.0,
        params={
            "sensor_bias_deg": 8.0,  # eta=0.2时1.6度，eta=0.5时4度，eta=0.8时6.4度
        },
    ),
    "F4_severe": FaultScenario(
        id="F4_severe",
        description="F4 severe: 传感器偏置 10度",
        fault_type="sensor_bias",
        t_fault_s=100.0,
        t_detect_s=110.0,
        t_confirm_s=120.0,
        t_plan_horizon_s=140.0,
        params={
            "sensor_bias_deg": 10.0,
        },
    ),
    # F5：事件时序延迟
    # 级间分离或点火延迟导致重力损失增加
    # eta=0.2有偏离不坠毁，eta=0.5/0.8坠毁
    "F5_event_delay5s": FaultScenario(
        id="F5_event_delay5s",
        description="S4点火延迟 (最大200秒)",
        fault_type="event_delay",
        t_fault_s=200.0,
        t_detect_s=205.0,
        t_confirm_s=215.0,
        t_plan_horizon_s=140.0,
        params={
            "event_delay_s": 200.0,  # eta=0.2时40s，eta=0.5时100s，eta=0.8时160s
        },
    ),
    "F5_severe": FaultScenario(
        id="F5_severe",
        description="F5 severe: S4点火延迟 30秒",
        fault_type="event_delay",
        t_fault_s=200.0,
        t_detect_s=205.0,
        t_confirm_s=215.0,
        t_plan_horizon_s=140.0,
        params={
            "event_delay_s": 30.0,
        },
    ),
}

# ---- 辅助接口 ---------------------------------------------------------------
def get_scenario_ids() -> List[str]:
    """返回可用场景 ID。"""

    return list(SCENARIO_CATALOG.keys())


def get_scenario(scenario_id: str) -> FaultScenario:
    """根据 ID 获取场景，若不存在则抛出 ValueError。"""

    try:
        return SCENARIO_CATALOG[scenario_id]
    except KeyError as exc:
        raise ValueError(f"未知场景 {scenario_id}, 可选值: {', '.join(get_scenario_ids())}") from exc


def list_scenarios() -> str:
    """返回每个场景的 ID 与简要描述，可用于 CLI 输出。"""

    lines = [f"{sid}: {scenario.description}" for sid, scenario in SCENARIO_CATALOG.items()]
    return "\n".join(lines)


def scale_scenario_by_eta(base: FaultScenario, eta: float) -> FaultScenario:
    """根据 eta ∈ [0, 1] 对基础 FaultScenario 进行强度缩放，返回新的 FaultScenario。

    设计原则:
    - eta = 0.0: 无故障或极轻微故障
    - eta = 0.5: 中等故障强度
    - eta = 1.0: 最大故障强度（使用 base 中定义的参数值）

    各故障类型的缩放方式:
    - thrust_degradation: degrade_frac *= eta
    - tvc_rate_limit: tvc_rate_deg_s 按 (1 - eta) 缩放（eta 越大限制越严）
                      angle_bias_deg *= eta
    - tvc_stuck: stuck_angle_deg *= eta
    - sensor_bias: sensor_bias_deg *= eta
    - event_delay: event_delay_s *= eta

    Parameters
    ----------
    base : FaultScenario
        基础故障场景（定义最大故障强度）
    eta : float
        故障严重度，范围 [0, 1]

    Returns
    -------
    FaultScenario
        缩放后的故障场景
    """
    eta = float(np.clip(eta, 0.0, 1.0))

    # Deep copy to avoid modifying original
    scenario = copy.deepcopy(base)
    params = scenario.params.copy()

    fault_type = scenario.fault_type

    if fault_type == "thrust_degradation":
        # 推力降级：degrade_frac 按 eta 缩放
        if "degrade_frac" in params:
            params["degrade_frac"] = eta * params["degrade_frac"]
        # 推力锥角限制也可以按 eta 缩放（越严重锥角越小）
        if "theta_max_deg_after" in params:
            base_theta = params["theta_max_deg_after"]
            # 从名义锥角（8度）线性插值到故障锥角
            nominal_theta = 8.0
            params["theta_max_deg_after"] = nominal_theta - eta * (nominal_theta - base_theta)

    elif fault_type == "tvc_rate_limit":
        # TVC 速率限制：eta 越大，限制越严（速率越小）
        # 使用指数缩放使中等eta值也有明显效果
        if "tvc_rate_deg_s" in params:
            base_rate = params["tvc_rate_deg_s"]  # eta=1.0时的最小速率
            nominal_rate = 10.0  # 名义 TVC 速率 deg/s
            # 指数缩放: rate = nominal * (base/nominal)^eta
            # eta=0: rate=10, eta=0.5: rate~1, eta=1: rate=base
            params["tvc_rate_deg_s"] = nominal_rate * (base_rate / nominal_rate) ** eta
        # 角度偏置按 eta 缩放
        if "angle_bias_deg" in params:
            params["angle_bias_deg"] = eta * params["angle_bias_deg"]

    elif fault_type == "tvc_stuck":
        # TVC 卡滞：卡滞角度按 eta 缩放
        if "stuck_angle_deg" in params:
            params["stuck_angle_deg"] = eta * params["stuck_angle_deg"]

    elif fault_type == "sensor_bias":
        # 传感器偏置：偏置量按 eta 缩放
        if "sensor_bias_deg" in params:
            params["sensor_bias_deg"] = eta * params["sensor_bias_deg"]

    elif fault_type == "event_delay":
        # 事件延迟：延迟时间按 eta 缩放
        if "event_delay_s" in params:
            params["event_delay_s"] = eta * params["event_delay_s"]

    # 更新场景参数
    scenario.params = params

    # 更新描述以反映 eta
    scenario.description = f"{base.description} [eta={eta:.2f}]"

    return scenario
