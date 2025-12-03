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

SCENARIO_CATALOG: Dict[str, FaultScenario] = {
    # F1：推力降级
    # t_fault=85s 是第二级点火初期，推力 ~354 kN
    "F1_thrust_deg15": FaultScenario(
        id="F1_thrust_deg15",
        description="第二级推力降级 30%，锥角收紧至 5°",
        fault_type="thrust_degradation",
        t_fault_s=85.0,  # Changed to Stage 2 burn phase
        t_detect_s=95.0,
        t_confirm_s=105.0,
        t_plan_horizon_s=150.0,
        params={
            "degrade_frac": 0.30,  # 30% thrust loss at eta=1.0
            "theta_max_deg_after": 5.0,
        },
    ),
    "F1_severe": FaultScenario(
        id="F1_severe",
        description="F1 severe thrust degradation (50% drop)",
        fault_type="thrust_degradation",
        t_fault_s=85.0,
        t_detect_s=95.0,
        t_confirm_s=105.0,
        t_plan_horizon_s=150.0,
        params={
            "degrade_frac": 0.50,
            "theta_max_deg_after": 3.0,
        },
    ),
    # F2：TVC 速率限制 + 俯仰偏置
    # t_fault=100s 是第二级中期
    "F2_tvc_rate4": FaultScenario(
        id="F2_tvc_rate4",
        description="TVC 速率限制 2 deg/s 并叠加 -8° 俯仰偏置",
        fault_type="tvc_rate_limit",
        t_fault_s=100.0,  # Changed to middle of Stage 2
        t_detect_s=110.0,
        t_confirm_s=120.0,
        t_plan_horizon_s=140.0,
        params={
            "tvc_rate_deg_s": 2.0,  # Relaxed from 0.2 to 2.0
            "angle_bias_deg": -8.0,  # Reduced from -18.0 to -8.0
        },
    ),
    "F2_severe": FaultScenario(
        id="F2_severe",
        description="F2 severe TVC rate limit 0.5 deg/s with -15 deg bias",
        fault_type="tvc_rate_limit",
        t_fault_s=100.0,
        t_detect_s=110.0,
        t_confirm_s=120.0,
        t_plan_horizon_s=140.0,
        params={
            "tvc_rate_deg_s": 0.5,
            "angle_bias_deg": -15.0,
        },
    ),
    # F3：TVC 卡滞
    # t_fault=200s 是第三级点火阶段
    "F3_tvc_stuck3deg": FaultScenario(
        id="F3_tvc_stuck3deg",
        description="TVC 卡在 +12° 处",
        fault_type="tvc_stuck",
        t_fault_s=200.0,  # Changed to Stage 3 burn
        t_detect_s=210.0,
        t_confirm_s=220.0,
        t_plan_horizon_s=140.0,
        params={
            "stuck_angle_deg": 12.0,  # Increased from 5.0 to 12.0 for more realistic effect
        },
    ),
    "F3_severe": FaultScenario(
        id="F3_severe",
        description="F3 severe TVC stuck at +20 deg",
        fault_type="tvc_stuck",
        t_fault_s=200.0,
        t_detect_s=210.0,
        t_confirm_s=220.0,
        t_plan_horizon_s=140.0,
        params={
            "stuck_angle_deg": 20.0,  # Increased from 10.0 to 20.0
        },
    ),
    # F4：姿态传感器偏置
    # t_fault=100s 是第二级中期
    "F4_sensor_bias2deg": FaultScenario(
        id="F4_sensor_bias2deg",
        description="惯导俯仰量测出现 +5° 偏置",
        fault_type="sensor_bias",
        t_fault_s=100.0,  # Changed to Stage 2
        t_detect_s=110.0,
        t_confirm_s=120.0,
        t_plan_horizon_s=140.0,
        params={
            "sensor_bias_deg": 5.0,  # Reduced from 10.0 to 5.0
        },
    ),
    "F4_severe": FaultScenario(
        id="F4_severe",
        description="F4 severe attitude sensor bias +10 deg",
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
    # t_fault=200s 是第三级点火阶段，推力~148 kN
    "F5_event_delay5s": FaultScenario(
        id="F5_event_delay5s",
        description="第三级点火延迟 15 s",
        fault_type="event_delay",
        t_fault_s=200.0,
        t_detect_s=205.0,
        t_confirm_s=215.0,
        t_plan_horizon_s=140.0,
        params={
            "event_delay_s": 15.0,  # Reduced from 25.0 to 15.0
            "delay_thrust_scale": 0.20,  # 20% thrust during delay
        },
    ),
    "F5_severe": FaultScenario(
        id="F5_severe",
        description="F5 severe event delay 30 s",
        fault_type="event_delay",
        t_fault_s=200.0,
        t_detect_s=205.0,
        t_confirm_s=215.0,
        t_plan_horizon_s=140.0,
        params={
            "event_delay_s": 30.0,
            "delay_thrust_scale": 0.15,
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
        if "tvc_rate_deg_s" in params:
            base_rate = params["tvc_rate_deg_s"]
            nominal_rate = 10.0  # 名义 TVC 速率 deg/s
            # eta=0: 正常速率; eta=1: 最大限制（最小速率）
            params["tvc_rate_deg_s"] = nominal_rate - eta * (nominal_rate - base_rate)
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
