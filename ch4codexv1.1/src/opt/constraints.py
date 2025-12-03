import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List

MU_EARTH = 3.986004418e14
R_EARTH = 6371000.0
G0 = 9.80665

@dataclass
class OptimizationConfig:
    """汇集 SCP 约束配置和故障参数."""
    dt: float = 1.0 # 离散时间步长
    nodes: int = 40 # 离散节点数
    state_dim: int = 7   # r(3) + v(3) + m(1)
    control_dim: int = 3 # T(3)
    
    # --- 物理约束常数 ---
    # 假设 Nominal Thrust Profile (N)
    nominal_thrust_N: float = 300000.0 
    # 假设 Nominal Thrust Cone Angle (deg)
    nominal_thrust_cone_deg: float = 15.0 
    max_dynamic_pressure_Pa: float = 80000.0
    max_normal_load_G: float = 6.0

    # --- 终端目标 ---
    # 目标位置 r_target (m)
    r_target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 420000.0 + R_EARTH])) 
    # 目标速度 v_target (m/s)
    v_target: np.ndarray = field(default_factory=lambda: np.array([0.0, 7800.0, 0.0])) 

    # --- 故障等效参数 (来自诊断) ---
    # 推力降级比例 [0, 1]
    fault_thrust_drop: float = 0.0 
    # 推力锥收紧角度 (有效半角 = nom -收紧) [rad]
    fault_cone_tightening_rad: float = 0.0 
    # 终端目标软约束的松弛系数 (用于任务级切换)
    terminal_slack_penalty: float = 1.0
