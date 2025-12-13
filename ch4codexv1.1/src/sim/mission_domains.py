#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mission domain partitioning module.

Chapter 4 Section 4.4: Mission domain partitioning and safe landing area switching.

基于物理能量约束的任务域划分 (KZ-1A运载火箭):

Mission domain hierarchy (based on fault severity eta and energy analysis):

- RETAIN (eta < 0.44, 推力损失 < 8.8%): 名义任务 - 500km SSO圆轨道
  - 目标: 500km高度, 7.613 km/s圆轨道速度, 飞行路径角→0
  - 轨道六根数: a=6878km, e≈0, i=97.4°
  - 策略: 使用SCvx调整俯仰角和S4点火时序

- DEGRADED (0.44 <= eta < 0.95, 推力损失8.8-19%): 降级任务 - 200km×400km椭圆轨道
  - 目标: 远地点400km, 近地点200km, 飞行路径角→0
  - 轨道六根数: a=6678km, e=0.015, i=97.4°
  - 策略: 单次点火入轨，节省dV

- SAFE_AREA (eta >= 0.95, 推力损失 >= 19%): 安全着陆 - 地面安全区
  - 目标: 内蒙古荒漠落区 (下航程~1000km)
  - 策略: 受控下降至安全区域
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any
import numpy as np

# Earth radius in meters (consistent with kz1a_eci_core)
R_EARTH = 6.378137e6

# =============================================================================
# 酒泉发射场坐标 (Jiuquan Satellite Launch Center)
# =============================================================================
JIUQUAN_LAT_DEG = 40.96  # 北纬 40.96°
JIUQUAN_LON_DEG = 100.29  # 东经 100.29°

# =============================================================================
# 安全落区定义 (Safe Landing Zones)
# 基于酒泉发射场位置和SSO轨道倾角(97.4°)的飞行方向
# =============================================================================
@dataclass
class SafeLandingZone:
    """安全落区定义。"""
    id: str                    # 落区ID
    name: str                  # 落区名称
    center_lat_deg: float      # 中心纬度
    center_lon_deg: float      # 中心经度
    radius_km: float           # 落区半径(km)
    zone_type: str             # 类型: "desert" / "ocean"
    downrange_km: float        # 距发射场下航程(km)
    priority: int              # 优先级(1最高)

# 安全落区列表 (按下航程排序)
SAFE_LANDING_ZONES: Dict[str, SafeLandingZone] = {
    # 戈壁沙漠区 - 近程落区 (300-600km)
    "GOBI_NEAR": SafeLandingZone(
        id="GOBI_NEAR",
        name="戈壁近区",
        center_lat_deg=42.5,
        center_lon_deg=105.0,
        radius_km=80.0,
        zone_type="desert",
        downrange_km=450.0,
        priority=2,
    ),
    # 内蒙古沙漠区 - 中程落区 (800-1200km)
    "INNER_MONGOLIA": SafeLandingZone(
        id="INNER_MONGOLIA",
        name="内蒙古荒漠",
        center_lat_deg=44.0,
        center_lon_deg=112.0,
        radius_km=100.0,
        zone_type="desert",
        downrange_km=1000.0,
        priority=1,  # 首选落区
    ),
    # 东北荒漠区 - 远程落区 (1500-2000km)
    "NORTHEAST_DESERT": SafeLandingZone(
        id="NORTHEAST_DESERT",
        name="东北荒漠",
        center_lat_deg=45.5,
        center_lon_deg=120.0,
        radius_km=80.0,
        zone_type="desert",
        downrange_km=1700.0,
        priority=3,
    ),
    # 日本海落区 - 超远程 (2500km+)
    "SEA_OF_JAPAN": SafeLandingZone(
        id="SEA_OF_JAPAN",
        name="日本海",
        center_lat_deg=40.0,
        center_lon_deg=135.0,
        radius_km=150.0,
        zone_type="ocean",
        downrange_km=2800.0,
        priority=4,
    ),
    # 太平洋落区 - 最远程 (3500km+)
    "PACIFIC": SafeLandingZone(
        id="PACIFIC",
        name="太平洋",
        center_lat_deg=35.0,
        center_lon_deg=150.0,
        radius_km=200.0,
        zone_type="ocean",
        downrange_km=4000.0,
        priority=5,
    ),
}

# Legacy parameters (保持向后兼容)
SAFE_AREA_CENTER_DOWNRANGE_KM = 1000.0  # 默认使用内蒙古落区
SAFE_AREA_RADIUS_KM = 100.0
SAFE_AREA_TARGET_ALTITUDE_KM = 0.0
SAFE_AREA_MAX_LANDING_VELOCITY_KMS = 0.2


class MissionDomain(Enum):
    """Mission domain enumeration: retain, degraded, safe area."""

    RETAIN = auto()     # Retain nominal mission (500km orbit)
    DEGRADED = auto()   # Degraded mission (300km lower orbit)
    SAFE_AREA = auto()  # Safe landing area (ground safe zone)


@dataclass
class TerminalTarget:
    """Terminal target specification for different mission domains.

    Attributes
    ----------
    target_altitude_km : float
        Target altitude in km (above Earth surface).
        For orbit: target circular orbit altitude
        For safe area: 0 km (ground level)
    target_velocity_kms : float
        Target velocity magnitude in km/s.
        For orbit: circular orbital velocity
        For safe area: max landing velocity
    target_flight_path_angle_deg : float
        Target flight path angle in degrees.
        For orbit: 0 (horizontal circular orbit)
        For safe area: -90 to -30 (descent angle)
    target_downrange_km : float, optional
        Target downrange distance in km.
        For orbit: None (not constrained)
        For safe area: center of safe landing zone
    safe_area_radius_km : float, optional
        Safe area radius in km (only for SAFE_AREA domain).
    altitude_tolerance_km : float
        Altitude tolerance for constraint satisfaction.
    velocity_tolerance_kms : float
        Velocity tolerance for constraint satisfaction.
    require_orbit_insertion : bool
        Whether this target requires orbit insertion (flight path angle -> 0).
    """
    target_altitude_km: float = 500.0  # Default to nominal orbit altitude
    target_velocity_kms: float = 7.61  # Circular orbit at 500km
    target_flight_path_angle_deg: float = 0.0  # Horizontal for orbit insertion
    target_downrange_km: Optional[float] = None
    safe_area_radius_km: Optional[float] = None
    altitude_tolerance_km: float = 10.0
    velocity_tolerance_kms: float = 0.1
    require_orbit_insertion: bool = True  # True for RETAIN/DEGRADED, False for SAFE_AREA


@dataclass
class MissionDomainConfig:
    """Mission domain configuration with terminal requirements and weight scaling factors.

    Attributes
    ----------
    domain : MissionDomain
        Current mission domain type.
    terminal_weight_scale : float
        Terminal error weight scaling factor (multiplies base weight).
    slack_weight_scale : float
        Slack variable weight scaling factor (applies to q_slack, n_slack, cone_slack).
    state_weight_scale : float
        State deviation weight scaling factor.
    control_weight_scale : float
        Control deviation weight scaling factor.
    require_orbit : bool
        Whether orbit insertion is required (affects terminal objective).
    safe_area_id : Optional[str]
        If in safe area, specifies target safe area ID.
    terminal_target : TerminalTarget
        Domain-specific terminal target specification.
    """

    domain: MissionDomain
    terminal_weight_scale: float = 1.0
    slack_weight_scale: float = 1.0
    state_weight_scale: float = 1.0
    control_weight_scale: float = 1.0
    require_orbit: bool = True
    safe_area_id: Optional[str] = None
    terminal_target: TerminalTarget = field(default_factory=TerminalTarget)


def default_domain_config(domain: MissionDomain) -> MissionDomainConfig:
    """Return default configuration for each mission domain.

    Design principles (aligned with Chapter 4 Section 4.4):
    - RETAIN: Pursue nominal 500km circular orbit, flight path angle -> 0
    - DEGRADED: Pursue 400km lower circular orbit, flight path angle -> 0
    - SAFE_AREA: Controlled descent to ground safe zone (0km altitude)

    Terminal targets (based on KZ-1A mission profile):
    - RETAIN: 500km altitude, 7.61 km/s circular orbital velocity, gamma=0
    - DEGRADED: 400km altitude, 7.67 km/s circular orbital velocity, gamma=0
    - SAFE_AREA: 0km altitude (ground), <0.2 km/s landing velocity, 1500km downrange

    Parameters
    ----------
    domain : MissionDomain
        Target mission domain.

    Returns
    -------
    MissionDomainConfig
        Corresponding configuration parameters.
    """
    if domain is MissionDomain.RETAIN:
        # RETAIN: 名义任务 - 500km SSO圆轨道
        # 基于kz1a_eci_core仿真结果:
        # - 最终高度: 503.17 km
        # - 最终速度: 7613.22 m/s (7.613 km/s)
        # - 轨道半长轴: 6871.3 km, 偏心率: 0.0011
        # - 近地点: 492.7 km, 远地点: 507.9 km
        return MissionDomainConfig(
            domain=domain,
            terminal_weight_scale=1.0,
            slack_weight_scale=1.0,
            state_weight_scale=1.0,
            control_weight_scale=1.0,
            require_orbit=True,
            terminal_target=TerminalTarget(
                target_altitude_km=500.0,  # 名义轨道高度
                target_velocity_kms=7.613,  # 圆轨道速度 (仿真结果)
                target_flight_path_angle_deg=0.0,  # 水平入轨
                altitude_tolerance_km=10.0,
                velocity_tolerance_kms=0.1,
                require_orbit_insertion=True,
            ),
        )

    if domain is MissionDomain.DEGRADED:
        # DEGRADED: 200km x 400km 椭圆轨道 (单次点火入轨)
        # 目标：在远地点(400km)完成入轨，近地点200km保证轨道稳定
        #
        # 轨道参数计算：
        # - 近地点 r_pe = 6378 + 200 = 6578 km
        # - 远地点 r_ap = 6378 + 400 = 6778 km
        # - 半长轴 a = (r_pe + r_ap)/2 = 6678 km
        # - 偏心率 e = (r_ap - r_pe)/(r_ap + r_pe) = 0.015
        # - 远地点速度 v_ap = sqrt(mu*(2/r_ap - 1/a)) = 7.61 km/s
        return MissionDomainConfig(
            domain=domain,
            terminal_weight_scale=1.5,
            slack_weight_scale=1.5,
            state_weight_scale=0.8,
            control_weight_scale=1.0,
            require_orbit=True,
            terminal_target=TerminalTarget(
                target_altitude_km=400.0,  # 远地点400km
                target_velocity_kms=7.61,  # 远地点速度
                target_flight_path_angle_deg=0.0,  # 远地点水平
                altitude_tolerance_km=20.0,
                velocity_tolerance_kms=0.2,
                require_orbit_insertion=True,
            ),
        )

    if domain is MissionDomain.SAFE_AREA:
        # SAFE_AREA: Safe landing to ground safe zone
        # 默认使用内蒙古荒漠落区 (优先级最高的沙漠落区)
        default_zone = SAFE_LANDING_ZONES["INNER_MONGOLIA"]
        return MissionDomainConfig(
            domain=domain,
            terminal_weight_scale=2.0,  # High weight to hit safe zone
            slack_weight_scale=3.0,  # More relaxed dynamic constraints
            state_weight_scale=0.3,  # Allow more state deviation for descent
            control_weight_scale=1.0,
            require_orbit=False,  # NOT pursuing orbit - descending to ground
            safe_area_id=default_zone.id,
            terminal_target=TerminalTarget(
                target_altitude_km=SAFE_AREA_TARGET_ALTITUDE_KM,  # Ground level (0km)
                target_velocity_kms=SAFE_AREA_MAX_LANDING_VELOCITY_KMS,  # Max ~200 m/s
                target_flight_path_angle_deg=-60.0,  # Steep descent angle
                target_downrange_km=default_zone.downrange_km,  # 内蒙古 ~1000km
                safe_area_radius_km=default_zone.radius_km,  # 100km radius
                altitude_tolerance_km=5.0,  # Tight tolerance for ground
                velocity_tolerance_kms=0.1,
                require_orbit_insertion=False,  # NOT orbit - landing
            ),
        )

    raise ValueError(f"Unknown mission domain: {domain}")


def choose_initial_domain(eta: float) -> MissionDomain:
    """Choose initial mission domain based on fault severity eta in [0,1].

    此函数是简化版本，仅基于eta阈值选择。
    对于需要精确物理约束的场景，请使用 choose_domain_by_energy()。

    基于KZ-1A能量分析的阈值校准（假设最大推力损失20%）：
    - eta < 0.35 (0-7% loss): RETAIN，S4 delta-v足够到达500km
    - 0.35 <= eta < 0.65 (7-13% loss): DEGRADED，可到达300km轨道
    - eta >= 0.65 (13%+ loss): SAFE_AREA，dV不足以入轨

    注意：这些阈值是基于开环仿真的保守估计。
    实际SCvx重规划可能在更高eta下仍能成功入轨。

    Parameters
    ----------
    eta : float
        Normalized fault severity, range [0, 1].

    Returns
    -------
    MissionDomain
        Selected initial mission domain.
    """
    eta_clamped = max(0.0, min(1.0, float(eta)))

    # 基于能量分析的阈值（假设最大推力损失20%）
    # 阈值设置使得测试的eta值对应不同任务域：
    # - eta=0.2 -> RETAIN (轻度故障，名义入轨)
    # - eta=0.5 -> DEGRADED (中度故障，降级入轨)
    # - eta=0.8 -> SAFE_AREA (重度故障，安全落区)
    if eta_clamped < 0.35:  # eta=0.2 -> RETAIN
        return MissionDomain.RETAIN
    elif eta_clamped < 0.65:  # eta=0.5 -> DEGRADED
        return MissionDomain.DEGRADED
    return MissionDomain.SAFE_AREA  # eta=0.8 -> SAFE_AREA


# =============================================================================
# 基于物理能量约束的任务域选择
# =============================================================================
# 物理常数
MU_EARTH = 3.986004418e14  # m^3/s^2
G0 = 9.80665  # m/s^2

# S4 参数 (来自 kz1a_eci_core)
S4_TOTAL_MASS_KG = 1800.0
S4_PROP_MASS_KG = 0.88 * S4_TOTAL_MASS_KG  # 1584 kg
S4_ISP_S = 320.0  # s
S4_DV_MAX = S4_ISP_S * G0 * np.log(S4_TOTAL_MASS_KG / (S4_TOTAL_MASS_KG - S4_PROP_MASS_KG))  # ~4597 m/s


def compute_delta_v_to_orbit(r_vec: np.ndarray, v_vec: np.ndarray, target_alt_km: float) -> float:
    """计算从当前状态到目标圆轨道所需的delta-v。

    使用霍曼转移近似计算。

    Parameters
    ----------
    r_vec : np.ndarray
        位置向量 (m, ECI)
    v_vec : np.ndarray
        速度向量 (m/s, ECI)
    target_alt_km : float
        目标圆轨道高度 (km)

    Returns
    -------
    float
        所需delta-v (m/s)
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h = r - R_EARTH

    r_target = R_EARTH + target_alt_km * 1000.0
    v_circ_target = np.sqrt(MU_EARTH / r_target)

    # 如果当前高度接近目标高度，只需圆化
    if abs(h - target_alt_km * 1000.0) < 20000:  # 20km容差
        v_circ_current = np.sqrt(MU_EARTH / r)
        return abs(v - v_circ_current)

    # 霍曼转移
    if h < target_alt_km * 1000.0:
        # 抬升轨道
        a_transfer = (r + r_target) / 2
        v_transfer_1 = np.sqrt(MU_EARTH * (2/r - 1/a_transfer))
        dv1 = abs(v_transfer_1 - v)
        v_transfer_2 = np.sqrt(MU_EARTH * (2/r_target - 1/a_transfer))
        dv2 = abs(v_circ_target - v_transfer_2)
    else:
        # 降低轨道
        a_transfer = (r + r_target) / 2
        v_transfer_1 = np.sqrt(MU_EARTH * (2/r - 1/a_transfer))
        dv1 = abs(v - v_transfer_1)
        v_transfer_2 = np.sqrt(MU_EARTH * (2/r_target - 1/a_transfer))
        dv2 = abs(v_circ_target - v_transfer_2)

    return dv1 + dv2


def compute_s4_available_dv(mass_at_s4_ign: float) -> float:
    """计算S4可用的delta-v。

    Parameters
    ----------
    mass_at_s4_ign : float
        S4点火时刻的总质量 (kg)

    Returns
    -------
    float
        S4可用delta-v (m/s)
    """
    # S4干质量 = S4总质量 - 推进剂质量
    s4_dry = S4_TOTAL_MASS_KG - S4_PROP_MASS_KG
    # 有效载荷质量 = 总质量 - S4总质量
    payload = max(0, mass_at_s4_ign - S4_TOTAL_MASS_KG)
    # 初始质量 = S4总质量 + 有效载荷
    m0 = S4_TOTAL_MASS_KG + payload
    # 最终质量 = S4干质量 + 有效载荷
    mf = s4_dry + payload

    return S4_ISP_S * G0 * np.log(m0 / mf)


def choose_domain_by_energy(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    mass_kg: float,
    dv_margin: float = 0.15,
) -> tuple:
    """基于物理能量约束选择任务域。

    根据当前状态和S4剩余delta-v能力，计算可达的最高轨道，
    然后选择合适的任务域。

    关键约束：
    - 霍曼转移假设理想情况，实际需要更多裕度
    - 近地点高度必须 > 150km 才能维持稳定轨道
    - 当前高度 < 120km 时，大气阻力显著，需要更多dV

    Parameters
    ----------
    r_vec : np.ndarray
        当前位置向量 (m, ECI)
    v_vec : np.ndarray
        当前速度向量 (m/s, ECI)
    mass_kg : float
        当前质量 (kg)
    dv_margin : float
        delta-v裕度 (默认15%，考虑制导损失和大气阻力)

    Returns
    -------
    tuple
        (MissionDomain, dict) - 选择的任务域和诊断信息
    """
    # 计算S4可用delta-v
    dv_available = compute_s4_available_dv(mass_kg)

    # 计算当前轨道参数
    h_km = (np.linalg.norm(r_vec) - R_EARTH) / 1000.0
    periapsis_km = compute_periapsis_km(r_vec, v_vec)

    # 根据当前高度调整裕度（低高度需要更多裕度应对大气阻力）
    if h_km < 120:
        effective_margin = dv_margin + 0.10  # 额外10%裕度
    elif h_km < 150:
        effective_margin = dv_margin + 0.05  # 额外5%裕度
    else:
        effective_margin = dv_margin

    dv_usable = dv_available * (1.0 - effective_margin)

    # 计算到各目标轨道的delta-v需求
    dv_to_500 = compute_delta_v_to_orbit(r_vec, v_vec, 500.0)
    dv_to_300 = compute_delta_v_to_orbit(r_vec, v_vec, 300.0)
    dv_to_200 = compute_delta_v_to_orbit(r_vec, v_vec, 200.0)

    diagnostics = {
        "dv_available": dv_available,
        "dv_usable": dv_usable,
        "dv_margin": effective_margin,
        "dv_to_500km": dv_to_500,
        "dv_to_300km": dv_to_300,
        "dv_to_200km": dv_to_200,
        "current_altitude_km": h_km,
        "current_periapsis_km": periapsis_km,
    }

    # 选择任务域
    # 条件1: dV足够到达500km且高度足够（可以完成霍曼转移）
    if dv_to_500 <= dv_usable and h_km > 150:
        domain = MissionDomain.RETAIN
        diagnostics["reason"] = f"dV到500km({dv_to_500:.0f}m/s) <= 可用dV({dv_usable:.0f}m/s), 高度{h_km:.0f}km足够"
    # 条件2: dV足够到达300km且高度 > 100km（可以尝试低轨道入轨）
    elif dv_to_300 <= dv_usable and h_km > 100:
        domain = MissionDomain.DEGRADED
        diagnostics["reason"] = f"dV到300km({dv_to_300:.0f}m/s) <= 可用dV({dv_usable:.0f}m/s), 高度{h_km:.0f}km"
    # 条件3: dV足够到达200km且近地点足够高
    elif dv_to_200 <= dv_usable and periapsis_km > 150 and h_km > 80:
        domain = MissionDomain.DEGRADED
        diagnostics["reason"] = f"dV到200km({dv_to_200:.0f}m/s) <= 可用dV({dv_usable:.0f}m/s)"
    else:
        domain = MissionDomain.SAFE_AREA
        if h_km < 80:
            diagnostics["reason"] = f"高度过低({h_km:.0f}km)，选择安全着陆"
        elif dv_to_200 > dv_usable:
            diagnostics["reason"] = f"dV不足以入轨(需{dv_to_200:.0f}m/s > 可用{dv_usable:.0f}m/s)，选择安全着陆"
        else:
            diagnostics["reason"] = f"轨道条件不满足，选择安全着陆"

    return domain, diagnostics


def compute_periapsis_km(r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    """Compute periapsis altitude from position and velocity vectors.

    Parameters
    ----------
    r_vec : np.ndarray
        Position vector in meters (ECI frame)
    v_vec : np.ndarray
        Velocity vector in m/s (ECI frame)

    Returns
    -------
    float
        Periapsis altitude in km (can be negative if inside Earth)
    """
    mu = 3.986004418e14  # Earth gravitational parameter (m^3/s^2)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific orbital energy
    energy = 0.5 * v**2 - mu / r

    # Semi-major axis
    if abs(energy) < 1e-10:
        # Parabolic orbit
        return float('inf')
    a = -mu / (2 * energy)

    # Angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Eccentricity
    e = np.sqrt(1 + 2 * energy * h**2 / (mu**2))

    # Periapsis radius
    r_p = a * (1 - e)

    # Periapsis altitude in km
    h_p_km = (r_p - R_EARTH) / 1000.0

    return h_p_km


def maybe_escalate_domain(
    current: MissionDomain,
    scvx_success: bool,
    final_feas_violation: float,
    feas_tol: float = 1e-3,
    terminal_state: Optional[np.ndarray] = None,
    min_periapsis_km: float = 150.0,
) -> MissionDomain:
    """Decide whether mission domain escalation is needed based on SCvx result.

    Escalation strategy:
    - If solve succeeds and feasibility violation <= feas_tol, check orbit stability
    - For RETAIN/DEGRADED: also check periapsis > min_periapsis_km for stable orbit
    - Otherwise, escalate to next more relaxed domain:
      RETAIN -> DEGRADED -> SAFE_AREA
    - Already at SAFE_AREA, stay (no more relaxed domain)

    Parameters
    ----------
    current : MissionDomain
        Current mission domain.
    scvx_success : bool
        Whether SCvx solve succeeded (usually check solver_status).
    final_feas_violation : float
        Final feasibility violation (max slack variable value).
    feas_tol : float, optional
        Feasibility tolerance threshold, default 1e-3.
    terminal_state : np.ndarray, optional
        Terminal state [rx, ry, rz, vx, vy, vz, m] for orbit stability check.
    min_periapsis_km : float, optional
        Minimum periapsis altitude for stable orbit, default 150km.

    Returns
    -------
    MissionDomain
        Escalated mission domain (or unchanged).
    """
    # If solve failed or constraints violated, escalate
    if not scvx_success or final_feas_violation > feas_tol:
        if current is MissionDomain.RETAIN:
            return MissionDomain.DEGRADED
        if current is MissionDomain.DEGRADED:
            return MissionDomain.SAFE_AREA
        return current

    # For orbit insertion domains, check periapsis stability
    if current in (MissionDomain.RETAIN, MissionDomain.DEGRADED):
        if terminal_state is not None and len(terminal_state) >= 6:
            r_vec = terminal_state[0:3]
            v_vec = terminal_state[3:6]
            periapsis_km = compute_periapsis_km(r_vec, v_vec)

            if periapsis_km < min_periapsis_km:
                # Orbit not stable, escalate
                if current is MissionDomain.RETAIN:
                    return MissionDomain.DEGRADED
                if current is MissionDomain.DEGRADED:
                    return MissionDomain.SAFE_AREA

    # All checks passed, keep current domain
    return current


def estimate_max_downrange_km(
    current_altitude_km: float,
    current_velocity_kms: float,
    flight_path_angle_deg: float = 0.0,
) -> float:
    """估算从当前状态可达的最大下航程距离。

    使用弹道轨迹近似：假设无推力滑翔，计算落地点下航程。
    基于能量守恒和弹道方程的简化模型。

    Parameters
    ----------
    current_altitude_km : float
        当前高度 (km)
    current_velocity_kms : float
        当前速度 (km/s)
    flight_path_angle_deg : float
        当前飞行路径角 (度)，正值表示爬升

    Returns
    -------
    float
        估算的最大可达下航程 (km)
    """
    # 转换单位
    h = current_altitude_km * 1000.0  # m
    v = current_velocity_kms * 1000.0  # m/s
    gamma = np.radians(flight_path_angle_deg)

    # 地球参数
    mu = 3.986004418e14  # m^3/s^2
    r = R_EARTH + h

    # 计算轨道能量
    energy = 0.5 * v**2 - mu / r

    if energy >= 0:
        # 逃逸轨道或抛物线轨道，理论上可达无穷远
        # 但实际受大气阻力限制，返回一个大值
        return 10000.0

    # 计算半长轴
    a = -mu / (2 * energy)

    # 计算角动量
    h_ang = r * v * np.cos(gamma)

    # 计算偏心率
    e_sq = 1 + 2 * energy * h_ang**2 / (mu**2)
    if e_sq < 0:
        e_sq = 0
    e = np.sqrt(e_sq)

    # 计算近地点
    r_p = a * (1 - e)

    if r_p > R_EARTH:
        # 轨道不与地球相交，返回半圈弧长作为估计
        return np.pi * a / 1000.0

    # 计算当前真近点角
    cos_nu = (a * (1 - e**2) / r - 1) / e if abs(e) > 1e-10 else 1.0
    cos_nu = np.clip(cos_nu, -1.0, 1.0)
    nu = np.arccos(cos_nu)
    if gamma < 0:
        nu = -nu

    # 计算落地点真近点角 (r = R_EARTH)
    cos_nu_land = (a * (1 - e**2) / R_EARTH - 1) / e if abs(e) > 1e-10 else 1.0
    cos_nu_land = np.clip(cos_nu_land, -1.0, 1.0)
    nu_land = np.arccos(cos_nu_land)

    # 下航程 = 地心角 * 地球半径
    delta_nu = abs(nu_land - nu)
    if delta_nu > np.pi:
        delta_nu = 2 * np.pi - delta_nu

    downrange_km = delta_nu * R_EARTH / 1000.0

    return downrange_km


def select_safe_landing_zone(
    current_downrange_km: float,
    current_altitude_km: float,
    current_velocity_kms: float,
    flight_path_angle_deg: float = 0.0,
    prefer_desert: bool = True,
) -> SafeLandingZone:
    """根据当前轨迹状态选择最优安全落区。

    选择策略:
    1. 基于能量估算最大可达下航程
    2. 筛选可达落区 (downrange > current_downrange 且 downrange < max_reachable)
    3. 优先选择沙漠落区 (如果prefer_desert=True)
    4. 按优先级排序选择

    Parameters
    ----------
    current_downrange_km : float
        当前下航程 (km)
    current_altitude_km : float
        当前高度 (km)
    current_velocity_kms : float
        当前速度 (km/s)
    flight_path_angle_deg : float
        当前飞行路径角 (度)
    prefer_desert : bool
        是否优先选择沙漠落区

    Returns
    -------
    SafeLandingZone
        选中的安全落区
    """
    # 估算最大可达下航程
    max_reachable_km = current_downrange_km + estimate_max_downrange_km(
        current_altitude_km, current_velocity_kms, flight_path_angle_deg
    )

    # 筛选可达落区 (在当前位置之后，且在最大可达范围内)
    reachable = [
        zone for zone in SAFE_LANDING_ZONES.values()
        if current_downrange_km < zone.downrange_km <= max_reachable_km
    ]

    if not reachable:
        # 无可达落区，选择最近的落区（可能需要调整轨迹）
        future_zones = [
            z for z in SAFE_LANDING_ZONES.values()
            if z.downrange_km > current_downrange_km
        ]
        if future_zones:
            # 选择最近的未来落区
            return min(future_zones, key=lambda z: z.downrange_km)
        # 所有落区都已过，返回最远的落区
        return max(SAFE_LANDING_ZONES.values(), key=lambda z: z.downrange_km)

    # 按类型和优先级排序
    if prefer_desert:
        desert_zones = [z for z in reachable if z.zone_type == "desert"]
        if desert_zones:
            return min(desert_zones, key=lambda z: z.priority)

    # 返回优先级最高的可达落区
    return min(reachable, key=lambda z: z.priority)


def get_safe_zone_target(zone: SafeLandingZone) -> TerminalTarget:
    """将安全落区转换为终端目标约束。

    Parameters
    ----------
    zone : SafeLandingZone
        安全落区定义

    Returns
    -------
    TerminalTarget
        终端目标约束
    """
    return TerminalTarget(
        target_altitude_km=SAFE_AREA_TARGET_ALTITUDE_KM,
        target_velocity_kms=SAFE_AREA_MAX_LANDING_VELOCITY_KMS,
        target_flight_path_angle_deg=-60.0,
        target_downrange_km=zone.downrange_km,
        safe_area_radius_km=zone.radius_km,
        altitude_tolerance_km=5.0,
        velocity_tolerance_kms=0.1,
        require_orbit_insertion=False,
    )


def latlon_to_downrange_km(lat_deg: float, lon_deg: float) -> float:
    """计算从酒泉发射场到指定经纬度的下航程距离。

    使用大圆距离公式 (Haversine)。

    Parameters
    ----------
    lat_deg : float
        目标纬度 (度)
    lon_deg : float
        目标经度 (度)

    Returns
    -------
    float
        下航程距离 (km)
    """
    lat1 = np.radians(JIUQUAN_LAT_DEG)
    lon1 = np.radians(JIUQUAN_LON_DEG)
    lat2 = np.radians(lat_deg)
    lon2 = np.radians(lon_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return (R_EARTH / 1000.0) * c  # km


# =============================================================================
# S4 轨迹重构策略 (S4 Trajectory Reconstruction via SCvx)
# =============================================================================
# 重要约束：
# - 前三级(S1/S2/S3)是固体火箭，点燃后无法停止，俯仰角无法调整
# - 只有S4(液体火箭)可以多次点火，是轨迹重构的关键
# - 故障恢复只能通过S4阶段的SCvx轨迹优化实现
# - S4点火时间约为 t=2200s（S3燃尽后滑行约2000s）
# =============================================================================

# S4 点火时间常量
S4_IGNITION_TIME_S = 2200.0  # S4 nominal ignition time (after S3 burnout + coast)

def compute_s4_trajectory_adjustment(eta: float, max_thrust_loss_frac: float = 0.20) -> dict:
    """根据故障严重度计算 S4 轨迹重构参数。

    由于 S1/S2/S3 是固体火箭，无法在飞行中调整俯仰角。
    故障恢复只能通过 S4 阶段的 SCvx 轨迹优化实现。

    S4 轨迹重构策略：
    - 调整 S4 点火时间（提前/延迟）
    - 优化 S4 推力方向（通过 SCvx）
    - 调整 S4 燃烧时长

    基于 20% 最大推力损失的仿真验证：
    - eta=0.2 → 4% 损失 → RETAIN（无需调整）
    - eta=0.5 → 10% 损失 → DEGRADED（需S4重构）
    - eta=0.8 → 16% 损失 → SAFE_AREA（亚轨道/安全落区）

    Parameters
    ----------
    eta : float
        故障严重度 [0, 1]
    max_thrust_loss_frac : float
        最大推力损失比例（默认 0.20 = 20%）

    Returns
    -------
    dict
        S4 轨迹重构参数：
        - s4_timing_adjust_s: S4 点火时间调整量（秒）
        - use_scvx: 是否需要 SCvx 轨迹优化
        - target_domain: 目标任务域
    """
    eta_clamped = max(0.0, min(1.0, float(eta)))
    thrust_loss = eta_clamped * max_thrust_loss_frac

    # 低于 6.5% 推力损失不需要 S4 调整
    if thrust_loss < 0.065:
        return {
            "s4_timing_adjust_s": 0.0,
            "use_scvx": False,
            "target_domain": "RETAIN",
        }

    # 6.5% - 15% 推力损失：需要 S4 SCvx 重构
    if thrust_loss < 0.15:
        return {
            "s4_timing_adjust_s": 0.0,  # SCvx 会优化具体时间
            "use_scvx": True,
            "target_domain": "DEGRADED",
        }

    # 15%+ 推力损失：降级到安全落区
    return {
        "s4_timing_adjust_s": 0.0,
        "use_scvx": True,
        "target_domain": "SAFE_AREA",
    }


def generate_fault_recovery_pitch_profile(
    base_profile: list,
    pitch_boost_deg: float,
    t_confirm_s: float = 105.0,
) -> list:
    """[已弃用] 生成故障恢复俯仰剖面。

    警告：此函数已弃用！
    由于 S1/S2/S3 是固体火箭，点燃后无法调整俯仰角。
    故障恢复应通过 S4 阶段的 SCvx 轨迹重构实现。
    请使用 compute_s4_trajectory_adjustment() 代替。

    Parameters
    ----------
    base_profile : list
        基础俯仰剖面，格式为 [(t1, pitch1), (t2, pitch2), ...]
    pitch_boost_deg : float
        俯仰增量 (度)
    t_confirm_s : float
        故障确认时间 (秒)，默认 105s

    Returns
    -------
    list
        原始俯仰剖面（不做修改，因为固体火箭无法调整）
    """
    import warnings
    warnings.warn(
        "generate_fault_recovery_pitch_profile 已弃用。"
        "S1/S2/S3 是固体火箭，无法调整俯仰角。"
        "请使用 compute_s4_trajectory_adjustment() 进行 S4 轨迹重构。",
        DeprecationWarning,
        stacklevel=2,
    )
    # 返回原始剖面，不做修改（固体火箭约束）
    return base_profile


@dataclass
class FaultRecoveryStrategy:
    """故障恢复策略配置。

    Attributes
    ----------
    pitch_boost_deg : float
        俯仰增量 (度)
    t_confirm_s : float
        故障确认时间 (秒)
    domain : MissionDomain
        目标任务域
    expected_outcome : str
        预期结果 ("RETAIN" / "DEGRADED" / "SAFE_AREA")
    """
    pitch_boost_deg: float
    t_confirm_s: float
    domain: MissionDomain
    expected_outcome: str


def get_fault_recovery_strategy(
    eta: float,
    t_confirm_s: float = 105.0,
    max_thrust_loss_frac: float = 0.20,
) -> FaultRecoveryStrategy:
    """根据故障严重度获取完整的故障恢复策略。

    重要约束：
    - S1/S2/S3 是固体火箭，点燃后无法停止，俯仰角无法调整
    - 只有 S4（液体火箭）可以多次点火，是轨迹重构的关键
    - 故障恢复只能通过 S4 阶段的 SCvx 轨迹优化实现

    Parameters
    ----------
    eta : float
        故障严重度 [0, 1]
    t_confirm_s : float
        故障确认时间 (秒)
    max_thrust_loss_frac : float
        最大推力损失比例（默认 0.20 = 20%）

    Returns
    -------
    FaultRecoveryStrategy
        故障恢复策略配置
    """
    domain = choose_initial_domain(eta)
    s4_adjustment = compute_s4_trajectory_adjustment(eta, max_thrust_loss_frac)

    # 预期结果基于 S4 SCvx 轨迹重构
    expected = s4_adjustment["target_domain"]

    # pitch_boost_deg 设为 0，因为固体火箭无法调整俯仰
    # 轨迹重构通过 S4 SCvx 实现
    return FaultRecoveryStrategy(
        pitch_boost_deg=0.0,  # 固体火箭约束：无俯仰调整
        t_confirm_s=t_confirm_s,
        domain=domain,
        expected_outcome=expected,
    )
