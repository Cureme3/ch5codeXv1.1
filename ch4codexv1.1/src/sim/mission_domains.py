#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mission domain partitioning module.

Chapter 4 Section 4.4: Mission domain partitioning and safe landing area switching.

Mission domain hierarchy (based on fault severity eta):
- RETAIN (eta < 0.3): Nominal mission, pursue orbit insertion at 500km circular orbit
  - Target: 500km altitude, ~7.61 km/s circular velocity, flight path angle -> 0
  - Strategy: Continue mission using SCvx to adjust pitch and S4 burn timing

- DEGRADED (0.3 <= eta < 0.7): Degraded mission, pursue lower orbit at 300km
  - Target: 300km altitude, ~7.73 km/s circular velocity, flight path angle -> 0
  - Strategy: Lower orbit insertion, adjust pitch and S4 burn timing

- SAFE_AREA (eta >= 0.7): Safe landing area, target ground safe zone
  - Target: Ground landing within safe zone radius (centered at ~1500km downrange)
  - Strategy: Controlled descent to safe area, adjust pitch and S4 for safe landing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any
import numpy as np

# Earth radius in meters (consistent with kz1a_eci_core)
R_EARTH = 6.378137e6

# Safe area parameters for SAFE_AREA domain
# Centered at approximately 1500km downrange from launch site (Jiuquan)
# This corresponds to potential recovery zones in inner Mongolia/northern China
SAFE_AREA_CENTER_DOWNRANGE_KM = 1500.0  # Center of safe zone (downrange from launch)
SAFE_AREA_RADIUS_KM = 100.0  # Acceptable landing radius around center
SAFE_AREA_TARGET_ALTITUDE_KM = 0.0  # Ground level (0 km altitude)
SAFE_AREA_MAX_LANDING_VELOCITY_KMS = 0.2  # Max velocity at touchdown (~200 m/s)


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
    - DEGRADED: Pursue 300km lower circular orbit, flight path angle -> 0
    - SAFE_AREA: Controlled descent to ground safe zone (0km altitude)

    Terminal targets (based on KZ-1A mission profile):
    - RETAIN: 500km altitude, 7.61 km/s circular orbital velocity, gamma=0
    - DEGRADED: 300km altitude, 7.73 km/s circular orbital velocity, gamma=0
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
        # RETAIN: Full nominal mission - 500km circular orbit
        # Target: Circular orbit insertion with flight path angle approaching zero
        return MissionDomainConfig(
            domain=domain,
            terminal_weight_scale=1.0,
            slack_weight_scale=1.0,
            state_weight_scale=1.0,
            control_weight_scale=1.0,
            require_orbit=True,  # Must achieve circular orbit
            terminal_target=TerminalTarget(
                target_altitude_km=500.0,  # Nominal orbit altitude
                target_velocity_kms=7.61,  # Circular orbit velocity at 500km
                target_flight_path_angle_deg=0.0,  # Horizontal insertion
                altitude_tolerance_km=10.0,
                velocity_tolerance_kms=0.1,
                require_orbit_insertion=True,
            ),
        )

    if domain is MissionDomain.DEGRADED:
        # DEGRADED: Lower orbit mission - 300km circular orbit
        # Target: Lower circular orbit insertion with flight path angle approaching zero
        return MissionDomainConfig(
            domain=domain,
            terminal_weight_scale=1.0,  # Strong terminal guidance for orbit
            slack_weight_scale=1.5,  # Slightly relaxed constraints
            state_weight_scale=0.8,
            control_weight_scale=1.0,
            require_orbit=True,  # Still require orbit insertion
            terminal_target=TerminalTarget(
                target_altitude_km=300.0,  # Lower orbit - 300km
                target_velocity_kms=7.73,  # Circular orbit velocity at 300km
                target_flight_path_angle_deg=0.0,  # Horizontal insertion
                altitude_tolerance_km=15.0,  # Slightly more tolerance
                velocity_tolerance_kms=0.15,
                require_orbit_insertion=True,
            ),
        )

    if domain is MissionDomain.SAFE_AREA:
        # SAFE_AREA: Safe landing to ground safe zone
        # Target: Controlled descent to safe area (0km altitude, within safe radius)
        return MissionDomainConfig(
            domain=domain,
            terminal_weight_scale=2.0,  # High weight to hit safe zone
            slack_weight_scale=3.0,  # More relaxed dynamic constraints
            state_weight_scale=0.3,  # Allow more state deviation for descent
            control_weight_scale=1.0,
            require_orbit=False,  # NOT pursuing orbit - descending to ground
            safe_area_id="default",
            terminal_target=TerminalTarget(
                target_altitude_km=SAFE_AREA_TARGET_ALTITUDE_KM,  # Ground level (0km)
                target_velocity_kms=SAFE_AREA_MAX_LANDING_VELOCITY_KMS,  # Max ~200 m/s
                target_flight_path_angle_deg=-60.0,  # Steep descent angle
                target_downrange_km=SAFE_AREA_CENTER_DOWNRANGE_KM,  # ~1500km downrange
                safe_area_radius_km=SAFE_AREA_RADIUS_KM,  # 100km radius
                altitude_tolerance_km=5.0,  # Tight tolerance for ground
                velocity_tolerance_kms=0.1,
                require_orbit_insertion=False,  # NOT orbit - landing
            ),
        )

    raise ValueError(f"Unknown mission domain: {domain}")


def choose_initial_domain(eta: float) -> MissionDomain:
    """Choose initial mission domain based on fault severity eta in [0,1].

    Selection strategy:
    - eta < 0.3  -> RETAIN    (minor fault, attempt nominal mission)
    - 0.3 <= eta < 0.7 -> DEGRADED  (moderate fault, degraded mission)
    - eta >= 0.7 -> SAFE_AREA (severe fault, prioritize safe area)

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

    if eta_clamped < 0.3:
        return MissionDomain.RETAIN
    if eta_clamped < 0.7:
        return MissionDomain.DEGRADED
    return MissionDomain.SAFE_AREA


def maybe_escalate_domain(
    current: MissionDomain,
    scvx_success: bool,
    final_feas_violation: float,
    feas_tol: float = 1e-3,
) -> MissionDomain:
    """Decide whether mission domain escalation is needed based on SCvx result.

    Escalation strategy:
    - If solve succeeds and feasibility violation <= feas_tol, keep current domain
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

    Returns
    -------
    MissionDomain
        Escalated mission domain (or unchanged).
    """
    # If solve succeeded and constraints satisfied, keep current domain
    if scvx_success and final_feas_violation <= feas_tol:
        return current

    # Need escalation: RETAIN -> DEGRADED -> SAFE_AREA
    if current is MissionDomain.RETAIN:
        return MissionDomain.DEGRADED
    if current is MissionDomain.DEGRADED:
        return MissionDomain.SAFE_AREA

    # Already at SAFE_AREA, cannot escalate further
    return current
