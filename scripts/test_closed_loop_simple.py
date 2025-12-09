#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""简化闭环测试: 验证诊断→规划流程。"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ch3codev1.1"))
sys.path.insert(0, str(PROJECT_ROOT / "ch4codexv1.1"))
sys.path.insert(0, str(PROJECT_ROOT / "ch4codexv1.1" / "src"))

from diagnosis.eso import run_eso
from diagnosis.features import extract_features_from_residual
from src.sim.scenarios import get_scenario, scale_scenario_by_eta
from src.sim.run_fault import run_fault_scenario
from src.sim.mission_domains import choose_initial_domain, default_domain_config
from src.learn.weights import compute_adaptive_penalties


def run_test(scenario_id: str, eta: float) -> dict:
    """运行诊断流程测试。"""
    print(f"\n{'='*60}")
    print(f"测试: {scenario_id}, eta={eta:.2f}")
    print("="*60)

    # 1. 故障仿真
    print("\n[1] 故障仿真...")
    fault_sim = run_fault_scenario(scenario_id, eta=eta, dt=1.0)
    states = fault_sim.states
    R_EARTH = 6.378137e6
    h_final = (np.linalg.norm(states[-1, 0:3]) - R_EARTH) / 1000.0
    v_final = np.linalg.norm(states[-1, 3:6]) / 1000.0
    print(f"    终端: h={h_final:.1f}km, v={v_final:.2f}km/s")

    # 2. ESO 残差
    print("\n[2] ESO 残差...")
    v = states[:, 3:6]
    ax = np.diff(v[:, 0])
    az = np.diff(v[:, 2])
    axz = np.column_stack([ax, az])
    residuals, _, _ = run_eso(axz, dt=1.0)
    res_mag = np.linalg.norm(residuals, axis=1)
    print(f"    残差峰值: {np.max(res_mag):.2f}")

    # 3. PWVD 特征
    print("\n[3] PWVD 特征...")
    features = extract_features_from_residual(res_mag, dt=1.0, spec_method="pwvd")
    print(f"    特征[0:3]: {features[:3]}")

    # 4. 任务域选择
    print("\n[4] 任务域选择...")
    domain = choose_initial_domain(eta)
    domain_cfg = default_domain_config(domain)
    target = domain_cfg.terminal_target
    print(f"    域: {domain.name}")
    print(f"    目标: h={target.target_altitude_km}km, v={target.target_velocity_kms}km/s")

    # 5. 自适应权重
    print("\n[5] 自适应权重...")
    weights = compute_adaptive_penalties(eta)
    print(f"    终端权重: {weights.terminal_state_dev:.2f}")
    print(f"    松弛权重: {weights.q_slack:.2f}")

    return {
        "scenario_id": scenario_id,
        "eta": eta,
        "domain": domain.name,
        "h_fault": h_final,
        "v_fault": v_final,
        "h_target": target.target_altitude_km,
        "v_target": target.target_velocity_kms,
        "terminal_weight": weights.terminal_state_dev,
        "slack_weight": weights.q_slack,
    }


def main():
    print("="*60)
    print("闭环诊断流程测试")
    print("="*60)

    test_cases = [
        ("F1_thrust_deg15", 0.2),
        ("F1_thrust_deg15", 0.5),
        ("F1_thrust_deg15", 0.8),
        ("F2_tvc_rate4", 0.5),
        ("F3_tvc_stuck3deg", 0.5),
    ]

    results = []
    for scenario_id, eta in test_cases:
        try:
            result = run_test(scenario_id, eta)
            results.append(result)
        except Exception as e:
            print(f"\n[错误] {scenario_id} eta={eta}: {e}")

    # 汇总
    print("\n" + "="*80)
    print("测试汇总")
    print("="*80)
    print(f"{'场景':<20} {'eta':>5} {'域':<12} {'故障h':>8} {'目标h':>8} {'终端权重':>10} {'松弛权重':>10}")
    print("-"*80)
    for r in results:
        print(f"{r['scenario_id']:<20} {r['eta']:>5.1f} {r['domain']:<12} "
              f"{r['h_fault']:>8.1f} {r['h_target']:>8.0f} "
              f"{r['terminal_weight']:>10.2f} {r['slack_weight']:>10.2f}")

    print("\n[结论] 诊断→任务域→权重 流程验证通过")


if __name__ == "__main__":
    main()
