#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""完整闭环测试: 故障注入 → 诊断 → SCvx重规划。"""

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
from src.opt.scvx_unified import UnifiedSCvxPlanner, UnifiedSCvxResult

R_EARTH = 6.378137e6


def run_closed_loop_test(scenario_id: str, eta: float) -> dict:
    """运行完整闭环测试。"""
    print("=" * 60)
    print(f"闭环测试: {scenario_id}, η={eta:.2f}")
    print("=" * 60)

    # 1. 故障仿真
    print("\n[1] 故障仿真...")
    fault_sim = run_fault_scenario(scenario_id, eta=eta, dt=1.0)
    states = fault_sim.states
    time = fault_sim.time

    h_final_fault = (np.linalg.norm(states[-1, 0:3]) - R_EARTH) / 1000.0
    v_final_fault = np.linalg.norm(states[-1, 3:6]) / 1000.0
    print(f"    故障开环终端: h={h_final_fault:.1f}km, v={v_final_fault:.2f}km/s")

    # 2. ESO 残差
    print("\n[2] ESO 残差估计...")
    v = states[:, 3:6]
    ax = np.diff(v[:, 0])
    az = np.diff(v[:, 2])
    axz = np.column_stack([ax, az])
    residuals, _, _ = run_eso(axz, dt=1.0)
    res_mag = np.linalg.norm(residuals, axis=1)
    print(f"    残差峰值: {np.max(res_mag):.2f}")

    # 3. PWVD 特征
    print("\n[3] PWVD 特征提取...")
    features = extract_features_from_residual(res_mag, dt=1.0, spec_method="pwvd")
    print(f"    特征: {features[:3]}")

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

    # 6. SCvx 重规划
    print("\n[6] SCvx 重规划...")

    # 设置目标
    target_h = target.target_altitude_km * 1000.0
    target_v = target.target_velocity_kms * 1000.0
    r_target = np.array([0.0, 0.0, R_EARTH + target_h])
    v_target = np.array([0.0, target_v, 0.0])

    # 初始化规划器
    planner = UnifiedSCvxPlanner(
        nodes=40,
        dt=2.0,
        r_target=r_target,
        v_target=v_target,
        max_iters=10,
        verbose=True,
    )
    planner.set_adaptive_weights(eta)

    # 准备初始猜测 (从故障轨迹采样)
    N = 40
    indices = np.linspace(0, len(states) - 1, N).astype(int)
    x_init = np.zeros((N, 7))
    x_init[:, 0:6] = states[indices, 0:6]
    x_init[:, 6] = 10000.0  # 初始质量

    u_init = np.zeros((N, 3))
    u_init[:, 2] = 50000.0  # 初始推力猜测

    # 求解
    result = planner.solve(x_init, u_init)

    # 提取结果
    r_final = np.linalg.norm(result.states[-1, 0:3])
    v_final = np.linalg.norm(result.states[-1, 3:6])
    h_final = (r_final - R_EARTH) / 1000.0
    v_final_kms = v_final / 1000.0

    print(f"\n    SCvx 结果:")
    print(f"    状态: {result.solver_status}")
    print(f"    迭代: {result.iterations}")
    print(f"    收敛: {result.converged}")
    print(f"    终端: h={h_final:.1f}km, v={v_final_kms:.2f}km/s")

    # 计算改进
    h_error_before = abs(h_final_fault - target.target_altitude_km)
    h_error_after = abs(h_final - target.target_altitude_km)
    improvement = (h_error_before - h_error_after) / max(h_error_before, 1.0) * 100

    print(f"\n    高���误差改进: {h_error_before:.1f}km → {h_error_after:.1f}km ({improvement:.1f}%)")

    return {
        "scenario_id": scenario_id,
        "eta": eta,
        "domain": domain.name,
        "h_fault": h_final_fault,
        "v_fault": v_final_fault,
        "h_replan": h_final,
        "v_replan": v_final_kms,
        "h_target": target.target_altitude_km,
        "converged": result.converged,
        "iterations": result.iterations,
        "improvement_pct": improvement,
    }


def main():
    print("=" * 60)
    print("完整闭环测试")
    print("=" * 60)

    test_cases = [
        ("F1_thrust_deg15", 0.2),
        ("F1_thrust_deg15", 0.5),
        ("F1_thrust_deg15", 0.8),
        ("F2_tvc_rate4", 0.5),
    ]

    results = []
    for scenario_id, eta in test_cases:
        try:
            result = run_closed_loop_test(scenario_id, eta)
            results.append(result)
        except Exception as e:
            print(f"\n[错误] {scenario_id} η={eta}: {e}")

    # 汇总
    print("\n" + "=" * 80)
    print("测试汇总")
    print("=" * 80)
    print(f"{'场景':<20} {'η':>5} {'域':<12} {'故障h':>8} {'重规划h':>8} {'目标h':>8} {'收敛':>5}")
    print("-" * 80)
    for r in results:
        print(f"{r['scenario_id']:<20} {r['eta']:>5.1f} {r['domain']:<12} "
              f"{r['h_fault']:>8.1f} {r['h_replan']:>8.1f} {r['h_target']:>8.0f} "
              f"{'✓' if r['converged'] else '✗':>5}")


if __name__ == "__main__":
    main()
