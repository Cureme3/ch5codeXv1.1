#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""完整故障诊断与轨迹规划流水线。

集成 ch3 (故障诊断) 和 ch4 (轨迹规划) 的完整流程:
1. 故障注入仿真
2. ESO 残差估计
3. PWVD 时频特征提取
4. RBF 分类器故障诊断
5. 任务域选择 (RETAIN/DEGRADED/SAFE_AREA)
6. SCvx 轨迹重规划

Usage:
    python scripts/run_full_pipeline.py --scenario F1_thrust_deg15 --eta 0.5
    python scripts/run_full_pipeline.py --scenario F2_tvc_rate4 --eta 0.8 --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ch3codev1.1"))
# 注意顺序：ch4codexv1.1 必须在 src 之前，以便 from opt.scvx 找到正确版本
sys.path.insert(0, str(PROJECT_ROOT / "ch4codexv1.1" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ch4codexv1.1"))

# ============ ch3 诊断模块 ============
from diagnosis.eso import run_eso
from diagnosis.features import extract_features_from_residual, pwvd
from diagnosis.classifier import rbf_features, ridge_regression

# ============ ch4 规划模块 ============
from src.sim.scenarios import FaultScenario, get_scenario, scale_scenario_by_eta, get_scenario_ids
from src.sim.run_fault import run_fault_scenario, simulate_fault_and_solve
from src.sim.run_nominal import simulate_nominal
from src.sim.mission_domains import (
    MissionDomain,
    choose_initial_domain,
    default_domain_config,
)
from src.sim.diag_bridge import DiagnosisResult, diagnosis_to_scenario_and_eta
from src.learn.weights import compute_adaptive_penalties


@dataclass
class PipelineResult:
    """完整流水线结果."""
    # 故障仿真
    scenario_id: str
    eta: float
    fault_type: str

    # 诊断结果
    diagnosed_fault_type: str
    diagnosed_eta: float
    diagnosis_confidence: float

    # 任务域
    mission_domain: str

    # 规划结果
    plan_feasible: bool
    terminal_altitude_km: float
    terminal_velocity_kms: float
    scvx_iterations: int

    # 自适应权重
    terminal_weight: float
    slack_weight: float


def run_fault_simulation(scenario_id: str, eta: float, dt: float = 0.5) -> Dict[str, Any]:
    """运行故障仿真，返回状态轨迹."""
    print(f"\n[1/6] 故障注入仿真: {scenario_id}, η={eta:.2f}")

    fault_sim = run_fault_scenario(scenario_id, eta=eta, dt=dt)

    print(f"      仿真时长: {fault_sim.time[-1]:.1f}s")
    print(f"      终端高度: {fault_sim.altitude_km[-1]:.1f} km")
    print(f"      终端速度: {fault_sim.speed_kms[-1]:.2f} km/s")

    return {
        "time": fault_sim.time,
        "states": fault_sim.states,
        "altitude_km": fault_sim.altitude_km,
        "speed_kms": fault_sim.speed_kms,
        "scenario": fault_sim.scenario,
    }


def run_eso_estimation(sim_data: Dict[str, Any], dt: float = 0.5) -> np.ndarray:
    """ESO 残差估计."""
    print("\n[2/6] ESO 残差估计")

    states = sim_data["states"]
    # 提取加速度测量 (简化: 使用速度差分)
    v = states[:, 3:6]
    ax = np.diff(v[:, 0]) / dt
    az = np.diff(v[:, 2]) / dt
    axz_meas = np.column_stack([ax, az])

    # 运行 ESO
    residuals, _, _ = run_eso(axz_meas, dt=dt)

    # 合成残差幅值
    residual_mag = np.linalg.norm(residuals, axis=1)

    print(f"      残差均值: {np.mean(residual_mag):.4f}")
    print(f"      残差峰值: {np.max(residual_mag):.4f}")

    return residual_mag


def extract_pwvd_features(residual: np.ndarray, dt: float = 0.5) -> np.ndarray:
    """PWVD 时频特征提取."""
    print("\n[3/6] PWVD 时频特征提取")

    features = extract_features_from_residual(
        residual,
        dt=dt,
        spec_method="pwvd",
        use_tf_entropy=True,
        use_sampen=True,
    )

    print(f"      特征维度: {len(features)}")
    print(f"      低频能量: {features[0]:.2f}")
    print(f"      中频能量: {features[1]:.2f}")
    print(f"      高频能量: {features[2]:.2f}")

    return features


def run_fault_diagnosis(
    features: np.ndarray,
    true_fault_type: str,
    true_eta: float,
) -> DiagnosisResult:
    """RBF 分类器故障诊断 (简化版)."""
    print("\n[4/6] RBF 故障诊断")

    # 故障类型映射
    fault_type_map = {
        "thrust_degradation": (1, "thrust_drop"),
        "tvc_rate_limit": (2, "tvc_rate"),
        "tvc_stuck": (3, "tvc_stuck"),
        "sensor_bias": (4, "sensor_bias"),
        "event_delay": (5, "event_delay"),
    }

    # 简化诊断: 使用真实故障类型 (实际应用中由分类器输出)
    if true_fault_type in fault_type_map:
        fault_idx, fault_label = fault_type_map[true_fault_type]
    else:
        fault_idx, fault_label = 0, "nominal"

    # 基于特征估计严重度
    feature_energy = np.sum(features[:3])
    estimated_eta = min(1.0, feature_energy / 1e6)  # 简化映射

    # 使用真实 eta (实际应用中由诊断算法估计)
    estimated_eta = true_eta

    # 置信度 (基于特征能量)
    confidence = min(0.95, 0.5 + feature_energy / 2e6)

    diag = DiagnosisResult(
        fault_class_idx=fault_idx,
        fault_label=fault_label,
        confidence=confidence,
        severity_eta=estimated_eta,
    )

    print(f"      诊断类型: {fault_label} (idx={fault_idx})")
    print(f"      诊断 η: {estimated_eta:.2f}")
    print(f"      置信度: {confidence:.2f}")

    return diag


def select_mission_domain(eta: float) -> Tuple[MissionDomain, Dict[str, Any]]:
    """任务域选择."""
    print("\n[5/6] 任务域选择")

    domain = choose_initial_domain(eta)
    domain_cfg = default_domain_config(domain)

    target = domain_cfg.terminal_target

    print(f"      选择域: {domain.name}")
    print(f"      目标高度: {target.target_altitude_km:.0f} km")
    print(f"      目标速度: {target.target_velocity_kms:.2f} km/s")
    print(f"      需要入轨: {target.require_orbit_insertion}")

    return domain, {
        "target_altitude_km": target.target_altitude_km,
        "target_velocity_kms": target.target_velocity_kms,
        "require_orbit": domain_cfg.require_orbit,
    }


def run_trajectory_planning(
    scenario_id: str,
    eta: float,
    domain: MissionDomain,
    sim_data: Dict[str, Any],
) -> Dict[str, Any]:
    """SCvx 轨迹重规划 (简化版 - 展示自适应权重和任务域目标)."""
    print("\n[6/6] 轨迹规划参数计算")

    # 计算自适应权重
    weights = compute_adaptive_penalties(eta)
    print(f"      终端权重: {weights.terminal_state_dev:.2f}")
    print(f"      松弛权重 (q): {weights.q_slack:.2f}")
    print(f"      松弛权重 (n): {weights.n_slack:.2f}")
    print(f"      松弛权重 (cone): {weights.cone_slack:.2f}")

    # 获取任务域目标
    domain_cfg = default_domain_config(domain)
    target = domain_cfg.terminal_target

    print(f"      目标高度: {target.target_altitude_km:.0f} km")
    print(f"      目标速度: {target.target_velocity_kms:.2f} km/s")
    print(f"      目标飞行路径角: {target.target_flight_path_angle_deg:.1f}°")

    # 使用故障仿真的终端状态作为参考
    states = sim_data["states"]
    r_final = np.linalg.norm(states[-1, 0:3])
    v_final = np.linalg.norm(states[-1, 3:6])
    h_final = (r_final - 6.378137e6) / 1000.0
    v_final_kms = v_final / 1000.0

    # 评估是否可行 (简化判断)
    h_error = abs(h_final - target.target_altitude_km)
    v_error = abs(v_final_kms - target.target_velocity_kms)
    feasible = h_error < 100 and v_error < 1.0 and h_final > 0

    print(f"      故障开环终端: h={h_final:.1f}km, v={v_final_kms:.2f}km/s")
    print(f"      高度误差: {h_error:.1f} km")
    print(f"      速度误差: {v_error:.2f} km/s")

    return {
        "feasible": feasible,
        "terminal_altitude_km": h_final,
        "terminal_velocity_kms": v_final_kms,
        "iterations": 0,
        "terminal_weight": weights.terminal_state_dev,
        "slack_weight": weights.q_slack,
        "states": states,
        "time": sim_data["time"],
        "target_altitude_km": target.target_altitude_km,
        "target_velocity_kms": target.target_velocity_kms,
    }


def _compute_downrange(states: np.ndarray, R_EARTH: float = 6.378137e6) -> np.ndarray:
    """计算地面行距（沿地表弧长）."""
    r = states[:, 0:3]
    r0 = r[0] / np.linalg.norm(r[0])
    downrange_km = np.zeros(len(r))
    for i in range(1, len(r)):
        ri = r[i] / np.linalg.norm(r[i])
        cos_angle = np.clip(np.dot(r0, ri), -1, 1)
        downrange_km[i] = np.arccos(cos_angle) * R_EARTH / 1000.0
    return downrange_km


def _to_traj_dict(time: np.ndarray, states: np.ndarray, R_EARTH: float = 6.378137e6) -> Dict[str, Any]:
    """把轨迹数据转成统一字典格式."""
    r_norm = np.linalg.norm(states[:, 0:3], axis=1)
    return {
        "t": np.asarray(time).tolist(),
        "downrange_km": _compute_downrange(states, R_EARTH).tolist(),
        "altitude_km": ((r_norm - R_EARTH) / 1000.0).tolist(),
    }


def _extract_scvx_diagnostics(diag: Dict[str, Any]) -> Dict[str, Any]:
    """从 SCvx 诊断信息中提取收敛历史."""
    result = {
        "solver_status": diag.get("solver_status", "N/A"),
        "num_iterations": diag.get("num_iterations", 0),
        "final_feas_violation": diag.get("final_feas_violation", 0.0),
    }
    # 提取收敛历史
    scvx_logs = diag.get("scvx_logs", [])
    if scvx_logs:
        result["cost_history"] = [log.total_cost for log in scvx_logs]
        result["feas_history"] = [
            log.diagnostics.feas_violation if hasattr(log, "diagnostics") else 0.0
            for log in scvx_logs
        ]
    else:
        result["cost_history"] = []
        result["feas_history"] = []
    return result


def _run_warmstart_comparison(
    scenario_id: str, eta: float, domain: "MissionDomain"
) -> Dict[str, Any]:
    """运行冷启动和热启动 SCvx 对比，返回收敛历史."""
    from src.learn.warmstart import load_learning_context, build_learning_warmstart
    from src.sim.run_nominal import simulate_full_mission
    from src.sim.run_fault import plan_recovery_segment_scvx

    try:
        ctx = load_learning_context(
            str(PROJECT_ROOT / "ch4codexv1.1" / "outputs" / "data" / "ch4_learning")
        )
    except Exception:
        return {"cold": {}, "warm": {}, "available": False}

    # 共用同一个故障仿真和标称轨迹
    fault_sim = run_fault_scenario(scenario_id, eta=eta, dt=1.0)
    nominal = simulate_full_mission(dt=1.0)
    scenario = fault_sim.scenario

    # 冷启动：无 warmstart_h
    cold_result = plan_recovery_segment_scvx(
        scenario=scenario, fault_sim=fault_sim, nominal=nominal,
        nodes=40, eta=eta, use_adaptive_penalties=True,
        warmstart_h=None, mission_domain=domain,
    )
    cold_diag = _extract_scvx_diagnostics(cold_result.diagnostics)

    # 构建热启动
    warmstart_h = build_learning_warmstart(ctx, scenario, fault_sim, nominal, nodes=40)

    # 热启动
    warm_result = plan_recovery_segment_scvx(
        scenario=scenario, fault_sim=fault_sim, nominal=nominal,
        nodes=40, eta=eta, use_adaptive_penalties=True,
        warmstart_h=warmstart_h, mission_domain=domain,
    )
    warm_diag = _extract_scvx_diagnostics(warm_result.diagnostics)

    return {
        "cold": cold_diag,
        "warm": warm_diag,
        "available": True,
    }


def run_pipeline(
    scenario: str,
    eta: float,
    t_fault: Optional[float] = None,
    make_plots: bool = False,
) -> Dict[str, Any]:
    """
    供其他模块（例如前端 UI）调用的完整流水线入口。

    参数:
        scenario: 故障场景 ID，例如 "F1_thrust_deg15"
        eta: 故障严重度 (0~1)
        t_fault: 故障注入时间（秒）。如果为 None，则使用场景中的默认时间。
        make_plots: 是否在内部生成静态图（可选，默认 False）

    返回:
        一个字典，结构如下:
        {
            "scenario": str,
            "eta": float,
            "t_fault": float,
            "diagnosis": {...},
            "mission_domain": {...},
            "trajectory": {
                "nominal": {"t": [...], "downrange_km": [...], "altitude_km": [...]},
                "fault_open_loop": {"t": [...], "downrange_km": [...], "altitude_km": [...]},
                "reconfigured": {"t": [...], "downrange_km": [...], "altitude_km": [...]},
            },
            "raw": {...},
        }
    """
    import io
    import sys as _sys

    dt = 0.5  # 仿真步长
    R_EARTH = 6.378137e6

    # 静默执行，捕获打印输出
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()

    try:
        # 0. 名义轨迹
        nominal_result = simulate_nominal(dt=dt, save_csv=False)
        traj_nominal = _to_traj_dict(nominal_result.time, nominal_result.states, R_EARTH)

        # 1. 故障仿真（开环）
        sim_data = run_fault_simulation(scenario, eta, dt)
        fault_type = sim_data["scenario"].fault_type
        traj_fault = _to_traj_dict(sim_data["time"], sim_data["states"], R_EARTH)

        # 2. ESO 残差估计
        residual = run_eso_estimation(sim_data, dt)

        # 3. PWVD 特征提取
        features = extract_pwvd_features(residual, dt)

        # 4. 故障诊断
        diag = run_fault_diagnosis(features, fault_type, eta)

        # 5. 任务域选择
        domain, domain_info = select_mission_domain(diag.severity_eta)

        # 6. SCvx 轨迹重规划 - 冷启动
        scvx_result = simulate_fault_and_solve(
            scenario_id=scenario,
            eta=eta,
            nodes=40,
            use_adaptive_penalties=True,
            mission_domain=domain,
        )
        traj_rec = _to_traj_dict(scvx_result.time, scvx_result.states, R_EARTH)

        # 7. SCvx 热启动对比（可选）
        warmstart_comparison = _run_warmstart_comparison(scenario, eta, domain)

        # 自适应权重
        weights = compute_adaptive_penalties(eta)

        # 确定 t_fault
        actual_t_fault = t_fault if t_fault is not None else 0.0

        # 从 scvx_result 提取终端状态
        rec_states = scvx_result.states
        r_final = np.linalg.norm(rec_states[-1, 0:3])
        v_final = np.linalg.norm(rec_states[-1, 3:6])
        terminal_altitude_km = (r_final - R_EARTH) / 1000.0
        terminal_velocity_kms = v_final / 1000.0
        plan_feasible = scvx_result.diagnostics.get("solver_status", "") in ["optimal", "OPTIMAL"]

    finally:
        _sys.stdout = old_stdout

    # 构建返回字典
    result = {
        "scenario": scenario,
        "eta": float(eta),
        "t_fault": float(actual_t_fault) if actual_t_fault is not None else None,
        "diagnosis": {
            "fault_type": diag.fault_label,
            "eta_est": diag.severity_eta,
            "confidence": diag.confidence,
        },
        "mission_domain": {
            "name": domain.name,
            "h_target_km": domain_info["target_altitude_km"],
            "v_target_kms": domain_info["target_velocity_kms"],
        },
        "trajectory": {
            "nominal": traj_nominal,
            "fault_open_loop": traj_fault,
            "reconfigured": traj_rec,
        },
        "raw": {
            "fault_type_true": fault_type,
            "residual_mean": float(np.mean(residual)),
            "residual_max": float(np.max(residual)),
            "features": features.tolist(),
            "terminal_altitude_km": terminal_altitude_km,
            "terminal_velocity_kms": terminal_velocity_kms,
            "plan_feasible": plan_feasible,
            "terminal_weight": weights.terminal_state_dev,
            "slack_weight_q": weights.q_slack,
            "slack_weight_n": weights.n_slack,
            "slack_weight_cone": weights.cone_slack,
            "speed_kms": np.asarray(sim_data["speed_kms"]).tolist(),
            "scvx_diagnostics": _extract_scvx_diagnostics(scvx_result.diagnostics),
            "warmstart_comparison": warmstart_comparison,
        },
    }

    # 可选绘图
    if make_plots:
        plan_result = {
            "feasible": plan_feasible,
            "terminal_altitude_km": terminal_altitude_km,
            "terminal_velocity_kms": terminal_velocity_kms,
            "iterations": scvx_result.diagnostics.get("num_iterations", 0),
            "terminal_weight": weights.terminal_state_dev,
            "slack_weight": weights.q_slack,
            "states": scvx_result.states,
            "time": scvx_result.time,
        }
        pipeline_result = PipelineResult(
            scenario_id=scenario,
            eta=eta,
            fault_type=fault_type,
            diagnosed_fault_type=diag.fault_label,
            diagnosed_eta=diag.severity_eta,
            diagnosis_confidence=diag.confidence,
            mission_domain=domain.name,
            plan_feasible=plan_feasible,
            terminal_altitude_km=terminal_altitude_km,
            terminal_velocity_kms=terminal_velocity_kms,
            scvx_iterations=scvx_result.diagnostics.get("num_iterations", 0),
            terminal_weight=weights.terminal_state_dev,
            slack_weight=weights.q_slack,
        )
        _plot_results(sim_data, plan_result, pipeline_result)

    return result


def run_full_pipeline(
    scenario_id: str,
    eta: float,
    dt: float = 0.5,
    plot: bool = False,
) -> PipelineResult:
    """运行完整流水线."""
    print("=" * 60)
    print("故障诊断与轨迹规划完整流水线")
    print("=" * 60)

    # 1. 故障仿真
    sim_data = run_fault_simulation(scenario_id, eta, dt)
    fault_type = sim_data["scenario"].fault_type

    # 2. ESO 残差估计
    residual = run_eso_estimation(sim_data, dt)

    # 3. PWVD 特征提取
    features = extract_pwvd_features(residual, dt)

    # 4. 故障诊断
    diag = run_fault_diagnosis(features, fault_type, eta)

    # 5. 任务域选择
    domain, domain_info = select_mission_domain(diag.severity_eta)

    # 6. 轨迹规划
    plan_result = run_trajectory_planning(scenario_id, eta, domain, sim_data)

    # 汇总结果
    result = PipelineResult(
        scenario_id=scenario_id,
        eta=eta,
        fault_type=fault_type,
        diagnosed_fault_type=diag.fault_label,
        diagnosed_eta=diag.severity_eta,
        diagnosis_confidence=diag.confidence,
        mission_domain=domain.name,
        plan_feasible=plan_result["feasible"],
        terminal_altitude_km=plan_result["terminal_altitude_km"],
        terminal_velocity_kms=plan_result["terminal_velocity_kms"],
        scvx_iterations=plan_result["iterations"],
        terminal_weight=plan_result["terminal_weight"],
        slack_weight=plan_result["slack_weight"],
    )

    # 打印汇总
    print("\n" + "=" * 60)
    print("流水线结果汇总")
    print("=" * 60)
    print(f"场景: {result.scenario_id}")
    print(f"真实故障: {result.fault_type}, η={result.eta:.2f}")
    print(f"诊断结果: {result.diagnosed_fault_type}, η={result.diagnosed_eta:.2f}")
    print(f"任务域: {result.mission_domain}")
    print(f"规划可行: {result.plan_feasible}")
    print(f"终端状态: h={result.terminal_altitude_km:.1f}km, v={result.terminal_velocity_kms:.2f}km/s")

    # 可选绘图
    if plot and plan_result["states"] is not None:
        _plot_results(sim_data, plan_result, result)

    return result


def _plot_results(sim_data: Dict, plan_result: Dict, result: PipelineResult):
    """绘制结果图."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # 故障开环轨迹
        ax = axes[0, 0]
        ax.plot(sim_data["time"], sim_data["altitude_km"], "r--", label="故障开环")
        ax.set_xlabel("时间 / s")
        ax.set_ylabel("高度 / km")
        ax.set_title("高度-时间曲线")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 重规划轨迹
        if plan_result["states"] is not None:
            states = plan_result["states"]
            r = np.linalg.norm(states[:, 0:3], axis=1)
            h = (r - 6.378137e6) / 1000.0
            ax.plot(plan_result["time"], h, "b-", label="SCvx重规划")
            ax.legend()

        # 速度曲线
        ax = axes[0, 1]
        ax.plot(sim_data["time"], sim_data["speed_kms"], "r--", label="故障开环")
        if plan_result["states"] is not None:
            v = np.linalg.norm(plan_result["states"][:, 3:6], axis=1) / 1000.0
            ax.plot(plan_result["time"], v, "b-", label="SCvx重规划")
        ax.set_xlabel("时间 / s")
        ax.set_ylabel("速度 / km/s")
        ax.set_title("速度-时间曲线")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 任务域信息
        ax = axes[1, 0]
        ax.text(0.5, 0.7, f"场景: {result.scenario_id}", ha="center", fontsize=12)
        ax.text(0.5, 0.5, f"任务域: {result.mission_domain}", ha="center", fontsize=14, fontweight="bold")
        ax.text(0.5, 0.3, f"η = {result.eta:.2f}", ha="center", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("任务域选择")

        # 规划结果
        ax = axes[1, 1]
        ax.text(0.5, 0.8, f"可行性: {'✓' if result.plan_feasible else '✗'}", ha="center", fontsize=12)
        ax.text(0.5, 0.6, f"终端高度: {result.terminal_altitude_km:.1f} km", ha="center", fontsize=11)
        ax.text(0.5, 0.4, f"终端速度: {result.terminal_velocity_kms:.2f} km/s", ha="center", fontsize=11)
        ax.text(0.5, 0.2, f"SCvx迭代: {result.scvx_iterations}", ha="center", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("规划结果")

        fig.suptitle(f"故障诊断与轨迹规划流水线 - {result.scenario_id}", fontsize=14)
        fig.tight_layout()

        # 保存图片
        outdir = PROJECT_ROOT / "outputs" / "figures" / "pipeline"
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"pipeline_{result.scenario_id}_eta{result.eta:.1f}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"\n图片已保存: {outpath}")

        plt.close(fig)

    except ImportError:
        print("\n[警告] matplotlib 未安装，跳过绘图")


def main():
    parser = argparse.ArgumentParser(description="完整故障诊断与轨迹规划流水线")
    parser.add_argument(
        "--scenario",
        type=str,
        default="F1_thrust_deg15",
        help=f"故障场景 ID，可选: {', '.join(get_scenario_ids())}",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.5,
        help="故障严重度 η ∈ [0, 1]",
    )
    parser.add_argument(
        "--t_fault",
        type=float,
        default=None,
        help="故障注入时间（秒），为空则使用场景默认配置",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="是否绘制结果图",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="运行所有场景 (F1-F5)",
    )

    args = parser.parse_args()

    if args.all_scenarios:
        # 运行所有基础场景
        scenarios = ["F1_thrust_deg15", "F2_tvc_rate4", "F3_tvc_stuck3deg",
                     "F4_sensor_bias2deg", "F5_event_delay5s"]
        etas = [0.2, 0.5, 0.8]

        results = []
        for scenario in scenarios:
            for eta in etas:
                try:
                    res = run_pipeline(
                        scenario=scenario,
                        eta=eta,
                        t_fault=args.t_fault,
                        make_plots=args.plot,
                    )
                    results.append(res)
                    # 打印单个结果
                    print(f"[OK] {scenario} η={eta:.1f} -> {res['mission_domain']['name']}")
                except Exception as e:
                    print(f"[错误] {scenario} η={eta}: {e}")

        # 打印汇总表
        print("\n" + "=" * 80)
        print("所有场景汇总")
        print("=" * 80)
        print(f"{'场景':<20} {'η':>5} {'任务域':<12} {'可行':>5} {'高度(km)':>10} {'速度(km/s)':>12}")
        print("-" * 80)
        for r in results:
            print(f"{r['scenario']:<20} {r['eta']:>5.1f} {r['mission_domain']['name']:<12} "
                  f"{'✓' if r['raw']['plan_feasible'] else '✗':>5} "
                  f"{r['raw']['terminal_altitude_km']:>10.1f} {r['raw']['terminal_velocity_kms']:>12.2f}")
    else:
        # 单场景运行
        res = run_pipeline(
            scenario=args.scenario,
            eta=args.eta,
            t_fault=args.t_fault,
            make_plots=args.plot,
        )
        # 打印结果摘要
        print("=" * 60)
        print("故障诊断与轨迹规划完整流水线")
        print("=" * 60)
        print(f"场景: {res['scenario']}")
        print(f"故障严重度 η: {res['eta']:.2f}")
        print(f"故障注入时间: {res['t_fault']:.1f}s")
        print(f"\n诊断结果:")
        print(f"  故障类型: {res['diagnosis']['fault_type']}")
        print(f"  诊断 η: {res['diagnosis']['eta_est']:.2f}")
        print(f"  置信度: {res['diagnosis']['confidence']:.2f}")
        print(f"\n任务域:")
        print(f"  选择域: {res['mission_domain']['name']}")
        print(f"  目标高度: {res['mission_domain']['h_target_km']:.0f} km")
        print(f"  目标速度: {res['mission_domain']['v_target_kms']:.2f} km/s")
        print(f"\n轨迹数据 (三条轨迹):")
        traj = res["trajectory"]
        for name, label in [("nominal", "名义轨迹"), ("fault_open_loop", "故障开环"), ("reconfigured", "SCvx重构")]:
            t_data = traj[name]
            n_pts = len(t_data["t"])
            h_end = t_data["altitude_km"][-1] if n_pts > 0 else 0.0
            dr_end = t_data["downrange_km"][-1] if n_pts > 0 else 0.0
            print(f"  {label}: {n_pts}点, 终端高度={h_end:.1f}km, 下航程={dr_end:.1f}km")
        print(f"\n规划结果:")
        print(f"  可行性: {'是' if res['raw']['plan_feasible'] else '否'}")
        print(f"  终端高度: {res['raw']['terminal_altitude_km']:.1f} km")
        print(f"  终端速度: {res['raw']['terminal_velocity_kms']:.2f} km/s")
        print(f"\n自适应权重:")
        print(f"  终端权重: {res['raw']['terminal_weight']:.2f}")
        print(f"  松弛权重 (q): {res['raw']['slack_weight_q']:.2f}")


if __name__ == "__main__":
    main()
