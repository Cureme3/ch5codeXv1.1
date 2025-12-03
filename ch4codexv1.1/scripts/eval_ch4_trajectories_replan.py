#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四章：名义/故障开环/重规划轨迹数据生成脚本（增强版）。

对每种典型故障场景（F1~F5），在不同故障程度（eta）下生成：
1. 名义轨迹（无故障，名义制导）
2. 故障开环轨迹（注入故障但不启动重规划）
3. 故障 + 重规划轨迹（完整轨迹 = 故障前段 + 重规划段）

关键改进：
- 重规划轨迹覆盖完整飞行过程（故障前 + 重规划后）
- 支持多个 eta 值（轻度/中度/重度故障）
- 标识任务域（retain/degraded/safe_area）
- 保存故障信息供绘图使用

输出文件：
- outputs/data/ch4_trajectories_replan/Fk_eta{eta}_*.npz

用法:
    python -m scripts.eval_ch4_trajectories_replan
"""

from __future__ import annotations

import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from src.sim.run_fault import (
    run_fault_scenario,
    plan_recovery_segment_scvx,
    sample_state_at_time,
)
from src.sim.run_nominal import simulate_full_mission, R_EARTH
from src.sim.scenarios import get_scenario
from src.sim.mission_domains import MissionDomain, choose_initial_domain


# 故障场景 ID 映射
FAULT_ID_MAP: Dict[str, str] = {
    "F1": "F1_thrust_deg15",
    "F2": "F2_tvc_rate4",
    "F3": "F3_tvc_stuck3deg",
    "F4": "F4_sensor_bias2deg",
    "F5": "F5_event_delay5s",
}

# 故障中文名称
FAULT_NAMES: Dict[str, str] = {
    "F1": "推力降级",
    "F2": "TVC速率限制",
    "F3": "TVC卡滞",
    "F4": "传感器偏置",
    "F5": "事件延迟",
}

# 测试的 eta 值（故障严重度）
ETA_VALUES: List[float] = [0.2, 0.5, 0.8]  # 轻度、中度、重度

# eta 对应的标签
ETA_LABELS: Dict[float, str] = {
    0.2: "轻度 (η=0.2)",
    0.5: "中度 (η=0.5)",
    0.8: "重度 (η=0.8)",
}


@dataclass
class TrajectoryData:
    """轨迹数据结构。"""
    time: np.ndarray
    downrange: np.ndarray  # km
    altitude: np.ndarray   # km
    label: str
    eta: float = 0.0
    mission_domain: str = ""
    fault_type: str = ""
    t_fault: float = 0.0   # 故障发生时间
    t_confirm: float = 0.0  # 故障确认时间


@dataclass
class DebugRecord:
    """调试记录，用于分析SCvx是否真正改变了轨迹。"""
    fault_id: str
    eta: float
    domain_initial: str
    domain_final: str
    solver_success: bool
    outer_iterations: int
    max_virtual_control_norm: float
    trust_region_radius_final: float
    terminal_pos_error_norm: float
    terminal_vel_error_norm: float
    max_downrange_diff: float
    max_altitude_diff: float
    final_feas_violation: float
    total_cost: float


def compute_downrange_from_pos(pos: np.ndarray, ref_pos: np.ndarray = None) -> np.ndarray:
    """从 ECI 位置计算地面行距（相对于参考点）。"""
    if len(pos) == 0:
        return np.array([])
    if ref_pos is None:
        ref_pos = pos[0]
    pos_relative = pos - ref_pos
    downrange = np.sqrt(pos_relative[:, 0]**2 + pos_relative[:, 1]**2) / 1000.0
    return downrange


def compute_altitude_from_pos(pos: np.ndarray) -> np.ndarray:
    """从 ECI 位置计算海拔高度 (km)。"""
    if len(pos) == 0:
        return np.array([])
    r_norm = np.linalg.norm(pos, axis=1)
    altitude = (r_norm - R_EARTH) / 1000.0
    return altitude


def generate_nominal_trajectory(dt: float = 1.0) -> TrajectoryData:
    """生成名义轨迹。"""
    nominal = simulate_full_mission(dt=dt)
    time = np.asarray(nominal.time, dtype=float)
    states = np.asarray(nominal.states, dtype=float)
    pos = states[:, 0:3]

    return TrajectoryData(
        time=time,
        downrange=compute_downrange_from_pos(pos),
        altitude=compute_altitude_from_pos(pos),
        label="Nominal",
    )


def generate_fault_openloop_trajectory(
    scenario_id: str,
    eta: float,
    dt: float = 1.0,
) -> TrajectoryData:
    """生成故障开环轨迹。"""
    fault_sim = run_fault_scenario(scenario_id, dt=dt, eta=eta)
    scenario = fault_sim.scenario

    time = fault_sim.time
    states = fault_sim.states
    pos = states[:, 0:3]

    domain = choose_initial_domain(eta)

    return TrajectoryData(
        time=time,
        downrange=compute_downrange_from_pos(pos),
        altitude=fault_sim.altitude_km,
        label=f"Fault open-loop (η={eta})",
        eta=eta,
        mission_domain="open_loop",
        fault_type=scenario.fault_type,
        t_fault=scenario.t_fault_s,
        t_confirm=scenario.t_confirm_s,
    )


def generate_complete_replan_trajectory(
    scenario_id: str,
    eta: float,
    nominal_data: TrajectoryData,
    dt: float = 1.0,
    nodes: int = 40,
) -> Tuple[TrajectoryData, DebugRecord]:
    """生成完整的故障+重规划轨迹，并返回调试记录。

    轨迹组成：
    1. 故障前段（从起飞到故障确认时刻，与名义轨迹相同或使用故障仿真）
    2. 重规划段（从故障确认时刻到任务结束）

    Returns
    -------
    Tuple[TrajectoryData, DebugRecord]
        轨迹数据和调试记录
    """
    # 1. 运行故障仿真获取故障前段
    fault_sim = run_fault_scenario(scenario_id, dt=dt, eta=eta)
    scenario = fault_sim.scenario
    t_confirm = scenario.t_confirm_s

    # 2. 获取故障确认前的轨迹段（从故障仿真中截取）
    pre_fault_mask = fault_sim.time <= t_confirm
    pre_time = fault_sim.time[pre_fault_mask]
    pre_states = fault_sim.states[pre_fault_mask]
    pre_pos = pre_states[:, 0:3]

    # 3. 获取名义任务用于 SCvx
    nominal = simulate_full_mission(dt=dt)

    # 4. 确定任务域
    domain = choose_initial_domain(eta)
    domain_initial = domain.name

    # 5. 运行 SCvx 重规划（启用域升级 FSM）
    recovery = plan_recovery_segment_scvx(
        scenario=scenario,
        fault_sim=fault_sim,
        nominal=nominal,
        nodes=nodes,
        fault_eta=eta,
        use_adaptive_penalties=True,
        solver_profile="fast",
        mission_domain=domain,
        enable_domain_escalation=True,
    )
    # 获取 SCvx 最终确定的任务域（可能经过升级）
    final_domain = getattr(recovery, "mission_domain", domain)

    # 6. 提取重规划段轨迹
    replan_time = recovery.time  # SCvx 节点时间
    replan_states = recovery.states
    replan_pos = replan_states[:, 0:3]

    # 7. 拼接完整轨迹
    # 确保时间连续：去掉重规划段与前段重叠的部分
    if len(pre_time) > 0 and len(replan_time) > 0:
        # 找到重规划段中时间 > 故障确认时刻的部分
        replan_mask = replan_time > pre_time[-1]
        if np.any(replan_mask):
            replan_time_trim = replan_time[replan_mask]
            replan_pos_trim = replan_pos[replan_mask]
        else:
            # 如果重规划时间都小于等于前段最后时刻，取整个重规划段
            replan_time_trim = replan_time[1:]  # 跳过第一个点避免重复
            replan_pos_trim = replan_pos[1:]

        # 拼接
        full_time = np.concatenate([pre_time, replan_time_trim])
        full_pos = np.concatenate([pre_pos, replan_pos_trim], axis=0)
    else:
        full_time = replan_time
        full_pos = replan_pos

    # 8. 计算行距和高度（使用统一的参考点）
    ref_pos = nominal_data.time[0] if len(nominal_data.time) > 0 else None
    # 使用完整轨迹的起点作为参考
    if len(full_pos) > 0:
        downrange = compute_downrange_from_pos(full_pos, full_pos[0])
        altitude = compute_altitude_from_pos(full_pos)
    else:
        downrange = np.array([])
        altitude = np.array([])

    # 9. 任务域标签（使用大写，保存 SCvx 最终确定的域）
    try:
        domain_name = final_domain.name  # "RETAIN" / "DEGRADED" / "SAFE_AREA"
    except Exception:
        domain_name = str(final_domain).upper()

    # 10. 提取调试信息
    diag = recovery.diagnostics
    solver_status = diag.get("solver_status", "unknown")
    solver_success = solver_status in ["optimal", "OPTIMAL", "optimal_inaccurate"]
    outer_iterations = diag.get("num_iterations", 0)
    final_feas_violation = diag.get("final_feas_violation", 0.0)

    # 从 SCvx logs 获取更多信息
    scvx_logs = diag.get("scvx_logs", [])
    trust_region_radius_final = 0.0
    max_virtual_control_norm = 0.0
    total_cost = 0.0
    if scvx_logs and len(scvx_logs) > 0:
        last_log = scvx_logs[-1]
        trust_region_radius_final = getattr(last_log, "trust_radius", 0.0)
        total_cost = getattr(last_log, "total_cost", 0.0)
        # 从 slack 变量估算虚拟控制范数
        max_slack_q = getattr(last_log, "max_slack_q", 0.0)
        max_slack_n = getattr(last_log, "max_slack_n", 0.0)
        max_slack_cone = getattr(last_log, "max_slack_cone", 0.0)
        max_virtual_control_norm = max(max_slack_q, max_slack_n, max_slack_cone)

    # 计算终端误差
    terminal_pos_error_norm = 0.0
    terminal_vel_error_norm = 0.0
    if len(replan_states) > 0 and len(nominal.states) > 0:
        replan_final_pos = replan_states[-1, 0:3]
        replan_final_vel = replan_states[-1, 3:6] if replan_states.shape[1] >= 6 else np.zeros(3)
        nom_states = np.asarray(nominal.states)
        nom_final_pos = nom_states[-1, 0:3]
        nom_final_vel = nom_states[-1, 3:6] if nom_states.shape[1] >= 6 else np.zeros(3)
        terminal_pos_error_norm = float(np.linalg.norm(replan_final_pos - nom_final_pos)) / 1000.0  # km
        terminal_vel_error_norm = float(np.linalg.norm(replan_final_vel - nom_final_vel)) / 1000.0  # km/s

    # 计算与名义轨迹的差异
    max_downrange_diff = 0.0
    max_altitude_diff = 0.0
    if len(downrange) > 0 and len(nominal_data.downrange) > 0:
        # 插值到相同时间点比较
        min_len = min(len(downrange), len(nominal_data.downrange))
        dr_diff = np.abs(downrange[:min_len] - nominal_data.downrange[:min_len])
        alt_diff = np.abs(altitude[:min_len] - nominal_data.altitude[:min_len])
        max_downrange_diff = float(np.max(dr_diff))
        max_altitude_diff = float(np.max(alt_diff))

    # 获取 fault_id（从 scenario_id 提取）
    fault_id = scenario_id.split("_")[0] if "_" in scenario_id else scenario_id

    debug_record = DebugRecord(
        fault_id=fault_id,
        eta=eta,
        domain_initial=domain_initial,
        domain_final=domain_name,
        solver_success=solver_success,
        outer_iterations=outer_iterations,
        max_virtual_control_norm=max_virtual_control_norm,
        trust_region_radius_final=trust_region_radius_final,
        terminal_pos_error_norm=terminal_pos_error_norm,
        terminal_vel_error_norm=terminal_vel_error_norm,
        max_downrange_diff=max_downrange_diff,
        max_altitude_diff=max_altitude_diff,
        final_feas_violation=final_feas_violation,
        total_cost=total_cost,
    )

    traj_data = TrajectoryData(
        time=full_time,
        downrange=downrange,
        altitude=altitude,
        label=f"Replan ({domain_name}, η={eta})",
        eta=eta,
        mission_domain=domain_name,
        fault_type=scenario.fault_type,
        t_fault=scenario.t_fault_s,
        t_confirm=t_confirm,
    )

    return traj_data, debug_record


def save_trajectory_npz(out_path: Path, data: TrajectoryData) -> None:
    """保存轨迹数据到 npz 文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        t=data.time,
        downrange=data.downrange,
        altitude=data.altitude,
        label=data.label,
        eta=data.eta,
        mission_domain=data.mission_domain,
        fault_type=data.fault_type,
        t_fault=data.t_fault,
        t_confirm=data.t_confirm,
    )
    print(f"    Saved: {out_path.name} ({len(data.time)} points)")


def save_debug_records_to_csv(records: List[DebugRecord], out_path: Path) -> None:
    """将调试记录保存到 CSV 文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fault_id", "eta", "domain_initial", "domain_final",
        "solver_success", "outer_iterations", "max_virtual_control_norm",
        "trust_region_radius_final", "terminal_pos_error_norm",
        "terminal_vel_error_norm", "max_downrange_diff", "max_altitude_diff",
        "final_feas_violation", "total_cost"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "fault_id": rec.fault_id,
                "eta": rec.eta,
                "domain_initial": rec.domain_initial,
                "domain_final": rec.domain_final,
                "solver_success": rec.solver_success,
                "outer_iterations": rec.outer_iterations,
                "max_virtual_control_norm": rec.max_virtual_control_norm,
                "trust_region_radius_final": rec.trust_region_radius_final,
                "terminal_pos_error_norm": rec.terminal_pos_error_norm,
                "terminal_vel_error_norm": rec.terminal_vel_error_norm,
                "max_downrange_diff": rec.max_downrange_diff,
                "max_altitude_diff": rec.max_altitude_diff,
                "final_feas_violation": rec.final_feas_violation,
                "total_cost": rec.total_cost,
            })
    print(f"\n[DEBUG] 保存调试记录到: {out_path}")


def main() -> None:
    """主函数。"""
    print("=" * 80)
    print("第四章：名义/故障开环/重规划轨迹数据生成（增强版）")
    print("=" * 80)
    print(f"故障场景: {list(FAULT_ID_MAP.keys())}")
    print(f"eta 值: {ETA_VALUES}")
    print()

    out_dir = Path("outputs/data/ch4_trajectories_replan")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集调试记录
    debug_records: List[DebugRecord] = []

    # 生成名义轨迹（全局共用）
    print("[0/5] 生成名义轨迹...")
    nominal_data = generate_nominal_trajectory(dt=1.0)
    save_trajectory_npz(out_dir / "nominal.npz", nominal_data)
    print(f"    名义轨迹: {len(nominal_data.time)} 点")
    print()

    for fault_key, scenario_id in FAULT_ID_MAP.items():
        print(f"[{fault_key}] 处理故障场景: {FAULT_NAMES[fault_key]} ({scenario_id})")
        print(f"    " + "-" * 60)

        for eta in ETA_VALUES:
            domain = choose_initial_domain(eta)
            eta_str = f"eta{eta:.1f}".replace(".", "")
            print(f"    [η={eta}] 任务域: {domain.name}")

            # 生成故障开环轨迹
            try:
                openloop_data = generate_fault_openloop_trajectory(
                    scenario_id, eta=eta, dt=1.0
                )
                save_trajectory_npz(
                    out_dir / f"{fault_key}_{eta_str}_openloop.npz",
                    openloop_data,
                )
            except Exception as e:
                print(f"      [WARN] 故障开环失败: {e}")

            # 生成故障+重规划轨迹
            try:
                replan_data, debug_record = generate_complete_replan_trajectory(
                    scenario_id,
                    eta=eta,
                    nominal_data=nominal_data,
                    dt=1.0,
                    nodes=40,
                )
                save_trajectory_npz(
                    out_dir / f"{fault_key}_{eta_str}_replan.npz",
                    replan_data,
                )
                debug_records.append(debug_record)
                print(f"      [DEBUG] domain: {debug_record.domain_initial} → {debug_record.domain_final}, "
                      f"solver: {debug_record.solver_success}, iters: {debug_record.outer_iterations}, "
                      f"max_dr_diff: {debug_record.max_downrange_diff:.2f}km, "
                      f"max_alt_diff: {debug_record.max_altitude_diff:.2f}km")
            except Exception as e:
                print(f"      [WARN] 重规划失败: {e}")
                import traceback
                traceback.print_exc()

        print()

    # 保存调试记录到 CSV
    debug_csv_path = Path("outputs/data/ch4_scvx_replan_debug.csv")
    save_debug_records_to_csv(debug_records, debug_csv_path)

    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"输出目录: {out_dir}")
    print(f"调试CSV: {debug_csv_path}")
    print()
    print("下一步：运行 python -m scripts.make_figs_ch4_trajectories_replan 生成图表")


if __name__ == "__main__":
    main()
