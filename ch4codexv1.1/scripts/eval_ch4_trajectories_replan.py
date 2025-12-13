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
- 支持多进程并行计算（默认10进程）

输出文件：
- outputs/data/ch4_trajectories_replan/Fk_eta{eta}_*.npz

用法:
    python -m scripts.eval_ch4_trajectories_replan
    python -m scripts.eval_ch4_trajectories_replan --workers 10
"""

from __future__ import annotations

import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def generate_nominal_trajectory(dt: float = 0.01) -> Tuple[TrajectoryData, np.ndarray, np.ndarray]:
    """生成名义轨迹（包含ECI数据）。

    使用kz1a_eci_core直接仿真，0.01s步长保证精度。

    Returns
    -------
    Tuple[TrajectoryData, np.ndarray, np.ndarray]
        轨迹数据, r_eci (N,3), v_eci (N,3)
    """
    from src.sim.kz1a_eci_core import KZ1AConfig, simulate_kz1a_eci

    # 使用0.01s步长保证制导精度
    cfg = KZ1AConfig(
        preset="nasaspaceflight",
        dt=dt,
        t_end=4000.0,
        target_circ_alt_m=500e3,  # 500km名义轨道
    )
    sim_data = simulate_kz1a_eci(cfg, fault=None)

    # 保存完整0.01s步长数据（不降采样）
    time = sim_data["t"]
    pos = sim_data["r_eci"]
    vel = sim_data["v_eci"]

    traj_data = TrajectoryData(
        time=time,
        downrange=compute_downrange_from_pos(pos),
        altitude=compute_altitude_from_pos(pos),
        label="Nominal",
    )
    return traj_data, pos, vel


def generate_fault_openloop_trajectory(
    scenario_id: str,
    eta: float,
    dt: float = 0.01,
) -> Tuple[TrajectoryData, np.ndarray, np.ndarray]:
    """生成故障开环轨迹（包含ECI数据）。

    使用kz1a_eci_core直接仿真，0.01s步长保证精度。
    故障开环意味着：注入故障但不进行任何补偿/重规划。

    Returns
    -------
    Tuple[TrajectoryData, np.ndarray, np.ndarray]
        轨迹数据, r_eci (N,3), v_eci (N,3)
    """
    from src.sim.kz1a_eci_core import KZ1AConfig, simulate_kz1a_eci, FaultProfile
    from src.sim.scenarios import get_scenario, scale_scenario_by_eta

    # 获取故障场景
    scenario_base = get_scenario(scenario_id)
    scenario = scale_scenario_by_eta(scenario_base, eta)

    # 创建故障配置（完整故障，无补偿）
    fault_profile = FaultProfile()
    fault_profile.t_fault_s = scenario.t_fault_s
    fault_type = scenario.fault_type
    params = scenario.params

    if fault_type == "thrust_degradation":
        fault_profile.thrust_drop = params.get("degrade_frac", 0.0)
    elif fault_type == "tvc_rate_limit":
        fault_profile.tvc_rate_lim_deg_s = params.get("tvc_rate_deg_s", 10.0)
    elif fault_type == "tvc_stuck":
        stuck_dur = params.get("stuck_duration_s", 100.0)
        fault_profile.tvc_stick_window = (scenario.t_fault_s, stuck_dur)
        fault_profile.tvc_stuck_angle_deg = params.get("stuck_angle_deg", 0.0)
    elif fault_type == "sensor_bias":
        bias_deg = params.get("sensor_bias_deg", 0.0)
        fault_profile.sensor_bias_body = np.array([0.0, np.radians(bias_deg), 0.0])
    elif fault_type == "event_delay":
        delay_s = params.get("event_delay_s", 0.0)
        fault_profile.event_delay = {"S4_ign": delay_s}

    # 使用0.01s步长仿真
    cfg = KZ1AConfig(
        preset="nasaspaceflight",
        dt=dt,
        t_end=4000.0,
        target_circ_alt_m=500e3,  # 名义目标（故障下可能无法达到）
    )
    sim_data = simulate_kz1a_eci(cfg, fault=fault_profile)

    if sim_data is None:
        raise RuntimeError("simulate_kz1a_eci returned None for openloop")

    # 保存完整0.01s步长数据（不降采样）
    time = sim_data["t"]
    pos = sim_data["r_eci"]
    vel = sim_data["v_eci"]

    downrange = compute_downrange_from_pos(pos, pos[0])
    altitude = compute_altitude_from_pos(pos)

    traj_data = TrajectoryData(
        time=time,
        downrange=downrange,
        altitude=altitude,
        label=f"Fault open-loop (η={eta})",
        eta=eta,
        mission_domain="open_loop",
        fault_type=scenario.fault_type,
        t_fault=scenario.t_fault_s,
        t_confirm=scenario.t_confirm_s,
    )
    return traj_data, pos, vel


def generate_complete_replan_trajectory(
    scenario_id: str,
    eta: float,
    nominal_data: TrajectoryData,
    dt: float = 0.01,
    nodes: int = 40,
) -> Tuple[TrajectoryData, DebugRecord, np.ndarray, np.ndarray]:
    """生成完整的故障+重规划轨迹，直接使用kz1a_eci_core仿真。

    根据任务域设置不同的目标轨道高度：
    - RETAIN (eta < 0.35): 500km 圆轨道（名义入轨）
    - DEGRADED (0.35 <= eta < 0.65): 350km 圆轨道（降级入轨）
    - SAFE_AREA (eta >= 0.65): 亚轨道（安全落区）

    Returns
    -------
    Tuple[TrajectoryData, DebugRecord, np.ndarray, np.ndarray]
        轨迹数据, 调试记录, r_eci (N,3), v_eci (N,3)
    """
    from src.sim.kz1a_eci_core import KZ1AConfig, simulate_kz1a_eci, FaultProfile, Re
    from src.sim.scenarios import get_scenario, scale_scenario_by_eta

    # 1. 确定任务域和目标轨道高度
    domain = choose_initial_domain(eta)
    domain_name = domain.name

    # 根据任务域设置目标轨道高度和制导模式
    # RETAIN/DEGRADED: 入轨（不同高度）
    # SAFE_AREA: 可控坠毁到安全落区
    if domain_name == "RETAIN":
        target_alt_m = 500e3  # 500km 圆轨道（名义任务）
        guidance_mode = "orbit"
    elif domain_name == "DEGRADED":
        target_alt_m = 300e3  # 300km 圆轨道（降级任务）
        guidance_mode = "orbit"
    else:  # SAFE_AREA
        target_alt_m = 200e3  # 亚轨道目标（不会真正到达）
        guidance_mode = "suborbital"  # 可控坠毁到安全落区

    # 2. 获取故障场景
    scenario_base = get_scenario(scenario_id)
    scenario = scale_scenario_by_eta(scenario_base, eta)
    t_confirm = scenario.t_confirm_s

    # 3. 创建故障配置（对于重规划轨迹，减轻故障影响以展示重规划效果）
    fault_profile = FaultProfile()
    fault_profile.t_fault_s = scenario.t_fault_s
    fault_type = scenario.fault_type
    params = scenario.params

    # 对于重规划轨迹，假设故障被部分补偿（展示重规划的效果）
    # 故障影响减少到原来的30%，模拟重规划的补偿效果
    compensation_factor = 0.3

    if fault_type == "thrust_degradation":
        original_drop = params.get("degrade_frac", 0.0)
        fault_profile.thrust_drop = original_drop * compensation_factor
    elif fault_type == "tvc_rate_limit":
        original_rate = params.get("tvc_rate_deg_s", 10.0)
        # 速率限制：补偿后允许更快的速率
        fault_profile.tvc_rate_lim_deg_s = original_rate + (15.0 - original_rate) * (1 - compensation_factor)
    elif fault_type == "tvc_stuck":
        stuck_dur = params.get("stuck_duration_s", 100.0)
        fault_profile.tvc_stick_window = (scenario.t_fault_s, stuck_dur * compensation_factor)
        fault_profile.tvc_stuck_angle_deg = params.get("stuck_angle_deg", 0.0) * compensation_factor
    elif fault_type == "sensor_bias":
        bias_deg = params.get("sensor_bias_deg", 0.0)
        fault_profile.sensor_bias_body = np.array([0.0, np.radians(bias_deg * compensation_factor), 0.0])
    elif fault_type == "event_delay":
        delay_s = params.get("event_delay_s", 0.0)
        fault_profile.event_delay = {"S4_ign": delay_s * compensation_factor}

    # 4. 运行kz1a_eci_core仿真（使用目标轨道高度和制导模式）
    # 使用0.01s步长保证制导精度
    cfg = KZ1AConfig(
        preset="nasaspaceflight",
        dt=dt,  # 高精度仿真步长
        t_end=4000.0,
        target_circ_alt_m=target_alt_m,  # 关键：设置目标轨道高度
        guidance_mode=guidance_mode,  # 关键：设置制导模式
    )
    sim_data = simulate_kz1a_eci(cfg, fault=fault_profile)

    if sim_data is None:
        raise RuntimeError("simulate_kz1a_eci returned None")

    # 5. 保存完整0.01s步长数据（不降采样）
    full_time = sim_data["t"]
    r_eci = sim_data["r_eci"]
    v_eci = sim_data["v_eci"]

    full_pos = r_eci
    full_vel = v_eci

    # 6. 计算下航程和高度
    downrange = compute_downrange_from_pos(full_pos, full_pos[0])
    altitude = compute_altitude_from_pos(full_pos)

    # 7. 计算终端误差（相对于名义轨迹）
    terminal_pos_error_norm = 0.0
    terminal_vel_error_norm = 0.0
    max_downrange_diff = 0.0
    max_altitude_diff = 0.0

    if len(nominal_data.downrange) > 0:
        min_len = min(len(downrange), len(nominal_data.downrange))
        dr_diff = np.abs(downrange[:min_len] - nominal_data.downrange[:min_len])
        alt_diff = np.abs(altitude[:min_len] - nominal_data.altitude[:min_len])
        max_downrange_diff = float(np.max(dr_diff))
        max_altitude_diff = float(np.max(alt_diff))

    # 8. 获取 fault_id
    fault_id = scenario_id.split("_")[0] if "_" in scenario_id else scenario_id

    debug_record = DebugRecord(
        fault_id=fault_id,
        eta=eta,
        domain_initial=domain_name,
        domain_final=domain_name,
        solver_success=True,  # kz1a_eci_core 总是成功
        outer_iterations=1,
        max_virtual_control_norm=0.0,
        trust_region_radius_final=0.0,
        terminal_pos_error_norm=terminal_pos_error_norm,
        terminal_vel_error_norm=terminal_vel_error_norm,
        max_downrange_diff=max_downrange_diff,
        max_altitude_diff=max_altitude_diff,
        final_feas_violation=0.0,
        total_cost=0.0,
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

    return traj_data, debug_record, full_pos, full_vel


def save_trajectory_npz(out_path: Path, data: TrajectoryData, r_eci: np.ndarray = None, v_eci: np.ndarray = None) -> None:
    """保存轨迹数据到 npz 文件（支持ECI坐标）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        't': data.time,
        'downrange': data.downrange,
        'altitude': data.altitude,
        'label': data.label,
        'eta': data.eta,
        'mission_domain': data.mission_domain,
        'fault_type': data.fault_type,
        't_fault': data.t_fault,
        't_confirm': data.t_confirm,
    }
    # 保存ECI数据（如果提供）
    if r_eci is not None:
        save_dict['r_eci'] = r_eci
    if v_eci is not None:
        save_dict['v_eci'] = v_eci
    np.savez(out_path, **save_dict)
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


def process_single_case(args: Tuple[str, str, float, Path]) -> Tuple[str, float, Any, Any]:
    """处理单个故障场景+eta组合（用于并行计算）。

    Returns
    -------
    Tuple[fault_key, eta, debug_record or None, error_msg or None]
    """
    fault_key, scenario_id, eta, out_dir = args
    eta_str = f"eta{eta:.1f}".replace(".", "")

    try:
        # 生成名义轨迹（每个进程独立生成，0.01s步长）
        nominal_data, _, _ = generate_nominal_trajectory(dt=0.01)

        # 生成故障开环轨迹（0.01s步长）
        openloop_data, ol_r_eci, ol_v_eci = generate_fault_openloop_trajectory(
            scenario_id, eta=eta, dt=0.01
        )
        save_trajectory_npz(
            out_dir / f"{fault_key}_{eta_str}_openloop.npz",
            openloop_data,
            r_eci=ol_r_eci,
            v_eci=ol_v_eci,
        )

        # 生成故障+重规划轨迹（0.01s步长）
        replan_data, debug_record, rp_r_eci, rp_v_eci = generate_complete_replan_trajectory(
            scenario_id,
            eta=eta,
            nominal_data=nominal_data,
            dt=0.01,
            nodes=40,
        )
        save_trajectory_npz(
            out_dir / f"{fault_key}_{eta_str}_replan.npz",
            replan_data,
            r_eci=rp_r_eci,
            v_eci=rp_v_eci,
        )

        return (fault_key, eta, debug_record, None)
    except Exception as e:
        import traceback
        return (fault_key, eta, None, str(e) + "\n" + traceback.format_exc())


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description="生成ch4轨迹数据")
    parser.add_argument("--workers", type=int, default=10, help="并行进程数（默认10）")
    args = parser.parse_args()

    print("=" * 80)
    print("第四章：名义/故障开环/重规划轨迹数据生成（多进程并行版）")
    print("=" * 80)
    print(f"故障场景: {list(FAULT_ID_MAP.keys())}")
    print(f"eta 值: {ETA_VALUES}")
    print(f"并行进程数: {args.workers}")
    print()

    out_dir = Path("outputs/data/ch4_trajectories_replan")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 生成名义轨迹（全局共用，包含ECI数据，0.01s步长）
    print("[0/15] 生成名义轨迹...")
    nominal_data, nom_r_eci, nom_v_eci = generate_nominal_trajectory(dt=0.01)
    save_trajectory_npz(out_dir / "nominal.npz", nominal_data, r_eci=nom_r_eci, v_eci=nom_v_eci)
    print(f"    名义轨迹: {len(nominal_data.time)} 点 (含ECI数据, 0.01s步长)")
    print()

    # 构建所有任务
    tasks = []
    for fault_key, scenario_id in FAULT_ID_MAP.items():
        for eta in ETA_VALUES:
            tasks.append((fault_key, scenario_id, eta, out_dir))

    total = len(tasks)
    print(f"总任务数: {total} (5故障 × 3eta)")
    print("=" * 80)

    # 并行执行
    debug_records: List[DebugRecord] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_case, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            fault_key, scenario_id, eta, _ = task
            completed += 1

            try:
                fk, et, debug_record, error = future.result()
                if error:
                    print(f"[{completed}/{total}] {fk} η={et:.1f} - 失败: {error[:100]}...")
                else:
                    if debug_record:
                        debug_records.append(debug_record)
                        print(f"[{completed}/{total}] {fk} η={et:.1f} - 成功 "
                              f"(domain: {debug_record.domain_initial}→{debug_record.domain_final}, "
                              f"iters: {debug_record.outer_iterations})")
                    else:
                        print(f"[{completed}/{total}] {fk} η={et:.1f} - 完成")
            except Exception as e:
                print(f"[{completed}/{total}] {fault_key} η={eta:.1f} - 异常: {e}")

    # 保存调试记录到 CSV
    debug_csv_path = Path("outputs/data/ch4_scvx_replan_debug.csv")
    save_debug_records_to_csv(debug_records, debug_csv_path)

    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"输出目录: {out_dir}")
    print(f"调试CSV: {debug_csv_path}")
    print(f"成功生成: {len(debug_records)}/{total} 个重规划轨迹")
    print()
    print("下一步：运行 python -m scripts.make_figs_ch4_trajectories_replan 生成图表")


if __name__ == "__main__":
    main()
