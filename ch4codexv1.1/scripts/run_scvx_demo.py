#!/usr/bin/env python
"""离散网格检查 / SOCP / SCvx demo 工具。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))

from sim.run_nominal import simulate_full_mission  # noqa: E402
from sim.dynamics_wrapper import Dynamics3DOF # noqa: E402

from opt.discretization import DiscreteTrajectory, GridConfig, TrajectoryDiscretizer  # noqa: E402
from opt.socp_problem import (  # noqa: E402
    ConstraintBounds,
    PenaltyWeights,
    SOCPProblemBuilder,
    TrustRegionConfig,
)
from opt.scvx import SCvxPlanner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCvx 网格检查 / SOCP / SCvx demo")
    parser.add_argument("--nodes", type=int, default=80, help="离散节点数量（含起点）")
    parser.add_argument("--grid-only", action="store_true", help="仅构造网格并输出摘要")
    parser.add_argument("--socp-check", action="store_true", help="线性化一次 + 解一次 SOCP")
    parser.add_argument("--scvx", action="store_true", help="调用 SCvxPlanner.run 完成一次优化")
    parser.add_argument("--dt", type=float, default=1.0, help="名义仿真输出步长 s")
    parser.add_argument(
        "--save-log",
        type=str,
        default=None,
        help="保存 SCvx 迭代日志到 CSV 文件（如：outputs/ch4/tables/scvx_convergence_log.csv）",
    )
    return parser.parse_args()


def load_kz1a_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "kz1a_params.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def build_discretizer(cfg: dict, nodes: int, dynamics: Dynamics3DOF) -> TrajectoryDiscretizer:
    grid_cfg = GridConfig(
        t0=float(cfg["timeline"]["liftoff_s"]),
        tf=float(cfg["timeline"]["t_4_cutoff_s"]),
        num_nodes=nodes,
    )
    return TrajectoryDiscretizer(grid_cfg, dynamics=dynamics)


def summarize_grid(bundle) -> str:
    grid = bundle.grid
    stage_idx = bundle.stage_index
    dt = grid.dt
    lines = [
        f"节点数 N = {len(grid.nodes)}, 时间范围 {grid.nodes[0]:.1f}~{grid.nodes[-1]:.1f} s",
        f"Δt_min = {dt.min():.3f} s, Δt_max = {dt.max():.3f} s",
    ]
    for stage in np.unique(stage_idx):
        mask = stage_idx == stage
        if not np.any(mask):
            continue
        times = grid.nodes[mask]
        lines.append(f"Stage {stage}: 节点 {mask.sum()} 个, t≈{times[0]:.1f}~{times[-1]:.1f} s")
    return "\n".join(lines)


def save_traj_csv(bundle, traj: DiscreteTrajectory, outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    times = bundle.grid.nodes
    stages = bundle.stage_index
    states = traj.states
    controls = traj.controls
    state_cols = [f"x{i}" for i in range(states.shape[1])]
    control_cols = [f"u{i}" for i in range(controls.shape[1])]
    with outfile.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "stage"] + state_cols + control_cols)
        for idx, t in enumerate(times):
            writer.writerow(
                [f"{t:.3f}", int(stages[idx])]
                + [f"{val:.6e}" for val in states[idx]]
                + [f"{val:.6e}" for val in controls[idx]],
            )


def run_grid_only(discretizer: TrajectoryDiscretizer, nominal, outdir: Path) -> None:
    discretizer.build_grid()
    bundle = discretizer.project_nominal(nominal)
    summary = summarize_grid(bundle)
    print("=== Grid summary ===")
    print(summary)
    (outdir / "grid_summary.txt").write_text(summary, encoding="utf-8")
    traj = DiscreteTrajectory(grid=bundle.grid, states=bundle.states, controls=bundle.controls)
    save_traj_csv(bundle, traj, outdir / "nominal_grid.csv")
    print(f"离散轨迹已写入 {outdir / 'nominal_grid.csv'}")


def constraint_bounds_from_cfg(cfg: dict) -> ConstraintBounds:
    constraints = cfg.get("constraints", {})
    thrust_bounds: Dict[int, tuple[float, float]] = {}
    for stage in cfg.get("stages", []):
        thrust_bounds[int(stage["index"])] = (0.0, float(stage["thrust_kN"]) * 1e3)
    return ConstraintBounds(
        max_dynamic_pressure=float(constraints.get("max_dynamic_pressure_kpa", 55.0)),
        max_normal_load=float(constraints.get("max_normal_load_g", 3.5)),
        thrust_cone_deg=float(constraints.get("nominal_thrust_cone_deg", 8.0)),
        thrust_bounds=thrust_bounds,
    )


def default_trust_region() -> TrustRegionConfig:
    return TrustRegionConfig(radius_state=100.0, radius_control=10.0, min_radius=10.0, max_radius=500.0)


def run_socp_check(discretizer: TrajectoryDiscretizer, nominal, cfg: dict, outdir: Path) -> None:
    discretizer.build_grid()
    bundle = discretizer.project_nominal(nominal)
    state_dim = bundle.states.shape[1]
    control_dim = bundle.controls.shape[1]
    dynamics = discretizer.linearize_dynamics(bundle, state_dim=state_dim, control_dim=control_dim)
    bounds = constraint_bounds_from_cfg(cfg)
    penalties = PenaltyWeights(
        state_dev=1.0,
        control_dev=1.0,
        q_slack=10.0,
        n_slack=10.0,
        cone_slack=10.0,
        terminal_state_dev=50.0,
    )
    trust_cfg = default_trust_region()
    builder = SOCPProblemBuilder(bounds=bounds, weights=penalties, trust_region=trust_cfg)
    problem, variables = builder.build_problem(bundle, dynamics)
    problem.solve(solver="ECOS", verbose=False, max_iters=1000)
    traj, diagnostics = builder.extract_trajectory(variables, bundle, problem)
    print("=== SOCP check demo ===")
    for key in ("solver_status", "objective_value", "max_state_dev", "max_control_dev"):
        print(f"{key:>16s}: {diagnostics.get(key)}")
    for key in ("max_q_slack", "max_n_slack", "max_cone_slack"):
        print(f"{key:>16s}: {diagnostics.get(key)}")
    print(f"{'nodes':>16s}: {len(bundle.grid.nodes)}")
    save_traj_csv(bundle, traj, outdir / "socp_traj_simple.csv")
    print(f"优化轨迹写入 {outdir / 'socp_traj_simple.csv'}")


def run_scvx_planner(cfg: dict, nominal, nodes: int, outdir: Path, dynamics: Dynamics3DOF, save_log: str | None = None) -> None:
    bounds = constraint_bounds_from_cfg(cfg)
    penalties = PenaltyWeights(
        state_dev=1.0,
        control_dev=1.0,
        q_slack=10.0,
        n_slack=10.0,
        cone_slack=10.0,
        terminal_state_dev=50.0,
    )
    trust_cfg = default_trust_region()
    grid_cfg = GridConfig(
        t0=float(cfg["timeline"]["liftoff_s"]),
        tf=float(cfg["timeline"]["t_4_cutoff_s"]),
        num_nodes=nodes,
    )
    planner = SCvxPlanner(
        grid_cfg=grid_cfg,
        bounds=bounds,
        weights=penalties,
        trust_region=trust_cfg,
        solver_opts={"max_iters": 1000},
        dynamics=dynamics,
    )
    result = planner.run(nominal)
    log = result.logs[0]
    print("=== SCvx planner demo ===")
    print(
        f"iter {log.iter_idx}: status={log.solver_status}, cost={log.cost_candidate:.3e}, "
        f"feas_violation={log.feasibility_violation:.3e}, trust_radius={log.trust_radius:.2f}"
    )
    bundle = planner.initialize_from_nominal(nominal)
    save_traj_csv(bundle, result.trajectory, outdir / "scvx_traj.csv")
    print(f"SCvx 轨迹写入 {outdir / 'scvx_traj.csv'}")

    # Save iteration log to CSV if requested
    if save_log:
        log_path = Path(save_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # CSV header
            writer.writerow([
                "iter_idx",
                "total_cost",
                "cost_state",
                "cost_control",
                "cost_slack",
                "cost_terminal",
                "feas_violation",
                "max_slack_q",
                "max_slack_n",
                "max_slack_cone",
                "term_error_norm",
                "trust_radius",
                "solver_status",
                "rho",
                "cost_nominal",
                "cost_candidate",
            ])
            # Write each log entry
            for entry in result.logs:
                writer.writerow([
                    entry.iter_idx,
                    f"{entry.total_cost:.6e}",
                    f"{entry.cost_state:.6e}",
                    f"{entry.cost_control:.6e}",
                    f"{entry.cost_slack:.6e}",
                    f"{entry.cost_terminal:.6e}",
                    f"{entry.feasibility_violation:.6e}",
                    f"{entry.max_slack_q:.6e}",
                    f"{entry.max_slack_n:.6e}",
                    f"{entry.max_slack_cone:.6e}",
                    f"{entry.term_error_norm:.6e}",
                    f"{entry.trust_radius:.6e}",
                    entry.solver_status,
                    f"{entry.rho:.6e}",
                    f"{entry.cost_nominal:.6e}",
                    f"{entry.cost_candidate:.6e}",
                ])
        print(f"SCvx 收敛日志写入 {log_path}")


def main() -> None:
    args = parse_args()
    modes = [args.grid_only, args.socp_check, args.scvx]
    if sum(bool(m) for m in modes) > 1:
        raise SystemExit("--grid-only / --socp-check / --scvx 互斥，请只选择一个模式")
    cfg = load_kz1a_config()
    outdir = PROJECT_ROOT / "outputs" / "scvx"
    outdir.mkdir(parents=True, exist_ok=True)

    # Instantiate Dynamics3DOF
    dynamics = Dynamics3DOF(dt=0.5) # dt can be adjusted if needed

    nominal = simulate_full_mission(dt=args.dt)
    discretizer = build_discretizer(cfg, args.nodes, dynamics)

    if args.socp_check:
        run_socp_check(discretizer, nominal, cfg, outdir)
    elif args.scvx:
        # Set default save_log path if --save-log not specified but in scvx mode
        save_log_path = args.save_log
        if save_log_path is None:
            save_log_path = str(PROJECT_ROOT / "outputs" / "data" / "scvx_convergence_log.csv")
        run_scvx_planner(cfg, nominal, args.nodes, outdir, dynamics, save_log=save_log_path)
    else:
        run_grid_only(discretizer, nominal, outdir)


if __name__ == "__main__":
    main()
