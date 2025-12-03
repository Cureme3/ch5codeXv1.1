"""逐次凸优化主循环骨架 (修正版)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# 引入底层 SOCP 构建模块
from .socp_problem import Discretization, SOCPBuilder, TrustRegion
# 引入新增的约束配置和线性化模块
from .constraints import OptimizationConfig
from .linearization import linearize_dynamics_3dof

@dataclass
class SCvxLog:
    costs: List[float]
    virtual_norms: List[float]
    trust_radii: List[Tuple[float, float]]

class SCvxPlanner:
    """高层驱动逻辑：线性化 → 构建 SOCP → 求解 → 更新参考。"""

    def __init__(self, dynamics_model, config: OptimizationConfig):
        """
        Args:
            dynamics_model: 必须是 Dynamics3DOF 实例，用于线性化计算。
            config: OptimizationConfig 实例，包含节点数、时间步长、物理约束等。
        """
        self.dynamics_model = dynamics_model
        self.config = config
        
        # 初始化离散化参数 (基于 3-DoF 模型: 状态7维，控制3维)
        self.disc = Discretization(
            dt=config.dt,
            nodes=config.nodes,
            state_dim=7, 
            control_dim=3,
        )
        
        # 初始化信赖域 (参数可根据需要调整，或从 config 读取)
        self.trust = TrustRegion(
            state_radius=2000.0,   # 初始状态信赖半径
            control_radius=100000.0 # 初始控制信赖半径
        )

    def solve(self, x_init: np.ndarray, u_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, SCvxLog]:
        """
        执行 SCvx 迭代求解。
        
        Args:
            x_init: 初始状态轨迹猜测 (Shape: [N, 7])
            u_init: 初始控制序列猜测 (Shape: [N, 3])
            
        Returns:
            x_opt: 优化后的状态轨迹
            u_opt: 优化后的控制序列
            log: 迭代日志
        """
        # 1. 初始化参考轨迹
        x_ref = x_init.copy()
        u_ref = u_init.copy()
        
        # 准备终端目标 (位置+速度，质量不约束设为0) [cite: 30, 42, 61]
        # 注意：具体的终端约束逻辑在 socp_problem.py 的 build 方法中处理
        x_target = np.concatenate([self.config.r_target, self.config.v_target, [0.0]])

        log = SCvxLog(costs=[], virtual_norms=[], trust_radii=[])
        
        # 2. 主迭代循环
        max_iters = 15  # 最大迭代次数 [cite: 43]
        
        print(f"[SCvx] Starting optimization with {self.config.nodes} nodes over {self.config.dt * (self.config.nodes-1):.1f}s.")

        for iteration in range(max_iters):
            # --- A. 动力学线性化 ---
            A_seq = []
            B_seq = []
            # 生成时间网格用于线性化
            t_grid = np.linspace(0, self.config.dt * (self.config.nodes - 1), self.config.nodes)
            
            for k in range(self.config.nodes):
                t_k = t_grid[k]
                x_k = x_ref[k]
                u_k = u_ref[k]
                # 调用 3-DoF 线性化函数计算雅可比矩阵 [cite: 63]
                Ak, Bk = linearize_dynamics_3dof(self.dynamics_model, t_k, x_k, u_k)
                A_seq.append(Ak)
                B_seq.append(Bk)
            
            # --- B. 构建 SOCP 子问题 ---
            # 定义代价权重 (可从 config 扩展)
            weights = {
                "terminal": 100.0,      # 终端误差惩罚
                "control": 0.01,        # 控制量正则化
                "virtual": 10000.0      # 虚拟控制惩罚 (保证动力学可行性) [cite: 64]
            }
            
            builder = SOCPBuilder(self.disc, self.trust, weights)
            
            # 传入 A_seq, B_seq 和 config (含故障参数) 构建问题
            # 注意：这里调用的是修改后的 build 方法
            problem, vars_ = builder.build(x_ref, u_ref, x_target, (A_seq, B_seq), self.config)
            
            # --- C. 调用求解器 ---
            try:
                # ECOS 是处理 SOCP 的常用求解器
                problem.solve(solver="ECOS", verbose=False)
            except Exception as e:
                print(f"[SCvx] Solver raised exception: {e}")
                break

            if problem.status not in ("optimal", "optimal_inaccurate"):
                print(f"[SCvx] Warning: Solver status '{problem.status}' at iter {iteration}")
                # 如果求解失败，可以选择缩小信赖域重试或退出
            
            # --- D. 更新与收敛检查 ---
            # 提取新轨迹
            if vars_["x"].value is None:
                print("[SCvx] Failed to retrieve values. Stopping.")
                break

            x_new = np.array(vars_["x"].value)
            u_new = np.array(vars_["u"].value)
            nu_val = np.array(vars_["nu"].value) # 虚拟控制量
            
            # 计算虚拟控制范数 (衡量动力学违背程度)
            virtual_norm = float(np.sum(np.linalg.norm(nu_val, axis=1)))
            cost = problem.value
            
            # 记录日志
            log.costs.append(cost)
            log.virtual_norms.append(virtual_norm)
            log.trust_radii.append((self.trust.state_radius, self.trust.control_radius))
            
            print(f"  Iter {iteration+1}: Cost={cost:.4f}, Virtual_Norm={virtual_norm:.6f}, Trust={self.trust.state_radius:.1f}")
            
            # 更新参考轨迹
            x_ref = x_new
            u_ref = u_new
            
            # 收敛判据: 虚拟控制足够小且代价稳定
            if virtual_norm < 1e-3:
                print("[SCvx] Converged: Virtual control norm below tolerance.")
                break
                
            # 简单的信赖域更新逻辑 (可根据实际情况优化)
            # 这里保持恒定或简单衰减
            # self.trust.state_radius *= 0.95
            
        return x_ref, u_ref, log
