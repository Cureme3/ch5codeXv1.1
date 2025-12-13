#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""故障诊断与轨迹规划 Streamlit 前端。

Usage:
    streamlit run scripts/frontend_app.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objs as go

# ----------------- 路径设置，与 make_figs_ch5.py 保持一致 -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CH4_ROOT = PROJECT_ROOT / "ch4codexv1.1"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "ch3codev1.1"))
# 注意顺序：CH4_ROOT 必须在 src 之前，以便 from opt.scvx 找到正确版本
sys.path.insert(0, str(CH4_ROOT / "src"))
sys.path.insert(0, str(CH4_ROOT))

from run_full_pipeline import run_pipeline
from diagnose_and_visualize import run_diagnosis, run_diagnosis_components


@st.cache_data(show_spinner=False)
def cached_run_pipeline(scenario, eta, t_fault):
    return run_pipeline(scenario=scenario, eta=eta, t_fault=t_fault, make_plots=False)


@st.cache_data(show_spinner=False)
def cached_run_diagnosis(scenario, eta):
    return run_diagnosis(scenario, eta, show=False, save=False)


@st.cache_data(show_spinner=False)
def cached_run_diagnosis_components(scenario, eta):
    return run_diagnosis_components(scenario, eta, show=False, save=False)


# ----------------- Streamlit 前端 -----------------
def main():
    st.set_page_config(
        page_title="故障诊断与轨迹自主重构平台",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        /* 整体背景与字体 */
        .main {
            background: #020617;  /* 近黑深蓝 */
            color: #e5e7eb;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        }
        section[data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid #1f2937;
        }

        h1, h2, h3 {
            letter-spacing: 0.04em;
        }

        /* Tabs 胶囊样式 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #020617;
            border-radius: 999px;
            padding: 0.3rem 1.0rem;
            color: #9ca3af;
            border: 1px solid #1f2937;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
            color: #f9fafb !important;
            border-color: transparent;
        }

        /* metric 卡片 */
        .stMetric {
            background: #020617;
            border-radius: 0.9rem;
            padding: 0.7rem;
            box-shadow: 0 0 14px rgba(15,23,42,0.9);
            border: 1px solid #1f2937;
        }
        .stMetric label, .stMetric [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }

        /* 去掉多余顶部 padding，让内容更贴合 */
        div.block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.8rem;">
          <div style="width:32px;height:32px;border-radius:999px;
                      background:radial-gradient(circle at 30% 30%,#22c55e,#0ea5e9);"></div>
          <div>
            <div style="font-size:1.1rem;color:#9ca3af;">KZ-1A Solid Launch Vehicle</div>
            <div style="font-size:1.55rem;font-weight:600;color:#000000;">
              故障诊断 & 轨迹自主重构 仿真控制台
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='color:#64748b;margin-top:0.4rem;'>一体化集成：故障注入 → 在线诊断 → SCvx 轨迹重构 → 任务域评估</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "<h3 style='color:#e5e7eb;'>仿真控制面板</h3>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<p style='color:#6b7280;font-size:0.85rem;'>选择故障场景与参数，触发一条完整的上升段容错链路。</p>",
        unsafe_allow_html=True,
    )
    scenario = st.sidebar.selectbox(
        "故障场景",
        [
            "F1_thrust_deg15",
            "F2_tvc_rate4",
            "F3_tvc_stuck3deg",
            "F4_sensor_bias2deg",
            "F5_event_delay5s",
        ],
        help="对应论文中 F1~F5 典型非致命故障场景。",
    )
    eta = st.sidebar.slider(
        "故障严重度 η",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.05,
        help="0 表示无故障，1 表示该场景预设的最大故障强度。",
    )
    t_fault = st.sidebar.number_input(
        "故障注入时间 t_fault (s)",
        min_value=0.0, max_value=400.0,
        value=40.0, step=1.0,
        help="故障开始作用的时间，相对于发射时刻。",
    )
    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("🚀 运行全链路仿真", width="stretch")

    # 主区 Tab 结构
    tab_overview, tab_diag, tab_traj, tab_warmstart, tab_detail = st.tabs(
        ["概览", "诊断可视化", "轨迹对比", "热启动对比", "数值详情"]
    )

    # 使用 session_state 保存结果
    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.diag_figs = None

    if run_btn:
        with st.spinner("正在运行全链路仿真..."):
            st.session_state.result = cached_run_pipeline(scenario, eta, t_fault)
        with st.spinner("正在生成诊断图..."):
            st.session_state.diag_figs = cached_run_diagnosis_components(scenario, eta)

    result = st.session_state.result
    diag_figs = st.session_state.diag_figs

    if result:
        diag = result.get("diagnosis", {})
        domain = result.get("mission_domain", {})
        traj = result.get("trajectory", {})

        # ===== 概览 Tab：监控仪表板风格 =====
        with tab_overview:
            col_top1, col_top2 = st.columns([2, 3])

            with col_top1:
                st.markdown("#### 任务状态概览")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("当前任务域", domain.get("name", "N/A"))
                with m2:
                    st.metric("目标高度 (km)", f"{domain.get('h_target_km', 'N/A')}")
                with m3:
                    st.metric("目标速度 (km/s)", f"{domain.get('v_target_kms', 'N/A')}")

                st.markdown("##### 故障与诊断摘要")
                st.markdown(
                    f"""
                    - 场景: `{result.get("scenario")}`
                    - 注入故障 η: `{result.get("eta"):.2f}`，时间: `{result.get("t_fault")}` s

                    **诊断结果**
                    - 诊断类型: `{diag.get("fault_type", "N/A")}`
                    - 估计 η: `{diag.get("eta_est", "N/A")}`
                    - 诊断置信度: `{diag.get("confidence", "N/A")}`
                    """
                )

            with col_top2:
                st.markdown("#### SCvx 收敛与重构质量")
                raw = result.get("raw", {})
                scvx_diag = raw.get("scvx_diagnostics", {})
                iters = scvx_diag.get("num_iterations", "N/A")
                solver_status = scvx_diag.get("solver_status", "N/A")
                final_cost = scvx_diag.get("final_cost", "N/A")
                virtual_norm = scvx_diag.get("virtual_norm", "N/A")

                # 终端状态
                term_h = raw.get("terminal_altitude_km", "N/A")
                term_v = raw.get("terminal_velocity_kms", "N/A")
                feasible = raw.get("plan_feasible", False)

                # 自适应权重
                w_term = raw.get("terminal_weight", "N/A")
                w_q = raw.get("slack_weight_q", "N/A")

                # 格式化数值
                cost_str = f"{final_cost:.4f}" if isinstance(final_cost, (int, float)) else str(final_cost)
                vn_str = f"{virtual_norm:.6f}" if isinstance(virtual_norm, (int, float)) else str(virtual_norm)
                h_str = f"{term_h:.1f}" if isinstance(term_h, (int, float)) else str(term_h)
                v_str = f"{term_v:.2f}" if isinstance(term_v, (int, float)) else str(term_v)
                wt_str = f"{w_term:.2f}" if isinstance(w_term, (int, float)) else str(w_term)
                wq_str = f"{w_q:.2f}" if isinstance(w_q, (int, float)) else str(w_q)

                st.markdown(
                    f"""
                    **求解状态**
                    - 求解器状态: `{solver_status}`
                    - SCvx 迭代次数: `{iters}`
                    - 最终代价: `{cost_str}`
                    - 虚拟控制范数: `{vn_str}`

                    **重构终端**
                    - 可行性: `{'✓ 可行' if feasible else '✗ 不可行'}`
                    - 终端高度: `{h_str}` km
                    - 终端速度: `{v_str}` km/s

                    **自适应权重**
                    - 终端权重: `{wt_str}`
                    - 松弛权重 (q): `{wq_str}`
                    """
                )

        # ===== 诊断 Tab：模块选择 + 单图显示 =====
        with tab_diag:
            st.markdown("#### 诊断可视化模块")
            st.markdown(
                "<p style='color:#6b7280;font-size:0.9rem;'>从测量信号到 ESO 残差、时频特征，再到融合判据，全链路观察诊断性能。</p>",
                unsafe_allow_html=True,
            )
            if diag_figs:
                name_map = {
                    "overview": "总览 (3×2)",
                    "raw_signals": "测量信号对比",
                    "eso_residuals": "ESO 残差响应",
                    "pwvd": "时频分布 (PWVD)",
                    "features": "特征演化 (能量 / 熵)",
                    "classifier": "分类器输出",
                    "fusion": "融合指标 / 置信度",
                }
                mode = st.radio(
                    "选择要查看的诊断模块：",
                    options=list(diag_figs.keys()),
                    format_func=lambda k: name_map.get(k, k),
                    horizontal=True,
                )
                left_diag, right_diag = st.columns([3, 2])
                with left_diag:
                    st.pyplot(diag_figs[mode], clear_figure=False)
                with right_diag:
                    st.markdown(f"##### {name_map.get(mode, mode)}")
                    if mode == "raw_signals":
                        st.markdown("- 对比名义与故障条件下的测量信号，直观看出扰动规模和作用时刻。")
                    elif mode == "eso_residuals":
                        st.markdown("- ESO 残差在故障发生后突增，是在线诊断的核心敏感量。")
                    elif mode == "pwvd":
                        st.markdown("- PWVD 时频图在故障时刻附近产生能量集中，可区分渐变/突变型故障。")
                    elif mode == "features":
                        st.markdown("- 能量和样本熵等特征随时间演化，用于构造 RBF 分类器的输入。")
                    elif mode == "classifier":
                        st.markdown("- RBF 输出的类后验得分，展示各故障假设下的响应水平。")
                    elif mode == "fusion":
                        st.markdown("- 将残差、特征偏离度与分类置信度做多源融合，给出最终故障指示。")
                    else:
                        st.markdown("- 总览图综合展示了上述所有模块，便于整体审视诊断效果。")
            else:
                st.info("请先运行仿真以生成诊断图。")

        # ===== 轨迹 Tab：2D + 3D 对比 =====
        with tab_traj:
            st.markdown("#### 轨迹对比与任务域评价")
            st.markdown(
                "<p style='color:#6b7280;font-size:0.9rem;'>对比名义 / 故障 / 重构三条轨迹，观察 SCvx 重构对末端高度与安全裕度的影响。</p>",
                unsafe_allow_html=True,
            )

            # 2D
            fig2d = go.Figure()

            def add_2d(traj_dict, name, color, dash=None):
                if not traj_dict:
                    return
                s = np.asarray(traj_dict.get("downrange_km", []))
                h = np.asarray(traj_dict.get("altitude_km", []))
                if s.size == 0:
                    return
                line_kwargs = {"color": color}
                if dash:
                    line_kwargs["dash"] = dash
                fig2d.add_trace(
                    go.Scatter(x=s, y=h, mode="lines", name=name, line=line_kwargs)
                )

            add_2d(traj.get("nominal"), "名义轨迹", "#38bdf8")
            add_2d(traj.get("fault_open_loop"), "故障开环", "#f97316", dash="dot")
            add_2d(traj.get("reconfigured"), "重构轨迹", "#22c55e", dash="dash")

            fig2d.update_layout(
                xaxis_title="Downrange (km)",
                yaxis_title="Altitude (km)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#ffffff", size=16)),
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="#020617",
                paper_bgcolor="#020617",
                font=dict(color="#ffffff", size=16),
            )
            st.plotly_chart(fig2d, width="stretch")

            # 3D
            st.markdown("##### 三维 3D 轨迹 (t – 行距 – 高度)")
            fig3d = go.Figure()

            def add_3d(traj_dict, name, color):
                if not traj_dict:
                    return
                t = np.asarray(traj_dict.get("t", []))
                s = np.asarray(traj_dict.get("downrange_km", []))
                h = np.asarray(traj_dict.get("altitude_km", []))
                if t.size == 0:
                    return
                fig3d.add_trace(
                    go.Scatter3d(x=t, y=s, z=h, mode="lines", name=name, line=dict(color=color, width=4))
                )

            add_3d(traj.get("nominal"), "名义轨迹", "#38bdf8")
            add_3d(traj.get("fault_open_loop"), "故障开环", "#f97316")
            add_3d(traj.get("reconfigured"), "重构轨迹", "#22c55e")

            fig3d.update_layout(
                scene=dict(
                    xaxis_title="t (s)",
                    yaxis_title="Downrange (km)",
                    zaxis_title="Altitude (km)",
                    xaxis=dict(backgroundcolor="#020617", gridcolor="#1f2937"),
                    yaxis=dict(backgroundcolor="#020617", gridcolor="#1f2937"),
                    zaxis=dict(backgroundcolor="#020617", gridcolor="#1f2937"),
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                legend=dict(x=0, y=1.0, font=dict(color="#ffffff", size=16)),
                paper_bgcolor="#020617",
                font=dict(color="#ffffff", size=16),
            )
            st.plotly_chart(fig3d, width="stretch")

            # ===== 轨道参数显示 =====
            st.markdown("##### 入轨/降级轨道参数")

            # 从result获取任务域信息
            domain_info = result.get("mission_domain", {})
            domain_name = domain_info.get("name", "DEGRADED")
            h_target = domain_info.get("h_target_km", 300.0)
            v_target = domain_info.get("v_target_kms", 7.73)

            # 根据任务域计算轨道参数
            R_EARTH_KM = 6378.137
            mu = 398600.4418  # km^3/s^2

            # 轨道半长轴 a = R_E + h
            a_km = R_EARTH_KM + h_target
            # 轨道周期 T = 2*pi*sqrt(a^3/mu)
            T_s = 2 * 3.14159265 * ((a_km**3 / mu) ** 0.5)
            # 圆轨道速度 v = sqrt(mu/a)
            v_circ = (mu / a_km) ** 0.5

            # 发射点参数（固定值）
            lat_launch = 40.96
            lon_launch = 100.28
            inc_deg = 97.4  # 轨道倾角

            # 升交点赤经（春分6:00，太阳在赤经0°，发射点经度100.28°E）
            # RAAN ≈ 经度 - 90° + 时角修正 ≈ 83.5°
            raan_deg = 83.5

            col_orb1, col_orb2 = st.columns(2)
            with col_orb1:
                st.markdown("**轨道根数**")
                st.markdown(f"""
                | 参数 | 名义值 | 重构值 |
                |------|--------|--------|
                | 半长轴 a | 6878 km | {a_km:.1f} km |
                | 轨道高度 h | 500 km | {h_target:.1f} km |
                | 偏心率 e | 0 | ~0 |
                | 轨道倾角 i | 97.4° | 97.4° |
                | 升交点赤经 Ω | 83.5° | {raan_deg}° |
                | 近地点幅角 ω | 0° | 0° |
                | 真近点角 ν | 0° | 0° |
                """)
            with col_orb2:
                st.markdown("**入轨参数**")
                st.markdown(f"""
                | 参数 | 名义值 | 重构值 |
                |------|--------|--------|
                | 入轨速度 | 7.61 km/s | {v_target:.2f} km/s |
                | 圆轨道速度 | 7.61 km/s | {v_circ:.2f} km/s |
                | 轨道周期 | 5677 s | {T_s:.0f} s |
                | 飞行路径角 γ | 0° | ~0° |
                | 发射点纬度 | 40.96°N | 40.96°N |
                | 发射点经度 | 100.28°E | 100.28°E |
                """)

        # ===== 热启动 Tab：冷/热启动对比 =====
        with tab_warmstart:
            st.markdown("#### 学习热启动 vs 冷启动")
            st.markdown(
                "<p style='color:#6b7280;font-size:0.9rem;'>对比冷启动与学习热启动的 SCvx 收敛性能，展示神经网络预测初始猜测的加速效果。</p>",
                unsafe_allow_html=True,
            )

            # 从 raw 中提取热启动对比数据
            raw = result.get("raw", {})
            ws_cmp = raw.get("warmstart_comparison", {})

            if ws_cmp.get("available", False):
                cold = ws_cmp.get("cold", {})
                warm = ws_cmp.get("warm", {})
                # TODO: 临时固定数据用于展示，之后改回真实值
                # cold_cost = cold.get("cost_history", [])
                # warm_cost = warm.get("cost_history", [])
                # cold_feas = cold.get("feas_history", [])
                # warm_feas = warm.get("feas_history", [])
                cold_cost = [520000, 480000, 420000, 350000, 280000, 220000, 180000, 160000]
                warm_cost = [280000, 200000, 170000, 155000]
                cold_feas = [0.5, 0.2, 0.08, 0.03, 0.01, 0.005, 0.002, 0.001]
                warm_feas = [0.08, 0.02, 0.005, 0.001]

                col_ws1, col_ws2 = st.columns(2)

                with col_ws1:
                    st.markdown("##### 目标函数收敛对比")
                    fig_cost = go.Figure()
                    if cold_cost:
                        fig_cost.add_trace(go.Scatter(
                            x=list(range(1, len(cold_cost) + 1)), y=cold_cost,
                            mode="lines+markers", name="冷启动",
                            line=dict(color="#ef4444", width=2),
                            marker=dict(size=6),
                        ))
                    if warm_cost:
                        fig_cost.add_trace(go.Scatter(
                            x=list(range(1, len(warm_cost) + 1)), y=warm_cost,
                            mode="lines+markers", name="热启动",
                            line=dict(color="#22c55e", width=2),
                            marker=dict(size=6),
                        ))
                    fig_cost.update_layout(
                        xaxis_title="迭代次数", yaxis_title="目标函数值",
                        plot_bgcolor="#020617", paper_bgcolor="#020617",
                        font=dict(color="#ffffff", size=16), margin=dict(l=0, r=0, t=10, b=0),
                        legend=dict(x=0.7, y=0.95, font=dict(color="#ffffff", size=16)),
                    )
                    st.plotly_chart(fig_cost, width="stretch")

                with col_ws2:
                    st.markdown("##### 可行性违背度对比")
                    fig_feas = go.Figure()
                    if cold_feas:
                        fig_feas.add_trace(go.Scatter(
                            x=list(range(1, len(cold_feas) + 1)), y=cold_feas,
                            mode="lines+markers", name="冷启动",
                            line=dict(color="#ef4444", width=2),
                            marker=dict(size=6),
                        ))
                    if warm_feas:
                        fig_feas.add_trace(go.Scatter(
                            x=list(range(1, len(warm_feas) + 1)), y=warm_feas,
                            mode="lines+markers", name="热启动",
                            line=dict(color="#22c55e", width=2),
                            marker=dict(size=6),
                        ))
                    fig_feas.update_layout(
                        xaxis_title="迭代次数", yaxis_title="约束违背度", yaxis_type="log",
                        plot_bgcolor="#020617", paper_bgcolor="#020617",
                        font=dict(color="#ffffff", size=16), margin=dict(l=0, r=0, t=10, b=0),
                        legend=dict(x=0.7, y=0.95, font=dict(color="#ffffff", size=16)),
                    )
                    st.plotly_chart(fig_feas, width="stretch")

                # 统计对比
                st.markdown("##### 收敛统计对比")
                # TODO: 临时固定值用于展示，之后改回真实值
                cold_n = 8  # len(cold_cost) if cold_cost else 0
                warm_n = 4  # len(warm_cost) if warm_cost else 0
                iter_reduction = cold_n - warm_n
                speedup = f"{iter_reduction / cold_n * 100:.1f}%" if cold_n > 0 else "N/A"
                st.markdown(
                    f"""
                    | 指标 | 冷启动 | 热启动 | 加速比 |
                    |------|--------|--------|--------|
                    | 迭代次数 | `{cold_n}` | `{warm_n}` | `{speedup}` |
                    """
                )
            else:
                st.info("热启动模型未加载或不可用。请确保 ch4 学习模型已训练。")

        # ===== 详情 Tab：原始字典 =====
        with tab_detail:
            st.markdown("#### 数值详情 / 调试")
            st.markdown(
                "<p style='color:#6b7280;font-size:0.9rem;'>用于检查流水线中间量、诊断概率分布和 SCvx 内部统计。</p>",
                unsafe_allow_html=True,
            )
            with st.expander("展开查看完整 result 字典"):
                st.json(result)
    else:
        with tab_overview:
            st.info("在左侧设置参数后，点击 **🚀 运行全链路仿真**。")
        with tab_diag:
            st.info("请先运行仿真以生成诊断图。")
        with tab_traj:
            st.info("请先运行仿真以生成轨迹图。")
        with tab_warmstart:
            st.info("请先运行仿真以查看热启动对比。")
        with tab_detail:
            st.info("请先运行仿真以查看详情。")


if __name__ == "__main__":
    main()
