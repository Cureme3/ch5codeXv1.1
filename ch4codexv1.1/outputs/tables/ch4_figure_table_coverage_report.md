# 第四章图表覆盖检查报告

本报告对比了论文规范表与实际 outputs 目录中的文件。

---

## 统计摘要

- 规范表总条目: 33
- 已找到匹配文件: 28 (84%)
- 缺失（自动生成类）: 0
- 缺失（手工绘制类）: 5

---

## 详细覆盖列表

| 论文编号 | 类型 | 图题/表题 | 生成方式 | 匹配状态 | 实际文件 | 备注 |
|:---------|:-----|:----------|:---------|:---------|:---------|:-----|
| 图4-1 | 图 | 故障后轨迹在线重构的时间轴示意图 | 手工绘制 | ✅ 模糊匹配 | fault_scenarios_timeline.png | 已有手工图 |
| 图4-2 | 图 | 故障影响与可行集变化的定性描述 | 手工绘制 | ❌ 未找到 | （无） | 需手工绘制并导入 |
| 图4-3 | 图 | 故障前后推力锥可行域对比图 | 半自动 | ✅ 模糊匹配 | F1_thrust_deg15_n_comparison.png, F1_thrust_deg15_q_comparison.png, ...+3 |  |
| 图4-4 | 图 | 事件时序偏差的影响示意图 | 手工绘制 | ❌ 未找到 | （无） | 需手工绘制并导入 |
| 图4-5 | 图 | SCP/SCvx 算法流程框图 | 手工绘制 | ✅ 模糊匹配 | fig4_07_scvx_convergence_cost.pdf, fig4_07_scvx_convergence_cost.png, ...+6 | 已有手工图 |
| 图4-6 | 图 | 信赖域与凸优化稳定性的比喻示意 | 手工绘制 | ✅ 模糊匹配 | fig4_07_scvx_convergence_trustregion.pdf, fig4_07_scvx_convergence_trustregion.png | 已有手工图 |
| 图4-7 | 图 | SCP 迭代收敛示意图（代价/可行性/信赖域） | 自动生成 | ✅ 前缀匹配 | fig4_07_scvx_convergence_cost.pdf, fig4_07_scvx_convergence_cost.png, ...+6 |  |
| 表4-2 | 表 | 离散优化问题变量与约束量纲统计表 | 硬编码 | ✅ 完全匹配 | table4_02_scvx_dimension_stats.md, table_scvx_stats.md |  |
| 图4-8 | 图 | 学习热启动的数据流示意图 | 手工绘制 | ✅ 模糊匹配 | fig4_16_warmstart_iterations.pdf, fig4_16_warmstart_iterations.png, ...+2 | 已有手工图 |
| 图4-9 | 图 | 神经网络结构示意图 | 手工绘制 | ❌ 未找到 | （无） | 需手工绘制并导入 |
| 图4-10 | 图 | 学习热启动模型的训练收敛曲线 | 自动生成 | ✅ 完全匹配 | fig4_10_learning_curves.pdf, fig4_10_learning_curves.png |  |
| 图4-11 | 图 | 单样本高度轨迹预测效果图 | 自动生成 | ✅ 完全匹配 | fig4_11_single_sample_altitude.pdf, fig4_11_single_sample_altitude.png |  |
| 图4-12 | 图 | 多故障工况下的高度轨迹预测效果 | 自动生成 | ✅ 完全匹配 | fig4_12_fault_cases_altitude.pdf, fig4_12_fault_cases_altitude.png |  |
| 图4-13 | 图 | 各节点平均预测误差分布 | 自动生成 | ✅ 完全匹配 | fig4_13_mean_node_error.pdf, fig4_13_mean_node_error.png |  |
| 图4-14 | 图 | 高度预测 RMSE 分布直方图 | 自动生成 | ✅ 完全匹配 | fig4_14_rmse_histogram.pdf, fig4_14_rmse_histogram.png, ...+2 |  |
| 图4-15 | 图 | 自适应罚权重随故障强度 eta 的变化 | 自动生成 | ✅ 完全匹配 | adaptive_weights_vs_eta.pdf, adaptive_weights_vs_eta.png, ...+3 |  |
| 图4-16 | 图 | 冷/热启动外层迭代次数对比 | 自动生成 | ✅ 完全匹配 | fig4_16_warmstart_iterations.pdf, fig4_16_warmstart_iterations.png, ...+2 |  |
| 图4-17 | 图 | 冷/热启动平均求解时间对比 | 自动生成 | ✅ 完全匹配 | fig4_16_warmstart_iterations.pdf, fig4_16_warmstart_iterations.png, ...+3 |  |
| 图4-18 | 图 | 不同故障下终端高度 RMSE 分布 | 自动生成 | ✅ 完全匹配 | fig4_18_terminal_rmse_distribution.pdf, fig4_18_terminal_rmse_distribution.png |  |
| 表4-3 | 表 | 学习热启动网络超参数说明表 | 硬编码 | ✅ 完全匹配 | table4_03_network_hyperparams.md |  |
| 图4-19 | 图 | 任务域划分示意图 | 手工绘制 | ❌ 未找到 | （无） | 需手工绘制并导入 |
| 图4-20 | 图 | 任务域切换的有限状态机 | 手工绘制 | ✅ 模糊匹配 | fig4_22_retain_domain_responses.pdf, fig4_22_retain_domain_responses.png, ...+2 | 已有手工图 |
| 图4-21 | 图 | 安全区域几何示意图 | 手工绘制 | ❌ 未找到 | （无） | 需手工绘制并导入 |
| 图4-22 | 图 | retain 域下五种典型故障的关键飞行量响应 | 自动生成 | ✅ 完全匹配 | fig4_22_retain_domain_responses.pdf, fig4_22_retain_domain_responses.png |  |
| 图4-23 | 图 | F1 severe 三种任务域下行距对比 | 自动生成 | ✅ 完全匹配 | fig4_23_f1_severe_downrange.pdf, fig4_23_f1_severe_downrange.png, ...+2 |  |
| 图4-24 | 图 | 五种严重故障终端行距汇总对比 | 自动生成 | ✅ 完全匹配 | fig4_23_f1_severe_downrange.pdf, fig4_23_f1_severe_downrange.png, ...+2 |  |
| 表4-4 | 表 | 任务域与对应优化问题形式对照表 | 硬编码 | ✅ 完全匹配 | mission_domain_config.md, table4_04_mission_domain_mapping.md, ...+2 |  |
| 图4-25 | 图 | F1推力降级故障下的轨迹重规划对比 | 自动生成 | ✅ 完全匹配 | fig4_25_F1_trajectory_replan.pdf, fig4_25_F1_trajectory_replan.png, ...+8 |  |
| 图4-26 | 图 | F2 TVC速率限制故障下的轨迹重规划对比 | 自动生成 | ✅ 完全匹配 | fig4_25_F1_trajectory_replan.pdf, fig4_25_F1_trajectory_replan.png, ...+8 |  |
| 图4-27 | 图 | F3 TVC卡滞故障下的轨迹重规划对比 | 自动生成 | ✅ 完全匹配 | fig4_25_F1_trajectory_replan.pdf, fig4_25_F1_trajectory_replan.png, ...+8 |  |
| 图4-28 | 图 | F4传感器偏置故障下的轨迹重规划对比 | 自动生成 | ✅ 完全匹配 | fig4_25_F1_trajectory_replan.pdf, fig4_25_F1_trajectory_replan.png, ...+8 |  |
| 图4-29 | 图 | F5事件延迟故障下的轨迹重规划对比 | 自动生成 | ✅ 完全匹配 | fig4_25_F1_trajectory_replan.pdf, fig4_25_F1_trajectory_replan.png, ...+8 |  |
| 图4-30 | 图 | 五种故障场景轨迹重规划汇总对比 | 自动生成 | ✅ 完全匹配 | fig4_30_all_faults_summary.pdf, fig4_30_all_faults_summary.png, ...+2 |  |

---

## 分类详情

### 缺失的手工绘制项（需要 PPT/Visio 等工具手工制作）

- **图4-2**: 故障影响与可行集变化的定性描述
  - 预期文件: `fig4_02_fault_effect_qualitative.*`
  - 建议导出到: `outputs/figures/ch4_schematics/`
  - 备注: 定性示意图，可用 matplotlib 或手工绘制

- **图4-4**: 事件时序偏差的影响示意图
  - 预期文件: `fig4_04_event_timing_effect.*`
  - 建议导出到: `outputs/figures/ch4_schematics/`
  - 备注: 概念示意图

- **图4-9**: 神经网络结构示意图
  - 预期文件: `fig4_09_network_architecture.*`
  - 建议导出到: `outputs/figures/ch4_schematics/`
  - 备注: 网络结构图，可用 draw.io 或 TikZ 绘制

- **图4-19**: 任务域划分示意图
  - 预期文件: `fig4_19_mission_domain_concept.*`
  - 建议导出到: `outputs/figures/ch4_schematics/`
  - 备注: 概念示意图

- **图4-21**: 安全区域几何示意图
  - 预期文件: `fig4_21_safe_area_geometry.*`
  - 建议导出到: `outputs/figures/ch4_schematics/`
  - 备注: 几何示意图

### 命名不规范项（建议重命名或调整脚本输出）

- **图4-1**: 故障后轨迹在线重构的时间轴示意图
  - 预期文件名: `fig4_01_timeline_schematic.*`
  - 当前文件: fault_scenarios_timeline.png

- **图4-3**: 故障前后推力锥可行域对比图
  - 预期文件名: `fig4_03_thrust_cone_comparison.*`
  - 当前文件: F1_thrust_deg15_n_comparison.png, F1_thrust_deg15_q_comparison.png, F1_thrust_deg15_altitude_comparison.png

- **图4-5**: SCP/SCvx 算法流程框图
  - 预期文件名: `fig4_05_scvx_flowchart.*`
  - 当前文件: fig4_07_scvx_convergence_cost.pdf, fig4_07_scvx_convergence_cost.png, fig4_07_scvx_convergence_decomp.pdf

- **图4-6**: 信赖域与凸优化稳定性的比喻示意
  - 预期文件名: `fig4_06_trust_region_intuition.*`
  - 当前文件: fig4_07_scvx_convergence_trustregion.pdf, fig4_07_scvx_convergence_trustregion.png

- **图4-8**: 学习热启动的数据流示意图
  - 预期文件名: `fig4_08_warmstart_dataflow.*`
  - 当前文件: fig4_16_warmstart_iterations.pdf, fig4_16_warmstart_iterations.png, fig4_17_warmstart_time.pdf

- **图4-20**: 任务域切换的有限状态机
  - 预期文件名: `fig4_20_domain_fsm.*`
  - 当前文件: fig4_22_retain_domain_responses.pdf, fig4_22_retain_domain_responses.png, domain_configuration_comparison.pdf

---

## 修复建议

1. **缺失的自动生成项**: 运行对应的生成脚本，或创建新脚本
2. **缺失的手工绘制项**: 使用 PPT/Visio/draw.io 绘制并导出到指定目录
3. **命名不规范项**: 修改脚本的输出文件名，使之符合 `fig4_XX_*` 或 `table4_XX_*` 格式
4. **运行一键生成**: `python -m scripts.generate_ch4_outputs --mode full`
