# 第四章 图表最终状态报告

生成时间: 2025-11-30 10:26:07

## 1. 统计摘要

- **总条目数**: 33
- **已匹配**: 22 (66.7%)
- **自动生成缺失**: 0
- **手工绘制缺失**: 11

## 2. 已生成项目

| 编号 | 标题 | 输出文件 | 生成脚本 | 匹配状态 |
|------|------|----------|----------|----------|
| 图4-7 | SCP 迭代收敛示意图（代价/可行性/信... | `figures\ch4_scvx_convergence\fig4_07_scvx_convergence_cost.pdf` | `make_figs_ch4_scvx_convergence.py` | 前缀匹配 |
| 表4-2 | 离散优化问题变量与约束量纲统计表... | `tables\table4_02_scvx_dimension_stats.md` | `make_tables_ch4_core.py` | 完全匹配 |
| 图4-10 | 学习热启动模型的训练收敛曲线... | `figures\ch4_learning\fig4_10_learning_curves.png` | `make_figs_ch4_learning_curves.py` | 完全匹配 |
| 图4-11 | 单样本高度轨迹预测效果图... | `figures\ch4_learning\fig4_11_single_sample_altitude.png` | `make_figs_ch4_learning_altitude.py` | 完全匹配 |
| 图4-12 | 多故障工况下的高度轨迹预测效果... | `figures\ch4_learning\fig4_12_fault_cases_altitude.png` | `make_figs_ch4_learning_altitude.py` | 完全匹配 |
| 图4-13 | 各节点平均预测误差分布... | `figures\ch4_learning\fig4_13_mean_node_error.png` | `make_figs_ch4_learning_altitude.py` | 完全匹配 |
| 图4-14 | 高度预测 RMSE 分布直方图... | `figures\ch4_learning\fig4_14_rmse_histogram.png` | `make_figs_ch4_learning_altitude.py` | 完全匹配 |
| 图4-15 | 自适应罚权重随故障强度 eta 的变化... | `figures\ch4_adaptive_weights\fig4_15_adaptive_weights_vs_eta.png` | `make_figs_ch4_adaptive_weights.py` | 完全匹配 |
| 图4-16 | 冷/热启动外层迭代次数对比... | `figures\ch4_warmstart\fig4_16_warmstart_iterations.png` | `make_figs_ch4_warmstart_performance.py` | 完全匹配 |
| 图4-17 | 冷/热启动平均求解时间对比... | `figures\ch4_warmstart\fig4_17_warmstart_time.png` | `make_figs_ch4_warmstart_performance.py` | 完全匹配 |
| 图4-18 | 不同故障下终端高度 RMSE 分布... | `figures\ch4_warmstart\fig4_18_terminal_rmse_distribution.png` | `make_figs_ch4_warmstart_performance.py` | 完全匹配 |
| 表4-3 | 学习热启动网络超参数说明表... | `tables\table4_03_network_hyperparams.md` | `make_tables_ch4_core.py` | 完全匹配 |
| 图4-22 | retain 域下五种典型故障的关键飞行... | `figures\ch4_mission_domains\fig4_22_retain_domain_responses.png` | `make_figs_ch4_mission_domains.py` | 完全匹配 |
| 图4-23 | F1 severe 三种任务域下行距对比... | `figures\ch4_mission_domains\fig4_23_f1_severe_downrange.png` | `make_figs_ch4_mission_domains.py` | 完全匹配 |
| 图4-24 | 五种严重故障终端行距汇总对比... | `figures\ch4_mission_domains\fig4_24_severe_downrange_summary.png` | `make_figs_ch4_mission_domains.py` | 完全匹配 |
| 表4-4 | 任务域与对应优化问题形式对照表... | `tables\table4_04_mission_domain_mapping.md` | `make_tables_ch4_core.py` | 完全匹配 |
| 图4-25 | F1推力降级故障下的轨迹重规划对比... | `figures\ch4_trajectories_replan\fig4_25_F1_trajectory_replan.png` | `make_figs_ch4_trajectories_replan.py` | 完全匹配 |
| 图4-26 | F2 TVC速率限制故障下的轨迹重规划对... | `figures\ch4_trajectories_replan\fig4_26_F2_trajectory_replan.png` | `make_figs_ch4_trajectories_replan.py` | 完全匹配 |
| 图4-27 | F3 TVC卡滞故障下的轨迹重规划对比... | `figures\ch4_trajectories_replan\fig4_27_F3_trajectory_replan.png` | `make_figs_ch4_trajectories_replan.py` | 完全匹配 |
| 图4-28 | F4传感器偏置故障下的轨迹重规划对比... | `figures\ch4_trajectories_replan\fig4_28_F4_trajectory_replan.png` | `make_figs_ch4_trajectories_replan.py` | 完全匹配 |
| 图4-29 | F5事件延迟故障下的轨迹重规划对比... | `figures\ch4_trajectories_replan\fig4_29_F5_trajectory_replan.png` | `make_figs_ch4_trajectories_replan.py` | 完全匹配 |
| 图4-30 | 五种故障场景轨迹重规划汇总对比... | `figures\ch4_trajectories_replan\fig4_30_all_faults_summary.png` | `make_figs_ch4_trajectories_replan.py` | 完全匹配 |

## 3. 缺失项目 (自动生成类型)

*所有自动生成项目均已覆盖*


## 4. 缺失项目 (手工绘制类型)

以下项目需要手工绘制（PPT/Visio/专业工具），无法自动生成：

| 编号 | 标题 | 说明 | 建议 |
|------|------|------|------|
| 图4-1 | 故障后轨迹在线重构的时间轴示意图... | 展示故障发生→异常检测→故障确认→轨迹重构的四个阶段时间轴... | 使用 PPT/Visio 手工绘制 |
| 图4-2 | 故障影响与可行集变化的定性描述... | 对比式展示故障对推力、TVC约束等可行集的影响... | 使用 PPT/Visio 手工绘制 |
| 图4-3 | 故障前后推力锥可行域对比图... | 几何图示展示推力锥在故障前后的变化... | 使用 PPT/Visio 手工绘制 |
| 图4-4 | 事件时序偏差的影响示意图... | 说明级间分离等事件时序偏差对飞行的影响... | 使用 PPT/Visio 手工绘制 |
| 图4-5 | SCP/SCvx 算法流程框图... | 展示逐次凸规划的迭代框架结构... | 使用 PPT/Visio 手工绘制 |
| 图4-6 | 信赖域与凸优化稳定性的比喻示意... | 直观展示信赖域约束对凸优化稳定性的作用... | 使用 PPT/Visio 手工绘制 |
| 图4-8 | 学习热启动的数据流示意图... | 展示从故障状态到热启动初值的数据流... | 使用 PPT/Visio 手工绘制 |
| 图4-9 | 神经网络结构示意图... | 展示学习热启动模型的网络结构... | 使用 PPT/Visio 手工绘制 |
| 图4-19 | 任务域划分示意图... | retain/degraded/safe-area 三种任务域的概念图... | 使用 PPT/Visio 手工绘制 |
| 图4-20 | 任务域切换的有限状态机... | FSM 状态转换图... | 使用 PPT/Visio 手工绘制 |
| 图4-21 | 安全区域几何示意图... | safe-area 目标区域的几何定义... | 使用 PPT/Visio 手工绘制 |

## 5. 生成脚本清单

以下脚本负责自动生成图表：

| 脚本名 | 用途 |
|--------|------|
| `make_figs_ch4_adaptive_weights.py` | 自适应权重图 (图4-15) |
| `make_figs_ch4_learning_altitude.py` | 高度预测效果图 (图4-11~14) |
| `make_figs_ch4_learning_curves.py` | 学习曲线图 (图4-10) |
| `make_figs_ch4_mission_domains.py` | 任务域响应图 (图4-22~24) |
| `make_figs_ch4_scvx_convergence.py` | SCvx 收敛分析图 (图4-7) |
| `make_figs_ch4_trajectories_replan.py` | 轨迹重规划对比图 (图4-25~30) |
| `make_figs_ch4_warmstart_performance.py` | 热启动性能图 (图4-16~18) |

## 6. 使用建议

### 快速生成所有可自动生成的图表

```bash
# 完整模式 (推荐)
python -m scripts.generate_ch4_outputs --mode full

# 快速模式 (跳过耗时的 SCvx 步骤)
python -m scripts.generate_ch4_outputs --quick
```

### 单独运行特定脚本

```bash
# 生成核心表格
python -m scripts.make_tables_ch4_core

# 生成热启动性能图
python -m scripts.make_figs_ch4_warmstart_performance

# 生成任务域响应图
python -m scripts.make_figs_ch4_mission_domains
```

### 查看图表索引

```bash
cat outputs/INDEX.md
```
