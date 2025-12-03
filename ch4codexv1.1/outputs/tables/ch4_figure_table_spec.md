# 第四章图表规范表（真相表）

本文档定义了论文第四章《基于学习增强的凸优化的轨迹在线重构方法》中所有图表的规范信息。

---

## 统计信息

- 总图数: 30
- 总表数: 3
- 自动生成: 19
- 手工绘制: 10
- 硬编码表格: 3

---

## 4.1 问题描述与建模

### 图4-1: 故障后轨迹在线重构的时间轴示意图

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_01_timeline_schematic.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: 展示故障发生→异常检测→故障确认→轨迹重构的四个阶段时间轴
- **备注**: 概念示意图，建议用 PPT/Visio 手工绘制

### 图4-2: 故障影响与可行集变化的定性描述

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_02_fault_effect_qualitative.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: 对比式展示故障对推力、TVC约束等可行集的影响
- **备注**: 定性示意图，可用 matplotlib 或手工绘制

### 图4-3: 故障前后推力锥可行域对比图

- **类型**: 图
- **生成方式**: 半自动
- **预期文件名**: `fig4_03_thrust_cone_comparison.*`
- **生成脚本**: `make_figs_ch4_schematics.py`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: 几何图示展示推力锥在故障前后的变化
- **备注**: 可用 matplotlib 3D 绘制推力锥

### 图4-4: 事件时序偏差的影响示意图

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_04_event_timing_effect.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: 说明级间分离等事件时序偏差对飞行的影响
- **备注**: 概念示意图

## 4.2 凸优化轨迹重构方法

### 图4-5: SCP/SCvx 算法流程框图

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_05_scvx_flowchart.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: 展示逐次凸规划的迭代框架结构
- **备注**: 算法流程图，建议用 Visio/draw.io 绘制

### 图4-6: 信赖域与凸优化稳定性的比喻示意

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_06_trust_region_intuition.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: 直观展示信赖域约束对凸优化稳定性的作用
- **备注**: 概念示意图

### 图4-7: SCP 迭代收敛示意图（代价/可行性/信赖域）

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_07_scvx_convergence.*`
- **生成脚本**: `make_figs_ch4_scvx_convergence.py`
- **输出目录**: `outputs/figures/ch4_scvx_convergence`
- **说明**: 由 SCvx 收敛日志绘制的迭代曲线
- **备注**: 需要 SCvx 收敛日志 CSV 数据

### 表4-2: 离散优化问题变量与约束量纲统计表

- **类型**: 表
- **生成方式**: 硬编码
- **预期文件名**: `table4_02_scvx_dimension_stats.*`
- **生成脚本**: `make_tables_ch4_core.py`
- **输出目录**: `outputs/tables/`
- **说明**: 统计 SCvx 离散化后的变量数和约束数随节点数 N 的关系
- **备注**: 可根据离散化公式硬编码

## 4.3 学习增强的热启动方法

### 图4-8: 学习热启动的数据流示意图

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_08_warmstart_dataflow.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: 展示从故障状态到热启动初值的数据流
- **备注**: 数据流示意图

### 图4-9: 神经网络结构示意图

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_09_network_architecture.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: 展示学习热启动模型的网络结构
- **备注**: 网络结构图，可用 draw.io 或 TikZ 绘制

### 图4-10: 学习热启动模型的训练收敛曲线

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_10_learning_curves.*`
- **生成脚本**: `make_figs_ch4_learning_curves.py`
- **输出目录**: `outputs/figures/ch4_learning`
- **说明**: Train/Val MSE loss vs epoch
- **备注**: 需要 train_log.json 训练日志

### 图4-11: 单样本高度轨迹预测效果图

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_11_single_sample_altitude.*`
- **生成脚本**: `make_figs_ch4_learning_altitude.py`
- **输出目录**: `outputs/figures/ch4_learning`
- **说明**: 展示模型对单个样本的高度轨迹预测 vs 真值
- **备注**: 需要 dataset.npz 和训练好的模型

### 图4-12: 多故障工况下的高度轨迹预测效果

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_12_fault_cases_altitude.*`
- **生成脚本**: `make_figs_ch4_learning_altitude.py`
- **输出目录**: `outputs/figures/ch4_learning`
- **说明**: F1-F5 不同故障下的预测效果对比
- **备注**: 多子图展示不同故障的预测效果

### 图4-13: 各节点平均预测误差分布

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_13_mean_node_error.*`
- **生成脚本**: `make_figs_ch4_learning_altitude.py`
- **输出目录**: `outputs/figures/ch4_learning`
- **说明**: 时间维度上的预测误差累积特性
- **备注**: 误差随节点编号的变化

### 图4-14: 高度预测 RMSE 分布直方图

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_14_rmse_histogram.*`
- **生成脚本**: `make_figs_ch4_learning_altitude.py`
- **输出目录**: `outputs/figures/ch4_learning`
- **说明**: 统计角度展示预测精度
- **备注**: RMSE 直方图

### 图4-15: 自适应罚权重随故障强度 eta 的变化

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_15_adaptive_weights_vs_eta.*`
- **生成脚本**: `make_figs_ch4_adaptive_weights.py`
- **输出目录**: `outputs/figures/ch4_adaptive_weights`
- **说明**: 展示自适应权重调整策略
- **备注**: 权重随 eta 的变化曲线

### 图4-16: 冷/热启动外层迭代次数对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_16_warmstart_iterations.*`
- **生成脚本**: `make_figs_ch4_warmstart_performance.py`
- **输出目录**: `outputs/figures/ch4_warmstart`
- **说明**: F1-F5 不同故障下 cold vs warm 的迭代次数
- **备注**: 条形图对比

### 图4-17: 冷/热启动平均求解时间对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_17_warmstart_time.*`
- **生成脚本**: `make_figs_ch4_warmstart_performance.py`
- **输出目录**: `outputs/figures/ch4_warmstart`
- **说明**: F1-F5 不同故障下 cold vs warm 的 CPU 时间
- **备注**: 条形图对比

### 图4-18: 不同故障下终端高度 RMSE 分布

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_18_terminal_rmse_distribution.*`
- **生成脚本**: `make_figs_ch4_warmstart_performance.py`
- **输出目录**: `outputs/figures/ch4_warmstart`
- **说明**: 展示不同故障类型下的终端高度误差分布
- **备注**: 箱线图或小提琴图

### 表4-3: 学习热启动网络超参数说明表

- **类型**: 表
- **生成方式**: 硬编码
- **预期文件名**: `table4_03_network_hyperparams.*`
- **生成脚本**: `make_tables_ch4_core.py`
- **输出目录**: `outputs/tables/`
- **说明**: 网络结构、训练参数等说明
- **备注**: 硬编码网络参数

## 4.4 任务域自适应重构策略

### 图4-19: 任务域划分示意图

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_19_mission_domain_concept.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: retain/degraded/safe-area 三种任务域的概念图
- **备注**: 概念示意图

### 图4-20: 任务域切换的有限状态机

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_20_domain_fsm.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: FSM 状态转换图
- **备注**: 状态机图，建议用 draw.io 绘制

### 图4-21: 安全区域几何示意图

- **类型**: 图
- **生成方式**: 手工绘制
- **预期文件名**: `fig4_21_safe_area_geometry.*`
- **输出目录**: `outputs/figures/ch4_schematics`
- **说明**: safe-area 目标区域的几何定义
- **备注**: 几何示意图

### 图4-22: retain 域下五种典型故障的关键飞行量响应

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_22_retain_domain_responses.*`
- **生成脚本**: `make_figs_ch4_mission_domains.py`
- **输出目录**: `outputs/figures/ch4_mission_domains`
- **说明**: 轨迹/约束时间历程对比
- **备注**: 多子图展示高度/速度/约束

### 图4-23: F1 severe 三种任务域下行距对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_23_f1_severe_downrange.*`
- **生成脚本**: `make_figs_ch4_mission_domains.py`
- **输出目录**: `outputs/figures/ch4_mission_domains`
- **说明**: retain/degraded/safe-area 下的行距曲线
- **备注**: 行距随时间变化曲线

### 图4-24: 五种严重故障终端行距汇总对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_24_severe_downrange_summary.*`
- **生成脚本**: `make_figs_ch4_mission_domains.py`
- **输出目录**: `outputs/figures/ch4_mission_domains`
- **说明**: F1-F5 severe 在三种任务域下的终端行距
- **备注**: 分组条形图

### 表4-4: 任务域与对应优化问题形式对照表

- **类型**: 表
- **生成方式**: 硬编码
- **预期文件名**: `table4_04_mission_domain_mapping.*`
- **生成脚本**: `make_tables_ch4_core.py`
- **输出目录**: `outputs/tables/`
- **说明**: retain/degraded/safe-area 对应的优化目标和约束形式
- **备注**: 硬编码任务域配置

## 4.5 仿真验证与分析

### 图4-25: F1推力降级故障下的轨迹重规划对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_25_F1_trajectory_replan.*`
- **生成脚本**: `make_figs_ch4_trajectories_replan.py`
- **输出目录**: `outputs/figures/ch4_trajectories_replan`
- **说明**: 名义/故障开环/重规划轨迹的高度对比（不同eta）
- **备注**: F1故障不同严重度下的轨迹对比

### 图4-26: F2 TVC速率限制故障下的轨迹重规划对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_26_F2_trajectory_replan.*`
- **生成脚本**: `make_figs_ch4_trajectories_replan.py`
- **输出目录**: `outputs/figures/ch4_trajectories_replan`
- **说明**: 名义/故障开环/重规划轨迹的高度对比（不同eta）
- **备注**: F2故障不同严重度下的轨迹对比

### 图4-27: F3 TVC卡滞故障下的轨迹重规划对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_27_F3_trajectory_replan.*`
- **生成脚本**: `make_figs_ch4_trajectories_replan.py`
- **输出目录**: `outputs/figures/ch4_trajectories_replan`
- **说明**: 名义/故障开环/重规划轨迹的高度对比（不同eta）
- **备注**: F3故障不同严重度下的轨迹对比

### 图4-28: F4传感器偏置故障下的轨迹重规划对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_28_F4_trajectory_replan.*`
- **生成脚本**: `make_figs_ch4_trajectories_replan.py`
- **输出目录**: `outputs/figures/ch4_trajectories_replan`
- **说明**: 名义/故障开环/重规划轨迹的高度对比（不同eta）
- **备注**: F4故障不同严重度下的轨迹对比

### 图4-29: F5事件延迟故障下的轨迹重规划对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_29_F5_trajectory_replan.*`
- **生成脚本**: `make_figs_ch4_trajectories_replan.py`
- **输出目录**: `outputs/figures/ch4_trajectories_replan`
- **说明**: 名义/故障开环/重规划轨迹的高度对比（不同eta）
- **备注**: F5故障不同严重度下的轨迹对比

### 图4-30: 五种故障场景轨迹重规划汇总对比

- **类型**: 图
- **生成方式**: 自动生成
- **预期文件名**: `fig4_30_all_faults_summary.*`
- **生成脚本**: `make_figs_ch4_trajectories_replan.py`
- **输出目录**: `outputs/figures/ch4_trajectories_replan`
- **说明**: F1-F5所有故障场景的轨迹重规划汇总对比图
- **备注**: 汇总对比图
