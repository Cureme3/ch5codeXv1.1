# 第四章 图表与数据索引

本文档由 `make_index_ch4_figs_tables.py` 自动生成，列出第四章所有图表和数据表文件。

---

## 图表索引

### 图 4.3：自适应罚权重

- `figures\ch4_adaptive_weights\adaptive_weights_vs_eta.png`
- `figures\ch4_adaptive_weights\fig4_15_adaptive_weights_vs_eta.png`

### 图 4.3：学习模型训练与预测

- `figures\ch4_learning\fig4_10_learning_curves.png`
- `figures\ch4_learning\fig4_11_single_sample_altitude.png`
- `figures\ch4_learning\fig4_12_fault_cases_altitude.png`
- `figures\ch4_learning\fig4_13_mean_node_error.png`
- `figures\ch4_learning\fig4_14_rmse_histogram.png`

### 图 4.4：任务域划分与切换

- `figures\ch4_mission_domains\fig4_22_retain_domain_responses.png`
- `figures\ch4_mission_domains\fig4_23_f1_severe_downrange.png`
- `figures\ch4_mission_domains\fig4_24_severe_downrange_summary.png`

### 图 4.2：SCvx 收敛分析

- `figures\ch4_scvx_convergence\fig4_07_scvx_convergence_cost.png`
- `figures\ch4_scvx_convergence\fig4_07_scvx_convergence_decomp.png`
- `figures\ch4_scvx_convergence\fig4_07_scvx_convergence_feas.png`
- `figures\ch4_scvx_convergence\fig4_07_scvx_convergence_trustregion.png`

### 图 4.X：Ch4 Trajectories Replan

- `figures\ch4_trajectories_replan\fig4_25_F1_trajectory_replan.png`
- `figures\ch4_trajectories_replan\fig4_26_F2_trajectory_replan.png`
- `figures\ch4_trajectories_replan\fig4_27_F3_trajectory_replan.png`
- `figures\ch4_trajectories_replan\fig4_28_F4_trajectory_replan.png`
- `figures\ch4_trajectories_replan\fig4_29_F5_trajectory_replan.png`
- `figures\ch4_trajectories_replan\fig4_30_all_faults_summary.png`

### 图 4.3：学习热启动性能

- `figures\ch4_warmstart\fig4_16_warmstart_iterations.png`
- `figures\ch4_warmstart\fig4_17_warmstart_time.png`
- `figures\ch4_warmstart\fig4_18_terminal_rmse_distribution.png`

### 图 4.2：约束分析

- `figures\constraints\F1_thrust_deg15_n_comparison.png`
- `figures\constraints\F1_thrust_deg15_q_comparison.png`
- `figures\constraints\F2_tvc_rate4_n_comparison.png`
- `figures\constraints\F2_tvc_rate4_q_comparison.png`
- `figures\constraints\F3_tvc_stuck3deg_n_comparison.png`
- `figures\constraints\F3_tvc_stuck3deg_q_comparison.png`
- `figures\constraints\F4_sensor_bias2deg_n_comparison.png`
- `figures\constraints\F4_sensor_bias2deg_q_comparison.png`
- `figures\constraints\F5_event_delay5s_n_comparison.png`
- `figures\constraints\F5_event_delay5s_q_comparison.png`

### 图 4.4：任务域对比

- `figures\domain_comparison\domain_configuration_comparison.png`

### 图 4.3：Eta 缩放分析

- `figures\eta_scaling\adaptive_weights_vs_eta.png`
- `figures\eta_scaling\eta_scaling_all_faults.png`

### 图 4.1：轨迹仿真

- `figures\trajectories\F1_thrust_deg15_altitude_comparison.png`
- `figures\trajectories\F1_thrust_deg15_fpa_comparison.png`
- `figures\trajectories\F1_thrust_deg15_hv_portrait.png`
- `figures\trajectories\F1_thrust_deg15_velocity_comparison.png`
- `figures\trajectories\F2_tvc_rate4_altitude_comparison.png`
- `figures\trajectories\F2_tvc_rate4_fpa_comparison.png`
- `figures\trajectories\F2_tvc_rate4_hv_portrait.png`
- `figures\trajectories\F2_tvc_rate4_velocity_comparison.png`
- `figures\trajectories\F3_tvc_stuck3deg_altitude_comparison.png`
- `figures\trajectories\F3_tvc_stuck3deg_fpa_comparison.png`
- `figures\trajectories\F3_tvc_stuck3deg_hv_portrait.png`
- `figures\trajectories\F3_tvc_stuck3deg_velocity_comparison.png`
- `figures\trajectories\F4_sensor_bias2deg_altitude_comparison.png`
- `figures\trajectories\F4_sensor_bias2deg_fpa_comparison.png`
- `figures\trajectories\F4_sensor_bias2deg_hv_portrait.png`
- `figures\trajectories\F4_sensor_bias2deg_velocity_comparison.png`
- `figures\trajectories\F5_event_delay5s_altitude_comparison.png`
- `figures\trajectories\F5_event_delay5s_fpa_comparison.png`
- `figures\trajectories\F5_event_delay5s_hv_portrait.png`
- `figures\trajectories\F5_event_delay5s_velocity_comparison.png`
- `figures\trajectories\fault_scenarios_timeline.png`

---

## 数据表索引

### CSV 数据表

- `ch4\tables\ch4_mission_domains.csv`
- `data\ch4_scvx_convergence_F1_thrust_deg15_eta0.50.csv`
- `data\ch4_scvx_convergence_F2_tvc_rate4_eta0.50.csv`
- `data\scvx_convergence_log.csv`

### Markdown 表格

- `ch4\tables\table_ch4_mission_domains.md`
- `tables\adaptive_weights.md`
- `tables\ch4_figure_table_coverage_report.md`
- `tables\ch4_figure_table_final_status.md`
- `tables\ch4_figure_table_spec.md`
- `tables\eta_scaling_parameters.md`
- `tables\mission_domain_config.md`
- `tables\table4_02_scvx_dimension_stats.md`
- `tables\table4_03_network_hyperparams.md`
- `tables\table4_04_mission_domain_mapping.md`
- `tables\table_ch4_adaptive_weights.md`
- `tables\table_scvx_stats.md`

---

## 使用说明

1. 图表引用：在论文中引用图表时，可参考本索引中的文件路径。
2. 图号占位符：索引中的图号（如"图 4.X"）为占位符，需根据论文实际章节结构手工调整。
3. 更新索引：重新生成图表后，运行 `python -m scripts.make_index_ch4_figs_tables` 更新本索引。
4. 批量生成：
   - 生成所有图表：`python -m scripts.make_figs_ch4`
   - 生成所有表格：`python -m scripts.make_tables_ch4`
   - 生成索引：`python -m scripts.make_index_ch4_figs_tables`
