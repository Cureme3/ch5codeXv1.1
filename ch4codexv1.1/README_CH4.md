# 第四章：基于凸优化学习的故障响应轨迹规划

本目录包含论文第四章所有图表的生成代码和输出。

## 快速开始

### 环境要求

- Python 3.8+
- NumPy, Pandas, Matplotlib
- PyTorch (用于学习模型)
- cvxpy (可选，用于 SCvx 求解)

### 一键生成所有图表

```bash
# 快速模式（跳过耗时步骤）
python -m scripts.generate_ch4_outputs --quick

# 完整模式（生成所有图表）
python -m scripts.generate_ch4_outputs --mode full

# 默认模式
python -m scripts.generate_ch4_outputs
```

## 图表索引

### 手动绘制图表（需单独准备）

| 图号 | 标题 | 存放目录 |
|------|------|----------|
| 图4-1 | 本章在全文中的位置 | `outputs/figures/ch4_manual/` |
| 图4-2 | SCvx算法与学习热启动的集成框架 | `outputs/figures/ch4_manual/` |
| 图4-3 | 连续时间凸优化问题的离散化示意 | `outputs/figures/ch4_manual/` |
| 图4-4 | 逐次凸化处理非凸约束示意 | `outputs/figures/ch4_manual/` |
| 图4-5 | 信赖域约束示意 | `outputs/figures/ch4_manual/` |
| 图4-6 | SCvx算法流程 | `outputs/figures/ch4_manual/` |
| 图4-8 | 学习热启动策略集成示意 | `outputs/figures/ch4_manual/` |
| 图4-9 | 残差网络结构 | `outputs/figures/ch4_manual/` |
| 图4-19 | 任务域选择逻辑 | `outputs/figures/ch4_manual/` |
| 图4-20 | 任务域状态机 | `outputs/figures/ch4_manual/` |
| 图4-21 | 自动升级流程 | `outputs/figures/ch4_manual/` |

### 自动生成图表

#### 学习模型相关 (4.3节)

| 图号 | 标题 | 生成脚本 | 输出文件 |
|------|------|----------|----------|
| 图4-10 | 学习曲线 | `make_figs_ch4_learning_curves.py` | `ch4_learning/fig4_10_*.png` |
| 图4-11 | 单样本高度预测 | `make_figs_ch4_learning_altitude.py` | `ch4_learning/fig4_11_*.png` |
| 图4-12 | 多故障高度预测 | `make_figs_ch4_learning_altitude.py` | `ch4_learning/fig4_12_*.png` |
| 图4-13 | 节点误差分布 | `make_figs_ch4_learning_altitude.py` | `ch4_learning/fig4_13_*.png` |
| 图4-14 | RMSE直方图 | `make_figs_ch4_learning_altitude.py` | `ch4_learning/fig4_14_*.png` |
| 图4-15 | 自适应权重曲线 | `make_figs_ch4_adaptive_weights.py` | `ch4_adaptive/fig4_15_*.png` |

#### 热启动性能 (4.3节)

| 图号 | 标题 | 生成脚本 | 输出文件 |
|------|------|----------|----------|
| 图4-16 | 迭代次数对比 | `make_figs_ch4_warmstart_performance.py` | `ch4_warmstart/fig4_16_*.png` |
| 图4-17 | 求解时间对比 | `make_figs_ch4_warmstart_performance.py` | `ch4_warmstart/fig4_17_*.png` |
| 图4-18 | 终端RMSE分布 | `make_figs_ch4_warmstart_performance.py` | `ch4_warmstart/fig4_18_*.png` |

#### 任务域相关 (4.4节)

| 图号 | 标题 | 生成脚本 | 输出文件 |
|------|------|----------|----------|
| 图4-22 | RETAIN域响应 | `make_figs_ch4_mission_domains.py` | `ch4_mission_domains/fig4_22_*.png` |
| 图4-23 | F1故障任务域对比 | `make_figs_ch4_mission_domains.py` | `ch4_mission_domains/fig4_23_*.png` |
| 图4-24 | 终端状态汇总 | `make_figs_ch4_mission_domains.py` | `ch4_mission_domains/fig4_24_*.png` |

#### 轨迹重规划 (4.5节)

| 图号 | 标题 | 生成脚本 | 输出文件 |
|------|------|----------|----------|
| 图4-25 | F1轨迹重规划 | `make_figs_ch4_trajectories_replan.py` | `ch4_trajectories_replan/fig4_25_*.png` |
| 图4-26 | F2轨迹重规划 | `make_figs_ch4_trajectories_replan.py` | `ch4_trajectories_replan/fig4_26_*.png` |
| 图4-27 | F3轨迹重规划 | `make_figs_ch4_trajectories_replan.py` | `ch4_trajectories_replan/fig4_27_*.png` |
| 图4-28 | F4轨迹重规划 | `make_figs_ch4_trajectories_replan.py` | `ch4_trajectories_replan/fig4_28_*.png` |
| 图4-29 | F5轨迹重规划 | `make_figs_ch4_trajectories_replan.py` | `ch4_trajectories_replan/fig4_29_*.png` |
| 图4-30 | 五故障汇总 | `make_figs_ch4_trajectories_replan.py` | `ch4_trajectories_replan/fig4_30_*.png` |

### 表格

| 表号 | 标题 | 生成脚本 | 输出文件 |
|------|------|----------|----------|
| 表4-2 | SCvx问题维度 | `make_tables_ch4_core.py` | `tables/table4_02_*.md` |
| 表4-3 | 任务域统计 | `eval_ch4_mission_domains.py` | `ch4/tables/table_ch4_mission_domains.md` |
| 表4-4 | 网络超参数 | `make_tables_ch4_core.py` | `tables/table4_04_*.md` |

## 目录结构

```
outputs/
├── data/                          # 中间数据
│   ├── ch4_learning/              # 学习模型数据
│   │   ├── dataset.npz            # 训练数据集
│   │   ├── model.pt               # 训练好的模型
│   │   └── train_log.json         # 训练日志
│   └── ch4_trajectories_replan/   # 轨迹重规划数据
│       ├── nominal.npz            # 名义轨迹
│       ├── F1_eta02_replan.npz    # F1故障重规划轨迹
│       └── ...                    # 其他故障轨迹
├── figures/                       # 图像文件
│   ├── ch4_learning/              # 学习曲线与预测图
│   ├── ch4_warmstart/             # 热启动性能图
│   ├── ch4_mission_domains/       # 任务域响应图
│   ├── ch4_trajectories_replan/   # 轨迹重规划对比图
│   ├── ch4_manual/                # 手动绘制图表占位
│   └── ...                        # 其他图表
├── tables/                        # Markdown表格
└── ch4/tables/                    # 第四章专用表格
```

## 关键脚本说明

### 数据生成

- `gen_ch4_learning_dataset.py`: 生成学习模型训练数据集
- `eval_ch4_trajectories_replan.py`: 运行故障场景仿真并保存轨迹数据

### 模型训练

- `train_ch4_learning.py`: 训练热启动预测神经网络

### 图表生成

- `make_figs_ch4_learning_curves.py`: 学习曲线图 (图4-10)
- `make_figs_ch4_learning_altitude.py`: 高度预测效果图 (图4-11~4-14)
- `make_figs_ch4_adaptive_weights.py`: 自适应权重图 (图4-15)
- `make_figs_ch4_warmstart_performance.py`: 热启动性能图 (图4-16~4-18)
- `make_figs_ch4_mission_domains.py`: 任务域响应图 (图4-22~4-24)
- `make_figs_ch4_trajectories_replan.py`: 轨迹重规划图 (图4-25~4-30)

### 表格生成

- `make_tables_ch4_core.py`: 核心表格 (表4-2/4-4)
- `eval_ch4_mission_domains.py`: 任务域统计表 (表4-3)

### 检查与报告

- `spec_ch4_fig_table_manifest.py`: 图表规范清单
- `check_ch4_fig_table_coverage.py`: 覆盖率检查
- `make_final_status_report.py`: 最终状态报告

## 故障场景

| ID | 类型 | 参数 | 描述 |
|----|------|------|------|
| F1 | 推力降级 | degrade_frac | 发动机推力下降 |
| F2 | TVC速率限制 | tvc_rate_deg_s | 推力矢量控制速率受限 |
| F3 | TVC卡滞 | stuck_angle_deg | 推力矢量控制卡在固定角度 |
| F4 | 传感器偏置 | sensor_bias_deg | 姿态传感器存在恒定偏差 |
| F5 | 事件延迟 | event_delay_s | 级间分离等事件延迟 |

## 任务域判定规则

根据故障严重度 η 自动判定任务域：

- η < 0.3: **RETAIN**（保持入轨）— 继续追求原定轨道插入
- 0.3 ≤ η < 0.7: **DEGRADED**（降级任务）— 降低轨道要求
- η ≥ 0.7: **SAFE_AREA**（安全区域）— 放弃入轨，转向安全区域

## 常见问题

### Q: 缺少轨迹数据？

运行轨迹生成脚本：
```bash
python -m scripts.eval_ch4_trajectories_replan
```

### Q: 模型训练失败？

先生成训练数据集：
```bash
python -m scripts.gen_ch4_learning_dataset --samples 200
```

然后训练模型：
```bash
python -m scripts.train_ch4_learning --epochs 200
```

### Q: 图表中文显示乱码？

确保系统安装了 SimHei 或 Microsoft YaHei 字体，或修改 matplotlib 配置使用其他中文字体。
