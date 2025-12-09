# 项目算法与方法清单

本项目是航天火箭着陆/轨道插入的综合系统，集成故障诊断、轨迹优化与动力学仿真。

---

## 第三章 (Ch3) - 故障诊断与检测

### 分类算法
| 算法 | 文件路径 | 描述 |
|-----|---------|------|
| K-Means聚类 | `ch3codev1.1/diagnosis/classifier.py` | 无监督聚类，用于RBF核心选择 |
| RBF特征映射 | `ch3codev1.1/diagnosis/classifier.py` | 高斯径向基函数：`φ(x,c) = exp(-0.5*||x-c||²/σ²)` |
| 岭回归 | `ch3codev1.1/diagnosis/classifier.py` | 闭式解：`W = (ΦᵀΦ + λI)⁻¹ΦᵀY` |
| 加权岭回归 | `ch3codev1.1/diagnosis/classifier.py` | 类别均衡加权 |

### 状态估计
| 算法 | 文件路径 | 描述 |
|-----|---------|------|
| 扩展状态观测器(ESO) | `ch3codev1.1/diagnosis/eso.py` | 4维状态估计，观测加速度残差 |

### 特征提取
| 算法 | 文件路径 | 描述 |
|-----|---------|------|
| 短时傅里叶变换(STFT) | `ch3codev1.1/diagnosis/features.py` | 时频分���，Hanning窗 |
| 伪Wigner-Ville分布(PWVD) | `ch3codev1.1/diagnosis/features.py` | 高分辨率时频表示 |
| 样本熵 | `ch3codev1.1/diagnosis/features.py` | 非线性复杂度度量 |
| 频带能量特征 | `ch3codev1.1/diagnosis/features.py` | 低/中/高频带能量 |
| 时频熵 | `ch3codev1.1/diagnosis/features.py` | Shannon熵 |

---

## 第四章 (Ch4) - 轨迹优化与重规划

### 凸优化算法
| 算法 | 文件路径 | 描述 |
|-----|---------|------|
| 逐次凸优化(SCvx) | `ch4codexv1.1/opt/scvx.py` | 主循环：线性化→SOCP→信赖域更新→收敛判据 |
| 动力学线性化 | `ch4codexv1.1/src/opt/linearization.py` | 雅可比矩阵：`A = ∂f/∂x`, `B = ∂f/∂u` |
| 二阶锥规划(SOCP) | `ch4codexv1.1/opt/socp_problem.py` | CVXPy求解器，支持ECOS/SCS |
| 信赖域策略 | `ch4codexv1.1/opt/scvx.py` | 改进比ρ自适应调整 |
| 时间离散化 | `ch4codexv1.1/opt/discretization.py` | 均匀/分段网格生成 |

### 约束与目标函数
| 约束类型 | 文件路径 | 描述 |
|---------|---------|------|
| 动力学等式约束 | `ch4codexv1.1/opt/socp_problem.py` | `x_{k+1} = A_k x_k + B_k u_k + c_k` |
| 推力锥约束 | `ch4codexv1.1/opt/socp_problem.py` | 二阶锥：`||T_xy|| ≤ tan(θ)*T_z` |
| 动压限制 | `ch4codexv1.1/opt/socp_problem.py` | 路径约束 |
| 过载限制 | `ch4codexv1.1/opt/socp_problem.py` | 路径约束 |
| 终端约束 | `ch4codexv1.1/opt/socp_problem.py` | 软约束：高度/速度/飞行路径角 |

### 学习与热启动
| 算法 | 文件路径 | 描述 |
|-----|---------|------|
| 多层感知机(MLP) | `ch4codexv1.1/src/learn/model.py` | 3层网络，ReLU激活，高度序列预测 |
| 监督学习训练 | `ch4codexv1.1/src/learn/train.py` | PyTorch，Adam优化器，MSE损失 |
| 热启动策略 | `ch4codexv1.1/src/learn/warmstart.py` | 从故障特征预测初始轨迹 |
| 自适应罚权重 | `ch4codexv1.1/src/learn/weights.py` | 根据故障强度η动态调整权重 |

### 故障场景与任务域
| 方法 | 文件路径 | 描述 |
|-----|---------|------|
| 故障场景库 | `ch4codexv1.1/src/sim/scenarios.py` | F1推力降级、F2TVC速率限制、F3TVC卡死、F4传感器偏差、F5事件延迟 |
| 故障诊断桥接 | `ch4codexv1.1/src/sim/diag_bridge.py` | Ch3诊断→Ch4场景映射 |
| 任务域分割 | `ch4codexv1.1/src/sim/mission_domains.py` | RETAIN(η<0.3)→DEGRADED(0.3≤η<0.7)→SAFE_AREA(η≥0.7) |

---

## 动力学与仿真

| 算法 | 文件路径 | 描述 |
|-----|---------|------|
| 标准大气模型(US Std 1976) | `ch4codexv1.1/src/sim/kz1a_eci_core.py` | 分层大气，温度/压力/密度/音速 |
| 阻力计算 | `ch4codexv1.1/src/sim/kz1a_eci_core.py` | `D = 0.5*ρ*v²*Cd*A` |
| ECI坐标系变换 | `ch4codexv1.1/src/sim/kz1a_eci_core.py` | ENU→ECI变换 |
| 3-DoF动力学 | `ch4codexv1.1/src/sim/kz1a_eci_core.py` | 位置/速度/质量6自由度简化模型 |
| 开环制导 | `ch4codexv1.1/src/sim/kz1a_eci_core.py` | 分段线性俯仰角制导 |
| 名义轨迹仿真 | `ch4codexv1.1/src/sim/run_nominal.py` | 无故障开环仿真 |
| 故障开环仿真 | `ch4codexv1.1/src/sim/run_fault.py` | 注入故障场景仿真 |

---

## 关键特性总结

**凸优化：** SCvx框架 + SOCP子问题 + 信赖域自适应

**故障诊断：** RBF分类器 + ESO残差估计 + 多特征融合(STFT/PWVD/样本熵)

**轨迹优化：** 多约束处理 + 自适应罚权重 + MLP热启动

**任务管理：** 三层任务域分级 + 故障诊断→重规划完整流程
