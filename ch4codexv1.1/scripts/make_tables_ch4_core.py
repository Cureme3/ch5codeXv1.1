#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成第四章核心硬编码表格。

输出文件（符合规范命名）：
- outputs/tables/table4_02_scvx_dimension_stats.md
- outputs/tables/table4_03_network_hyperparams.md
- outputs/tables/table4_04_mission_domain_mapping.md

命令行用法：
    python -m scripts.make_tables_ch4_core
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"


def generate_table4_02_scvx_dimension_stats() -> str:
    """表4-2：离散优化问题变量与约束量纲统计表。

    基于论文中的离散化公式，统计变量数和约束数随节点数 N 的关系。
    """
    content = """# 表4-2 离散优化问题变量与约束量纲统计表

本表统计 SCvx 离散化后的变量数和约束数随节点数 N 的关系。

## 变量统计

| 变量类型 | 维度 | 数量公式 | N=40 | N=60 | N=80 |
|----------|------|----------|------|------|------|
| 状态变量 x | 7 | 7×N | 280 | 420 | 560 |
| 控制变量 u | 3 | 3×(N-1) | 117 | 177 | 237 |
| 动压松弛 sq | 1 | N | 40 | 60 | 80 |
| 过载松弛 sn | 1 | N | 40 | 60 | 80 |
| 推力锥松弛 sc | 1 | N-1 | 39 | 59 | 79 |
| **总变量数** | - | 12N-2 | 478 | 718 | 958 |

## 约束统计

| 约束类型 | 数量公式 | N=40 | N=60 | N=80 |
|----------|----------|------|------|------|
| 动力学等式 | 7×(N-1) | 273 | 413 | 553 |
| 初始状态等式 | 7 | 7 | 7 | 7 |
| 终端约束 | 4 | 4 | 4 | 4 |
| 信赖域（状态） | 7×N | 280 | 420 | 560 |
| 信赖域（控制） | 3×(N-1) | 117 | 177 | 237 |
| 动压约束 | N | 40 | 60 | 80 |
| 过载约束 | N | 40 | 60 | 80 |
| 推力锥约束（SOCP） | N-1 | 39 | 59 | 79 |
| 推力边界约束 | 2×(N-1) | 78 | 118 | 158 |
| **总约束数** | - | 878 | 1318 | 1758 |

## 备注

1. 状态变量 x = [h, v, γ, m, r, θ, ψ]^T，共 7 维
2. 控制变量 u = [T, α, β]^T，共 3 维
3. 推力锥约束为二阶锥约束，统计为锥约束数量
4. 实际求解规模还受松弛变量和对偶变量影响
"""
    return content


def generate_table4_03_network_hyperparams() -> str:
    """表4-3：学习热启动网络超参数说明表。"""
    content = """# 表4-3 学习热启动网络超参数说明表

本表描述学习热启动模型的网络结构和训练参数。

## 网络结构

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入维度 | 11 | 故障状态特征（含 eta） |
| 输出维度 | N×7 | N 个节点的 7 维状态预测 |
| 隐藏层数 | 3 | 全连接隐藏层 |
| 隐藏层宽度 | 256, 512, 256 | 各层神经元数 |
| 激活函数 | ReLU | 隐藏层激活 |
| 输出激活 | 无 | 线性输出 |

## 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | Adam | 自适应学习率优化器 |
| 初始学习率 | 1e-3 | - |
| 学习率衰减 | StepLR | 每50轮衰减0.5 |
| 批大小 | 32 | Mini-batch 大小 |
| 训练轮数 | 200 | 最大训练轮数 |
| 早停耐心 | 20 | 验证损失不下降则停止 |
| 损失函数 | MSE | 均方误差 |

## 数据集划分

| 划分 | 比例 | 样本数（典型） |
|------|------|---------------|
| 训练集 | 70% | 1400 |
| 验证集 | 15% | 300 |
| 测试集 | 15% | 300 |

## 输入特征说明

| 序号 | 特征名 | 说明 |
|------|--------|------|
| 1 | h_fault | 故障时刻高度 (km) |
| 2 | v_fault | 故障时刻速度 (m/s) |
| 3 | gamma_fault | 故障时刻航迹倾角 (rad) |
| 4 | m_fault | 故障时刻质量 (kg) |
| 5 | t_fault | 故障时刻时间 (s) |
| 6 | eta | 故障严重度 [0,1] |
| 7-11 | fault_type_onehot | 故障类型独热编码 (F1-F5) |
"""
    return content


def generate_table4_04_mission_domain_mapping() -> str:
    """表4-4：任务域与对应优化问题形式对照表。"""
    content = """# 表4-4 任务域与对应优化问题形式对照表

本表描述三种任务域对应的优化目标和约束形式。

## 任务域定义

| 任务域 | 中文名 | 触发条件 | 核心目标 |
|--------|--------|----------|----------|
| RETAIN | 保持入轨 | η < 0.3 或轻微故障 | 完成轨道插入任务 |
| DEGRADED | 降级任务 | 0.3 ≤ η < 0.7 | 尽量保持轨道要素 |
| SAFE_AREA | 安全区域 | η ≥ 0.7 或严重故障 | 安全落入指定区域 |

## 优化问题形式对照

| 任务域 | 终端约束 | 终端目标权重 | 路径约束放松 | 控制约束 |
|--------|----------|--------------|--------------|----------|
| RETAIN | 严格轨道要素 | 高（×1.5） | 不放松 | 标准边界 |
| DEGRADED | 软约束化轨道要素 | 中（×1.0） | 轻微放松（1.1倍） | 标准边界 |
| SAFE_AREA | 安全区域位置 | 低（×0.5） | 大幅放松（1.5倍） | 放宽边界 |

## 权重缩放系数

| 任务域 | 终端权重 | 状态权重 | 控制权重 | 松弛权重 |
|--------|----------|----------|----------|----------|
| RETAIN | 1.5 | 1.0 | 1.0 | 1.0 |
| DEGRADED | 1.0 | 1.0 | 0.8 | 2.0 |
| SAFE_AREA | 0.5 | 0.5 | 0.5 | 5.0 |

## 终端条件说明

### RETAIN 域终端约束
- 近地点高度: h_p = h_target ± 10 km
- 远地点高度: h_a = h_target ± 50 km
- 轨道倾角: i = i_target ± 0.5°
- 升交点经度: Ω = Ω_target ± 1°

### DEGRADED 域终端约束
- 近地点高度: h_p ≥ 150 km（最低可用轨道）
- 速度方向: 接近轨道切向
- 高度/速度软约束，允许偏差

### SAFE_AREA 域终端约束
- 落点经度: λ ∈ [λ_min, λ_max]（安全区域边界）
- 落点纬度: φ ∈ [φ_min, φ_max]
- 终端速度: v_f < v_max（安全着陆速度）
- 终端航迹角: γ_f < -60°（陡峭再入）

## 任务域切换逻辑

```
if η < 0.3 and constraints_feasible:
    domain = RETAIN
elif η < 0.7 and degraded_feasible:
    domain = DEGRADED
else:
    domain = SAFE_AREA
```

## 备注

1. η 为故障严重度参数，由故障诊断模块估计
2. 任务域可在飞行中动态切换（escalation）
3. 切换只允许从 RETAIN → DEGRADED → SAFE_AREA 单向进行
4. 每次切换后需重新规划轨迹
"""
    return content


def main() -> None:
    """主函数：生成所有核心表格。"""
    print("=" * 60)
    print("生成第四章核心表格")
    print("=" * 60)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 表4-2
    print("\n[1/3] 生成表4-2：离散优化问题变量与约束量纲统计表...")
    content = generate_table4_02_scvx_dimension_stats()
    path = TABLES_DIR / "table4_02_scvx_dimension_stats.md"
    path.write_text(content, encoding="utf-8")
    print(f"  - 已保存: {path}")

    # 表4-3
    print("\n[2/3] 生成表4-3：学习热启动网络超参数说明表...")
    content = generate_table4_03_network_hyperparams()
    path = TABLES_DIR / "table4_03_network_hyperparams.md"
    path.write_text(content, encoding="utf-8")
    print(f"  - 已保存: {path}")

    # 表4-4
    print("\n[3/3] 生成表4-4：任务域与对应优化问题形式对照表...")
    content = generate_table4_04_mission_domain_mapping()
    path = TABLES_DIR / "table4_04_mission_domain_mapping.md"
    path.write_text(content, encoding="utf-8")
    print(f"  - 已保存: {path}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
