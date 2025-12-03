# 快舟一号甲第 4 章仿真实验工程

本目录提供论文《基于学习增强的凸优化的轨迹在线重构方法》第 4 章的代码实现，覆盖 6DoF 飞行仿真、SCvx 重构、学习热启动对比与任务域判别。

## 依赖安装

Python 3.10，配合下列依赖：

- numpy
- scipy
- pandas
- matplotlib
- cvxpy（默认使用 ECOS，可按需安装 OSQP-SOCP）
- pyyaml

```bash
pip install -r requirements.txt
```

## 目录概览

- `configs/`：KZ-1A 火箭参数、SCvx 配置、Monte Carlo 场景设定。
- `src/dyn6dof/`：坐标系工具、标准大气、气动力、火箭模型与 6DoF 动力学。
- `src/opt/`：SCvx 主循环与 SOCP 子问题定义。
- `src/sim/`：阶段时序、名义/故障仿真入口。
- `src/plots/`：统一的中文画图工具。
- `scripts/`：批量生成第 4 章全部图表/表格的脚本。

## 快速“烟雾测试”

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行名义 6DoF 仿真并生成基础曲线：
   ```bash
   python -m scripts.make_figs_ch4 --quick
   ```

quick 模式会：

- 调用 `src/sim/run_nominal.py`，用 6DoF 动力学模拟 KZ-1A 名义上升段；
- 在终端打印动压峰值、法向过载峰值、末端高度与速度，辅助判断模型是否物理合理；
- 将高度、速度、弹道倾角、动压、法向过载、推力、推力锥余量等曲线保存至 `outputs/figures/ch4/`。

## 完整复现实验

准备好 `configs/kz1a_params.yaml` 与 `configs/scvx.yaml`，即可生成第 4 章全部图表/表格：

```bash
python -m scripts.make_figs_ch4
python -m scripts.make_tables_ch4
```

流程：

1. 运行名义轨迹仿真，缓存 6DoF 状态与路径约束历史；
2. 根据 `sim/scenarios.py` 中的 Monte Carlo 集合生成故障轨迹，并调用 SCvx 重构；
3. 使用 `plots/plotting.py` 输出所有中文图题的图像到 `outputs/figures/ch4/`；
4. 汇总终端误差、SCvx 收敛效率、学习热启动前后对比等表格到 `outputs/tables/ch4/`。

命令行参数可控制样本规模、求解器配置与输出目录，详见脚本 `--help`。

## 进一步开发建议

1. 在 `dyn6dof` / `sim` 中继续细化气动力矩、姿态控制律，使名义轨迹与实测趋势更吻合。
2. 在 `opt/socp_problem.py` 与 `opt/scvx.py` 中完成信赖域、推力锥约束、虚拟控制惩罚等实现。
3. 在 `sim/run_fault.py` 中接入学习热启动策略与任务域判别逻辑，产生故障重构统计。
4. 扩展 `scripts/` 的命令行接口（Monte Carlo 批次数、热启动开关、安全落区设定等），满足论文第 4 章的所有实验需求。

如需查看论文原文或公式推导，请参考同目录下的 Word 文档。本工程旨在作为可复现的代码附件，配合一键命令生成全部结果。

