# 故障诊断与轨迹规划 前端UI运行说明

## 环境准备 (Windows + Conda)

打开 **命令提示符 (cmd)** 或 **Anaconda Prompt**，依次执行：

```cmd
cd /d "D:\python projects\ch5codeXv1.1"

call D:\Users\anaconda3\Scripts\activate.bat
conda activate chap3

python -m pip install --upgrade pip
python -m pip install streamlit plotly
```

或者使用 requirements 文件安装：

```cmd
python -m pip install -r frontend_requirements.txt
```

## 运行 UI

```cmd
streamlit run scripts/frontend_app.py
```

运行后会自动打开浏览器，访问 `http://localhost:8501`

## 命令行使用

也可以直接使用命令行运行流水线：

```cmd
# 单场景运行
python scripts/run_full_pipeline.py --scenario F1_thrust_deg15 --eta 0.5

# 指定故障注入时间
python scripts/run_full_pipeline.py --scenario F1_thrust_deg15 --eta 0.5 --t_fault 40.0

# 运行所有场景
python scripts/run_full_pipeline.py --all-scenarios

# 生成图表
python scripts/run_full_pipeline.py --scenario F1_thrust_deg15 --eta 0.5 --plot
```

## Python API 调用

```python
from scripts.run_full_pipeline import run_pipeline

res = run_pipeline("F1_thrust_deg15", 0.5, t_fault=40.0, make_plots=False)

print(res.keys())
# ['scenario', 'eta', 't_fault', 'diagnosis', 'mission_domain', 'trajectory', 'raw']

print(res["trajectory"].keys())
# ['t', 'downrange_km', 'altitude_km']

print(res["diagnosis"])
# {'fault_type': 'thrust_drop', 'eta_est': 0.5, 'confidence': 0.95}

print(res["mission_domain"])
# {'name': 'DEGRADED', 'h_target_km': 300.0, 'v_target_kms': 7.73}
```

## UI 使用说明

1. **左侧控件**：
   - 故障类型下拉框：选择 F1-F5 故障场景
   - η 滑条：设置故障严重度 (0-1)
   - 故障注入时间：设置故障发生时刻

2. **点击「运行仿真」按钮**

3. **查看结果**：
   - 任务域选择 (RETAIN/DEGRADED/SAFE_AREA)
   - 诊断结果与置信度
   - 终端状态 (高度、速度)
   - 诊断六子图 (ESO残差、样本熵、PWVD时频图等)
   - 3D轨迹图 (时间-地面行距-高度)
