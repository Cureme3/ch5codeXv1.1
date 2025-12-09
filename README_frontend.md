# 故障诊断与轨迹规划 前端可视化界面

## 前端可视化界面使用说明

### 1. 启动环境与依赖

```bat
cd /d "D:\python projects\ch5codeXv1.1"

call D:\Users\anaconda3\Scripts\activate.bat
conda activate chap3

python -m pip install -r frontend_requirements.txt
```

### 2. 启动前端应用

```bat
cd /d "D:\python projects\ch5codeXv1.1"
streamlit run scripts/frontend_app.py
```

### 3. 使用界面

打开浏览器中自动弹出的地址（一般为 http://localhost:8501），即可看到：

- **左侧**：故障类型 / 故障强度 η / 故障注入时间 t_fault 参数面板
- **中间**：文本摘要，包括真实故障、诊断结果、任务域、SCvx 自适应权重等
- **右侧上方**：故障诊断六子图（残差、PWVD、特征融合等）
- **右侧下方**：时间–行距–高度 3D 轨迹图（可旋转）

在不同故障类型和 η 下，点击「运行全链路仿真」按钮即可一键完成故障注入–诊断–轨迹规划–任务级处置的全流程仿真，并实时查看可视化结果。
