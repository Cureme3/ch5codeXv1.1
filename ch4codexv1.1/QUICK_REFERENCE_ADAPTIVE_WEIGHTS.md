# 自适应权重集成 - 快速参考

## 修改摘要

### 文件变更
1. **opt/scvx.py** - ✅ 无需修改 (已有 `set_penalty_weights`)
2. **opt/socp_problem.py** - ✅ 无需修改 (已用 `self.weights`)
3. **src/sim/run_fault.py** - ✅ 添加权重到诊断信息 (3 行)
4. **scripts/test_adaptive_weights.py** - ✅ 新建测试脚本

### 核心修改

#### run_fault.py (Line 308-310)
```python
# Add penalty weights to diagnostics if using adaptive weights
if use_adaptive_penalties and eta_value is not None:
    final_diagnostics["penalty_weights"] = dataclasses.asdict(adaptive_weights)
```

## 验证结果

### 权重变化数值

**eta 影响** (Terminal weight):
- eta=0.2: 39.5
- eta=0.5: 33.7
- eta=0.8: 29.4
- 差异: 10.1

**任务域影响** (eta=0.5, Terminal weight):
- RETAIN: 33.7
- DEGRADED: 16.8
- SAFE_AREA: 6.7
- 差异: 27.0

**组合效果** (q_slack):
```
          RETAIN  DEGRADED  SAFE_AREA
eta=0.2    13.5     26.9      40.4
eta=0.5    22.2     44.4      66.7
eta=0.8    33.7     67.3     101.0
```

## 测试命令

### 1. 快速权重测试
```bash
cd ch4codexv1.1
python scripts/test_adaptive_weights.py
```

### 2. 单个故障测试
```bash
python -c "
from src.sim.scenarios import get_scenario
from src.sim.run_fault import run_fault_scenario, plan_recovery_segment_scvx
from src.sim.run_nominal import simulate_full_mission

scenario = get_scenario('F1_thrust_deg15')
fault_sim = run_fault_scenario('F1_thrust_deg15', dt=1.0)
nominal = simulate_full_mission(dt=1.0)

result = plan_recovery_segment_scvx(
    scenario=scenario,
    fault_sim=fault_sim,
    nominal=nominal,
    eta=0.5,
    use_adaptive_penalties=True,
    enable_domain_escalation=True,
)

print('Domain:', result.mission_domain)
print('Attempts:', result.diagnostics['domain_attempts'])
if 'penalty_weights' in result.diagnostics:
    print('Terminal weight:', result.diagnostics['penalty_weights']['terminal_state_dev'])
"
```

### 3. 完整域评估
```bash
python scripts/eval_ch4_mission_domains.py --enable-escalation
```

## 数据流图

```
eta=0.5, DEGRADED domain
  ↓
compute_adaptive_penalties(0.5)
  → base_weights.terminal_state_dev = 33.7
  → base_weights.q_slack = 22.2
  ↓
default_domain_config(DEGRADED)
  → terminal_weight_scale = 0.5
  → slack_weight_scale = 2.0
  ↓
adaptive_weights = base × domain_scales
  → terminal_state_dev = 33.7 × 0.5 = 16.8
  → q_slack = 22.2 × 2.0 = 44.4
  ↓
planner.set_penalty_weights(adaptive_weights)
  → self.weights = adaptive_weights
  → self.builder.weights = adaptive_weights
  ↓
planner.iterate(bundle, max_iters=8)
  → SOCPProblemBuilder.build_problem(...)
     → objective = ... + 16.8 * terminal_term + 44.4 * q_slack + ...
```

## 预期输出

### test_adaptive_weights.py
```
[SUCCESS] 所有权重都随 eta 变化!
[SUCCESS] 任务域缩放也在工作!
```

### eval_ch4_mission_domains.py --enable-escalation
```csv
fault_id,eta,mission_domain,domain_attempts,final_feas_violation
F1,0.2,RETAIN,1,0.0001
F1,0.5,RETAIN,1,0.0005
F1,0.8,DEGRADED,2,0.0008
F2,0.8,SAFE_AREA,3,0.0012
```

## 关键验证点

- ✅ `set_penalty_weights()` 存在并更新 planner 和 builder
- ✅ `SOCPProblemBuilder` 使用 `self.weights` 构建目标
- ✅ `plan_recovery_segment_scvx` 在 FSM 循环中调用 `set_penalty_weights`
- ✅ 权重记录到 `final_diagnostics["penalty_weights"]`
- ✅ 数值测试证明权重确实在变化
- ✅ 向后兼容性保持

## 故障排查

### 问题: 权重没有变化
检查:
1. `use_adaptive_penalties=True` 是否传入?
2. `eta_value` 是否为 None?
3. 是否在 FSM 循环内调用?

### 问题: 诊断信息中没有 penalty_weights
原因: `use_adaptive_penalties=False` 或 `eta_value is None`
解决: 显式传入 `eta` 参数

### 问题: domain_attempts 总是 1
原因: `enable_domain_escalation=False`
解决: 传入 `enable_domain_escalation=True`
