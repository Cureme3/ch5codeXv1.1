# -*- coding: utf-8 -*-
"""
make_tables_ch3.py
生成第三章仿真实验相关的表格。
1. 表3-1：故障注入参数表
2. 表3-4：故障诊断性能表 (Precision, Recall, F1, Flight Recall)
3. 表3-5：算法复杂度与资源占用 (读取现有CSV)

用法：
  python scripts/make_tables_ch3.py --outdir exports/tables --summary exports/clf_phys6_eso/summary.json
"""

import os
import json
import argparse
import pandas as pd
import numpy as np

def make_table3_1_fault_params(outdir):
    """
    生成表3-1：故障注入参数表

    数据来源：从 table3_1_fault_params.csv 读取（如果存在），
    否则使用默认值。这样确保 CSV 更新后不会与脚本逻辑冲突。
    """
    print("[Table] Generating Table 3-1 (Fault Parameters)...")

    outpath = os.path.join(outdir, "table3_1_fault_params.csv")

    # 尝试读取现有 CSV 文件
    if os.path.exists(outpath):
        try:
            df = pd.read_csv(outpath, encoding="utf-8-sig")
            print(f"[Info] 从现有文件读取参数: {outpath}")
            # 验证必要列存在
            required_cols = ["Case ID", "Fault Type", "Parameters", "Description"]
            if all(col in df.columns for col in required_cols):
                print(f"[OK] 表3-1 已存在且格式正确，保留原文件")
                return df
            else:
                print(f"[Warn] 现有文件缺少必要列，使用默认数据重新生成")
        except Exception as e:
            print(f"[Warn] 读取现有文件失败 ({e})，使用默认数据")

    # 默认故障参数（与 simulate_ecifull 参数一致）
    # 注意: thrust_drop 全程生效，无时间触发
    data = [
        {"Case ID": "1", "Fault Type": "Nominal", "Parameters": "None", "Description": "名义工况，无故障"},
        {"Case ID": "2", "Fault Type": "Thrust Drop", "Parameters": "Ratio=0.15", "Description": "推力下降15%（全程生效）"},
        {"Case ID": "3", "Fault Type": "TVC Rate Limit", "Parameters": "Limit=0.35 deg/s", "Description": "TVC摆动速率限制 (0.35°/s)"},
        {"Case ID": "4", "Fault Type": "TVC Stuck", "Parameters": "t=30s, duration=20s", "Description": "TVC卡滞，发生于30s，持续20s"},
        {"Case ID": "5", "Fault Type": "Sensor Bias", "Parameters": "Bias=(1.5, 0, 0)", "Description": "加速度计x轴偏置 1.5 m/s²"},
        {"Case ID": "6", "Fault Type": "Event Delay", "Parameters": "S2_sep=2.0s, Fairing=2.0s", "Description": "关键事件延迟 (S2分离/整流罩)"},
    ]

    df = pd.DataFrame(data)
    df.to_csv(outpath, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {outpath}")
    return df

def make_table3_4_performance(summary_path, outdir):
    """
    生成表3-4：故障诊断性能表
    数据来源：train_eval_classifier.py 生成的 summary.json
    """
    print(f"[Table] Generating Table 3-4 (Performance) from {summary_path}...")
    
    if not os.path.exists(summary_path):
        print(f"[Error] Summary file not found: {summary_path}")
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    # 提取混淆矩阵 (Window Level)
    cm = np.array(summary["confusion"])
    # 提取飞行级召回率
    flight_recall = summary["diagnostics"].get("flight_recall_per_class", [])
    
    classes = ["Nominal", "Thrust Drop", "TVC Rate", "TVC Stuck", "Sensor Bias", "Event Delay"]
    
    # 计算 Window-Level 指标
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
    # 构建表格数据
    rows = []
    for i, cls_name in enumerate(classes):
        fr = flight_recall[i] if i < len(flight_recall) else 0.0
        rows.append({
            "Class": cls_name,
            "Precision (Win)": f"{precisions[i]:.4f}",
            "Recall (Win)": f"{recalls[i]:.4f}",
            "F1-Score (Win)": f"{f1s[i]:.4f}",
            "Recall (Flight)": f"{fr:.4f}"
        })
        
    # 添加平均值行
    avg_p = np.mean(precisions)
    avg_r = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    avg_fr = np.mean(flight_recall) if flight_recall else 0.0
    
    rows.append({
        "Class": "Average",
        "Precision (Win)": f"{avg_p:.4f}",
        "Recall (Win)": f"{avg_r:.4f}",
        "F1-Score (Win)": f"{avg_f1:.4f}",
        "Recall (Flight)": f"{avg_fr:.4f}"
    })
    
    df = pd.DataFrame(rows)
    outpath = os.path.join(outdir, "table3_4_performance.csv")
    df.to_csv(outpath, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {outpath}")
    return df

def make_table3_5_complexity(outdir):
    """
    生成表3-5：算法复杂度与资源占用
    目前主要是读取静态数据，或者生成默认模板
    """
    print("[Table] Generating Table 3-5 (Complexity)...")
    
    # 尝试读取现有文件，如果存在则保留，否则创建默认
    existing_path = os.path.join(outdir, "table3_5_complexity_resources.csv")
    if os.path.exists(existing_path):
        print(f"[Info] Found existing table: {existing_path}, keeping it.")
        # 可以在这里做格式化，但暂不修改内容
        return
        
    data = [
        {"Module": "STFT Feature", "Ops/Cycle": "448", "Cycle (ms)": "50", "Freq (Hz)": "20", "Note": "可裁剪"},
        {"Module": "PWVD Feature", "Ops/Cycle": "322", "Cycle (ms)": "50", "Freq (Hz)": "20", "Note": "可降采样"},
        {"Module": "RBF Inference", "Ops/Cycle": "192", "Cycle (ms)": "50", "Freq (Hz)": "20", "Note": "可合并"},
        {"Module": "ESO Gen", "Ops/Cycle": "2000", "Cycle (ms)": "10", "Freq (Hz)": "100", "Note": "硬实时"},
    ]
    df = pd.DataFrame(data)
    df.to_csv(existing_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Created default {existing_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="exports/ch3_tables")
    parser.add_argument("--summary", type=str, default="exports/clf_phys6_eso/summary.json")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    make_table3_1_fault_params(args.outdir)
    # make_table3_4_performance(args.summary, args.outdir) # Moved to train_eval_classifier.py
    make_table3_5_complexity(args.outdir)
    
    print("[Done] All tables generated.")

if __name__ == "__main__":
    main()
