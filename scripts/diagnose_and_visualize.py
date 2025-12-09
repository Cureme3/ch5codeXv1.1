#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""故障诊断全链路可视化：注入故障后生成诊断对比图。"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CH4_ROOT = PROJECT_ROOT / "ch4codexv1.1"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ch3codev1.1"))
sys.path.insert(0, str(CH4_ROOT / "src"))
sys.path.insert(0, str(CH4_ROOT))

from plots.plotting import setup_matplotlib, save_figure, DEFAULT_COLORS, LINE_WIDTH
from diagnosis.eso import run_eso
from diagnosis.features import extract_features_from_residual, pwvd, sample_entropy
from src.sim.run_fault import run_fault_scenario
from src.sim.run_nominal import simulate_nominal

setup_matplotlib()
# 强制使用 Windows 中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "diagnosis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_NOM = DEFAULT_COLORS["nominal"]
COLOR_FAULT = DEFAULT_COLORS["fault"]


def get_residual(states, dt=1.0):
    """从状态轨迹提取ESO残差。"""
    v = states[:, 3:6]
    ax = np.diff(v[:, 0])
    az = np.diff(v[:, 2])
    axz = np.column_stack([ax, az])
    residuals, _, _ = run_eso(axz, dt=dt)
    return np.linalg.norm(residuals, axis=1)


# ============ 子图绘制函数 ============

def _plot_eso_residuals(time_arr, nom_res, fault_res) -> Figure:
    """绘制ESO残差对比图。"""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(time_arr, nom_res, color=COLOR_NOM, linewidth=LINE_WIDTH, label='名义', alpha=0.7)
    ax.plot(time_arr, fault_res, color=COLOR_FAULT, linewidth=LINE_WIDTH, label='故障')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('残差幅值')
    ax.set_title('ESO残差对比')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_sample_entropy(nom_t, nom_se, fault_t, fault_se) -> Figure:
    """绘制样本熵对比图。"""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(nom_t, nom_se, color=COLOR_NOM, linewidth=LINE_WIDTH, label='名义', alpha=0.7)
    ax.plot(fault_t, fault_se, color=COLOR_FAULT, linewidth=LINE_WIDTH, label='故障')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('样本熵')
    ax.set_title('样本熵对比')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_pwvd(spec_nom, spec_fault, nom_res_len, fault_res_len, dt, vmax) -> Figure:
    """绘制PWVD时频图。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharex=False, sharey=True)

    im1 = ax1.imshow(spec_nom, aspect='auto', origin='lower',
                     extent=[0, nom_res_len*dt, 0, 0.5/dt], cmap='jet', vmin=0, vmax=vmax)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('频率 (Hz)')
    ax1.set_title('PWVD时频图: 名义')
    plt.colorbar(im1, ax=ax1, shrink=0.9, pad=0.02)

    im2 = ax2.imshow(spec_fault, aspect='auto', origin='lower',
                     extent=[0, fault_res_len*dt, 0, 0.5/dt], cmap='jet', vmin=0, vmax=vmax)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('频率 (Hz)')
    ax2.set_title('PWVD时频图: 故障')
    plt.colorbar(im2, ax=ax2, shrink=0.9, pad=0.02)

    fig.tight_layout()
    return fig


def _plot_3d_energy(nom_feats, fault_feats) -> Figure:
    """绘制三维能量坐标图。"""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')

    E_nom = nom_feats[:3] / (np.sum(nom_feats[:3]) + 1e-12)
    E_fault = fault_feats[:3] / (np.sum(fault_feats[:3]) + 1e-12)

    ax.scatter([E_nom[0]], [E_nom[1]], [E_nom[2]], c=COLOR_NOM, s=200, marker='o', label='名义', edgecolors='k')
    ax.scatter([E_fault[0]], [E_fault[1]], [E_fault[2]], c=COLOR_FAULT, s=200, marker='^', label='故障', edgecolors='k')
    ax.plot([E_nom[0], E_fault[0]], [E_nom[1], E_fault[1]], [E_nom[2], E_fault[2]], 'k--', alpha=0.5)

    ax.set_xlabel('低频', fontsize=9)
    ax.set_ylabel('中频', fontsize=9)
    ax.set_zlabel('高频', fontsize=9)
    ax.set_title('三维能量坐标')
    ax.legend(loc='upper left')
    ax.view_init(elev=25, azim=45)

    fig.tight_layout()
    return fig


def _plot_features_bar(nom_feats, fault_feats, diagnosed_eta) -> Figure:
    """绘制多源特征融合柱状图。"""
    fig, ax = plt.subplots(figsize=(6, 3))

    labels = ['低频', '中频', '高频', '时频熵', '样本熵', 'DC']
    x = np.arange(len(labels))
    width = 0.35

    # 分组归一化
    nom_norm = np.zeros_like(nom_feats)
    fault_norm = np.zeros_like(fault_feats)
    e_max = max(np.max(np.abs(nom_feats[:3])), np.max(np.abs(fault_feats[:3]))) + 1e-12
    nom_norm[:3] = nom_feats[:3] / e_max
    fault_norm[:3] = fault_feats[:3] / e_max
    for i in range(3, 6):
        m = max(abs(nom_feats[i]), abs(fault_feats[i])) + 1e-12
        nom_norm[i] = nom_feats[i] / m
        fault_norm[i] = fault_feats[i] / m

    ax.bar(x - width/2, nom_norm, width, label='名义', color=COLOR_NOM, alpha=0.8, edgecolor='k')
    ax.bar(x + width/2, fault_norm, width, label='故障', color=COLOR_FAULT, alpha=0.8, edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('归一化特征值')
    ax.set_title(f'多源特征融合 | 诊断$\\eta$={diagnosed_eta:.2f}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.1, 1.15)

    fig.tight_layout()
    return fig


def _plot_fusion_indicator(fusion_score, diagnosed_eta) -> Figure:
    """绘制融合指标图。"""
    fig, ax = plt.subplots(figsize=(6, 3))

    # 简单的融合指标可视化
    categories = ['融合评分', '诊断η']
    values = [fusion_score, diagnosed_eta]
    colors = ['steelblue', 'coral']

    bars = ax.bar(categories, values, color=colors, edgecolor='k', alpha=0.8)
    ax.set_ylabel('值')
    ax.set_title('融合诊断指标')
    ax.set_ylim(0, max(values) * 1.2 + 0.1)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    return fig


# ============ 核心诊断数据计算 ============

def _compute_diagnosis_data(fault_id: str, eta: float, dt: float = 1.0):
    """计算诊断所需的所有中间数据。"""
    # 1. 仿真
    nom_sim = simulate_nominal(dt=dt)
    nom_states = nom_sim.states
    fault_sim = run_fault_scenario(fault_id, eta=eta, dt=dt)
    fault_states = fault_sim.states

    # 2. 计算残差
    min_len = min(len(nom_states), len(fault_states)) - 1
    nom_res = get_residual(nom_states[:min_len+1], dt)[:min_len]
    fault_res = get_residual(fault_states[:min_len+1], dt)[:min_len]
    time_arr = np.arange(min_len) * dt

    # 3. 特征提取
    nom_feats = extract_features_from_residual(nom_res, dt=dt, spec_method='pwvd')
    fault_feats = extract_features_from_residual(fault_res, dt=dt, spec_method='pwvd')

    # 4. 样本熵滑窗计算
    win_size, step = 30, 5
    def sliding_sampen(res):
        vals, ts = [], []
        for i in range(0, len(res) - win_size, step):
            seg = res[i:i+win_size]
            se = sample_entropy(seg, m=2, r=0.2*np.std(seg)+1e-12)
            vals.append(se)
            ts.append((i + win_size//2) * dt)
        return np.array(ts), np.array(vals)
    nom_t, nom_se = sliding_sampen(nom_res)
    fault_t, fault_se = sliding_sampen(fault_res)

    # 5. PWVD时频图
    spec_nom = pwvd(nom_res, win_len=64)
    spec_fault = pwvd(fault_res, win_len=64)
    vmax = max(np.percentile(spec_nom, 95), np.percentile(spec_fault, 95))

    # 6. 融合评分
    feat_diff = np.abs(fault_feats - nom_feats)
    fusion_score = np.sum(feat_diff / (np.abs(nom_feats) + 1e-6)) / len(feat_diff)
    diagnosed_eta = min(1.0, fusion_score / 5.0)

    return {
        'time_arr': time_arr,
        'nom_res': nom_res,
        'fault_res': fault_res,
        'nom_feats': nom_feats,
        'fault_feats': fault_feats,
        'nom_t': nom_t,
        'nom_se': nom_se,
        'fault_t': fault_t,
        'fault_se': fault_se,
        'spec_nom': spec_nom,
        'spec_fault': spec_fault,
        'vmax': vmax,
        'fusion_score': fusion_score,
        'diagnosed_eta': diagnosed_eta,
        'dt': dt,
    }


# ============ 主接口函数 ============

def run_diagnosis(
    fault: str,
    eta: float,
    show: bool = False,
    save: bool = False,
) -> plt.Figure:
    """运行故障诊断并返回可视化Figure。

    Parameters
    ----------
    fault : str
        故障场景ID，如 'F1_thrust_deg15'
    eta : float
        故障严重度 [0, 1]
    show : bool
        是否显示图表
    save : bool
        是否保存到文件

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, _ = _create_diagnosis_figure_impl(fault, eta, dt=1.0, save=save, verbose=False)
    if show:
        plt.show()
    return fig


def run_diagnosis_components(
    fault: str,
    eta: float,
    show: bool = False,
    save: bool = False,
) -> Dict[str, Figure]:
    """
    运行一次诊断，并返回每个模块对应的 Figure 字典。

    Parameters
    ----------
    fault : str
        故障场景ID
    eta : float
        故障严重度 [0, 1]
    show : bool
        是否显示图表
    save : bool
        是否保存到文件

    Returns
    -------
    dict
        {
            "overview": fig_overview,         # 3x2 总览图
            "eso_residuals": fig_res,
            "sample_entropy": fig_se,
            "pwvd": fig_pwvd,
            "energy_3d": fig_energy,
            "features_bar": fig_feat,
            "fusion": fig_fusion,
        }
    """
    # 1) 计算诊断数据
    data = _compute_diagnosis_data(fault, eta, dt=1.0)

    # 2) 生成各子图
    fig_res = _plot_eso_residuals(data['time_arr'], data['nom_res'], data['fault_res'])
    fig_se = _plot_sample_entropy(data['nom_t'], data['nom_se'], data['fault_t'], data['fault_se'])
    fig_pwvd = _plot_pwvd(data['spec_nom'], data['spec_fault'],
                          len(data['nom_res']), len(data['fault_res']),
                          data['dt'], data['vmax'])
    fig_energy = _plot_3d_energy(data['nom_feats'], data['fault_feats'])
    fig_feat = _plot_features_bar(data['nom_feats'], data['fault_feats'], data['diagnosed_eta'])
    fig_fusion = _plot_fusion_indicator(data['fusion_score'], data['diagnosed_eta'])

    # 3) 生成总览图
    fig_overview = run_diagnosis(fault, eta, show=False, save=save)

    figs = {
        "overview": fig_overview,
        "eso_residuals": fig_res,
        "sample_entropy": fig_se,
        "pwvd": fig_pwvd,
        "energy_3d": fig_energy,
        "features_bar": fig_feat,
        "fusion": fig_fusion,
    }

    if show:
        for f in figs.values():
            plt.figure(f.number)
            plt.show()

    return figs


def create_diagnosis_figure(fault_id: str, eta: float, dt: float = 1.0, save: bool = False):
    """生成诊断可视化Figure对象，供UI直接展示。

    Parameters
    ----------
    fault_id : str
        故障场景ID
    eta : float
        故障严重度 [0, 1]
    dt : float
        仿真步长
    save : bool
        是否保存到文件

    Returns
    -------
    tuple : (fig, result_dict)
        fig: matplotlib.figure.Figure
        result_dict: 诊断结果字典
    """
    return _create_diagnosis_figure_impl(fault_id, eta, dt, save, verbose=False)


def diagnose_and_visualize(fault_id: str, eta: float, dt: float = 1.0):
    """
    注入故障并生成诊断可视化对比图。

    Parameters
    ----------
    fault_id : str
        故障场景ID，如 'F1_thrust_deg15', 'F2_tvc_rate4' 等
    eta : float
        故障严重度 [0, 1]
    dt : float
        仿真步长

    Returns
    -------
    dict : 诊断结果
    """
    _, result = _create_diagnosis_figure_impl(fault_id, eta, dt, save=True, verbose=True)
    return result


def _create_diagnosis_figure_impl(fault_id: str, eta: float, dt: float, save: bool, verbose: bool):
    def _print(msg):
        if verbose:
            print(msg)

    _print(f"\n{'='*60}")
    _print(f"故障诊断可视化: {fault_id}, eta={eta}")
    _print('='*60)

    # 1. 仿真
    _print("[1/6] 名义轨迹仿真...")
    nom_sim = simulate_nominal(dt=dt)
    nom_states = nom_sim.states

    _print(f"[2/6] 故障轨迹仿真: {fault_id}, eta={eta}...")
    fault_sim = run_fault_scenario(fault_id, eta=eta, dt=dt)
    fault_states = fault_sim.states

    # 2. 计算残差
    _print("[3/6] ESO残差估计...")
    min_len = min(len(nom_states), len(fault_states)) - 1
    nom_res = get_residual(nom_states[:min_len+1], dt)[:min_len]
    fault_res = get_residual(fault_states[:min_len+1], dt)[:min_len]
    time_arr = np.arange(min_len) * dt

    # 3. 特征提取
    _print("[4/6] PWVD时频特征提取...")
    nom_feats = extract_features_from_residual(nom_res, dt=dt, spec_method='pwvd')
    fault_feats = extract_features_from_residual(fault_res, dt=dt, spec_method='pwvd')

    # 4. 多源融合评分
    _print("[5/6] 多源融合评分...")
    # 简化的融合评分：基于特征差异
    feat_diff = np.abs(fault_feats - nom_feats)
    fusion_score = np.sum(feat_diff / (np.abs(nom_feats) + 1e-6)) / len(feat_diff)
    diagnosed_eta = min(1.0, fusion_score / 5.0)  # 归一化到[0,1]

    # 5. 生成可视化
    _print("[6/6] 生成可视化图表...")

    # 使用 GridSpec 精细控制布局
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.3)
    fig.suptitle(f'故障诊断: {fault_id}, $\\eta$={eta:.2f}', fontsize=14, fontweight='bold', y=0.98)

    # --- 第一行: 残差 + 样本熵 ---
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(time_arr, nom_res, color=COLOR_NOM, linewidth=LINE_WIDTH, label='名义', alpha=0.7)
    ax1.plot(time_arr, fault_res, color=COLOR_FAULT, linewidth=LINE_WIDTH, label='故障')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('残差幅值')
    ax1.set_title('(a) ESO残差对比')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2:])
    win_size, step = 30, 5
    def sliding_sampen(res):
        vals, ts = [], []
        for i in range(0, len(res) - win_size, step):
            seg = res[i:i+win_size]
            se = sample_entropy(seg, m=2, r=0.2*np.std(seg)+1e-12)
            vals.append(se)
            ts.append((i + win_size//2) * dt)
        return np.array(ts), np.array(vals)
    nom_t, nom_se = sliding_sampen(nom_res)
    fault_t, fault_se = sliding_sampen(fault_res)
    ax2.plot(nom_t, nom_se, color=COLOR_NOM, linewidth=LINE_WIDTH, label='名义', alpha=0.7)
    ax2.plot(fault_t, fault_se, color=COLOR_FAULT, linewidth=LINE_WIDTH, label='故障')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('样本熵')
    ax2.set_title('(b) 样本熵对比')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- 第二行: PWVD时频图 ---
    spec_nom = pwvd(nom_res, win_len=64)
    spec_fault = pwvd(fault_res, win_len=64)
    vmax = max(np.percentile(spec_nom, 95), np.percentile(spec_fault, 95))

    ax3 = fig.add_subplot(gs[1, :2])
    im3 = ax3.imshow(spec_nom, aspect='auto', origin='lower',
                     extent=[0, len(nom_res)*dt, 0, 0.5/dt], cmap='jet', vmin=0, vmax=vmax)
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('频率 (Hz)')
    ax3.set_title('(c) PWVD时频图: 名义')
    plt.colorbar(im3, ax=ax3, shrink=0.9, pad=0.02)

    ax4 = fig.add_subplot(gs[1, 2:])
    im4 = ax4.imshow(spec_fault, aspect='auto', origin='lower',
                     extent=[0, len(fault_res)*dt, 0, 0.5/dt], cmap='jet', vmin=0, vmax=vmax)
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('频率 (Hz)')
    ax4.set_title('(d) PWVD时频图: 故障')
    plt.colorbar(im4, ax=ax4, shrink=0.9, pad=0.02)

    # --- 第三行: 三维能量 + 特征柱状图 ---
    ax5 = fig.add_subplot(gs[2, :2], projection='3d')
    E_nom = nom_feats[:3] / (np.sum(nom_feats[:3]) + 1e-12)
    E_fault = fault_feats[:3] / (np.sum(fault_feats[:3]) + 1e-12)
    ax5.scatter([E_nom[0]], [E_nom[1]], [E_nom[2]], c=COLOR_NOM, s=200, marker='o', label='名义', edgecolors='k')
    ax5.scatter([E_fault[0]], [E_fault[1]], [E_fault[2]], c=COLOR_FAULT, s=200, marker='^', label='故障', edgecolors='k')
    # 连线
    ax5.plot([E_nom[0], E_fault[0]], [E_nom[1], E_fault[1]], [E_nom[2], E_fault[2]], 'k--', alpha=0.5)
    ax5.set_xlabel('低频', fontsize=9)
    ax5.set_ylabel('中频', fontsize=9)
    ax5.set_zlabel('高频', fontsize=9)
    ax5.set_title('(e) 三维能量坐标')
    ax5.legend(loc='upper left')
    ax5.view_init(elev=25, azim=45)

    ax6 = fig.add_subplot(gs[2, 2:])
    labels = ['低频', '中频', '高频', '时频熵', '样本熵', 'DC']
    x = np.arange(len(labels))
    width = 0.35
    # 分组归一化：能量类(0-2)和熵类(3-5)分别归一化
    nom_norm = np.zeros_like(nom_feats)
    fault_norm = np.zeros_like(fault_feats)
    # 能量归一化
    e_max = max(np.max(np.abs(nom_feats[:3])), np.max(np.abs(fault_feats[:3]))) + 1e-12
    nom_norm[:3] = nom_feats[:3] / e_max
    fault_norm[:3] = fault_feats[:3] / e_max
    # 熵和DC归一化
    for i in range(3, 6):
        m = max(abs(nom_feats[i]), abs(fault_feats[i])) + 1e-12
        nom_norm[i] = nom_feats[i] / m
        fault_norm[i] = fault_feats[i] / m
    bars1 = ax6.bar(x - width/2, nom_norm, width, label='名义', color=COLOR_NOM, alpha=0.8, edgecolor='k')
    bars2 = ax6.bar(x + width/2, fault_norm, width, label='故障', color=COLOR_FAULT, alpha=0.8, edgecolor='k')
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels)
    ax6.set_ylabel('归一化特征值')
    ax6.set_title(f'(f) 多源特征融合 | 诊断$\\eta$={diagnosed_eta:.2f}')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(-0.1, 1.15)

    # 保存
    out_path = OUTPUT_DIR / f"diag_{fault_id}_eta{eta:.1f}.png"
    if save:
        save_figure(fig, out_path)

    _print(f"\n诊断结果:")
    _print(f"  真实故障: {fault_id}, eta={eta:.2f}")
    _print(f"  诊断eta: {diagnosed_eta:.2f}")
    _print(f"  融合评分: {fusion_score:.2f}")
    if save:
        _print(f"  图表保存: {out_path}")

    result = {
        'fault_id': fault_id,
        'true_eta': eta,
        'diagnosed_eta': diagnosed_eta,
        'fusion_score': fusion_score,
        'figure_path': str(out_path) if save else None,
    }

    return fig, result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='故障诊断可视化')
    parser.add_argument('--fault', type=str, default='F1_thrust_deg15', help='故障ID')
    parser.add_argument('--eta', type=float, default=0.5, help='故障严重度')
    parser.add_argument('--components', action='store_true', help='生成各子图')
    args = parser.parse_args()

    if args.components:
        figs = run_diagnosis_components(args.fault, args.eta, show=True, save=True)
        print(f"生成了 {len(figs)} 张图")
    else:
        run_diagnosis(args.fault, args.eta, show=True, save=True)
