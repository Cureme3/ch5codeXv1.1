# -*- coding: utf-8 -*-
"""
make_fig_all.py
合并后的画图脚本，用于生成第三章相关的所有图片。
包含：
1. 图3-2：时频图 (STFT/PWVD)
2. 图3-3：六类故障残差波形
3. 图3-4：时频特征散点图 (2D)
4. 图3-7：残差与自适应阈值
5. 图3-8：时频特征散点图 (3D)

用法：
  python scripts/make_fig_all.py --outdir exports/ch3_figures --all
  python scripts/make_fig_all.py --fig 3-2
"""

from __future__ import annotations
import os, sys, argparse, time, json, math, inspect, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add path to find sim and diagnosis modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D

from sim.eci_full import simulate_ecifull
from diagnosis.eso import run_eso
from diagnosis.features import stft as stft_feat, sample_entropy

# =============================================================================
# 1. 出版级配色 (Science/Nature 风格)
# =============================================================================
DEFAULT_COLORS = {
    "nominal": "#00468B",  # 深蓝 (Navy)
    "fault": "#ED0000",    # 鲜红 (Red)
    "replan": "#42B540",   # 鲜绿 (Green)
    "ref": "#925E9F",      # 紫色
    "gray": "#ADB6B6",     # 灰色
}

LINE_WIDTH = 2.0
FIG_SIZE = (6, 4.5)
LEGEND_FONTSIZE = 12

# =============================================================================
# 2. 绘图风格与字体设置 (中文期刊出版标准)
# =============================================================================
def setup_style(dpi=300):
    """配置 Matplotlib 以符合中文期刊要求 (宋体 + Times New Roman)。"""
    import platform
    system_name = platform.system()

    if system_name == "Windows":
        font_serif = ["SimSun", "Times New Roman", "SimHei"]
        font_sans = ["SimHei", "Arial", "Microsoft YaHei"]
    else:
        font_serif = ["Noto Serif CJK SC", "STSong", "Times New Roman"]
        font_sans = ["Noto Sans CJK SC", "STHeiti", "Arial"]

    config = {
        # --- 字体核心设置 ---
        "font.family": "serif",
        "font.serif": font_serif + list(plt.rcParams["font.serif"]),
        "font.sans-serif": font_sans + list(plt.rcParams["font.sans-serif"]),

        # --- 数学公式设置 ---
        "mathtext.fontset": "stix",

        # --- 布局与线条 ---
        "axes.unicode_minus": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "lines.linewidth": LINE_WIDTH,
        "axes.linewidth": 1.5,

        # --- 字号设置 ---
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": LEGEND_FONTSIZE,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "legend.fancybox": False,

        # --- 输出设置 ---
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
    plt.rcParams.update(config)
    print(f"Plotting Style: Configured for {system_name} with serif fonts (宋体 + Times New Roman)")

# =============================================================================
# 2. 信号处理与计算工具 (Visualization Helpers)
# =============================================================================
def moving_mean_std(x, win=201):
    x = np.asarray(x, dtype=float)
    win = int(max(5, win//2*2+1))
    w = np.ones(win)/win
    mu = np.convolve(x, w, mode="same")
    ex2= np.convolve(x*x, w, mode="same")
    var= np.maximum(ex2 - mu*mu, 1e-12)
    return mu, np.sqrt(var)

def db_scale(S, eps=1e-12, dr_db=45.0):
    S = np.asarray(S, dtype=float)
    ref = np.percentile(S, 95.0) + eps
    SdB = 10.0*np.log10(np.maximum(S, eps)/ref)
    vmax = float(np.nanmax(SdB)); vmin = vmax - float(dr_db)
    return np.clip(SdB, vmin, vmax), (vmin, vmax)

def _gaussian_kernel1d(sigma: float, radius: int):
    if sigma <= 0 or radius <= 0: return np.array([1.0])
    x = np.arange(-radius, radius+1, dtype=float)
    k = np.exp(-(x*x)/(2*sigma*sigma)); return k/k.sum()

def tfr_smooth(P, sigma_t=0.0, sigma_f=0.0, rad_t=3, rad_f=3):
    Q = np.asarray(P, dtype=float)
    if sigma_t>0:
        kt=_gaussian_kernel1d(sigma_t,int(rad_t)); pad=len(kt)//2
        Q=np.pad(Q,((0,0),(pad,pad)),"reflect")
        Q=np.apply_along_axis(lambda m: np.convolve(m,kt,mode="valid"),1,Q)
    if sigma_f>0:
        kf=_gaussian_kernel1d(sigma_f,int(rad_f)); pad=len(kf)//2
        Q=np.pad(Q,((pad,pad),(0,0)),"reflect")
        Q=np.apply_along_axis(lambda m: np.convolve(m,kf,mode="valid"),0,Q)
    return Q

def stft_np(x, fs, win_sec=1.0, hop_sec=0.20, window="hann", center=True, nfft_mult=4):
    x = np.asarray(x, dtype=float)
    Nw = int(round(win_sec*fs)); Nh = int(round(hop_sec*fs))
    Nw = max(8, (Nw//2)*2); Nh = max(1, Nh)
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(Nw)/Nw) if window=="hann" else np.ones(Nw)
    xpad = np.pad(x, (Nw//2, Nw//2), mode="reflect") if center else x
    Nfft = int(max(Nw, nfft_mult*Nw))
    frames, tgrid = [], []
    for start in range(0, len(xpad)-Nw+1, Nh):
        seg = xpad[start:start+Nw]*w
        frames.append(np.fft.rfft(seg, n=Nfft))
        tgrid.append((start + Nw//2 - (Nw//2 if center else 0))/fs)
    X = np.stack(frames, axis=1)
    S = (np.abs(X)**2)/(np.sum(w*w))
    f = np.fft.rfftfreq(Nfft, d=1.0/fs)
    t = np.asarray(tgrid, dtype=float)
    return f, t, S

def _analytic_signal(x):
    x = np.asarray(x, dtype=float); N = x.size
    X = np.fft.fft(x); H = np.zeros(N)
    if N % 2 == 0: H[0]=1; H[N//2]=1; H[1:N//2]=2
    else: H[0]=1; H[1:(N+1)//2]=2
    return np.fft.ifft(X*H)

def pwvd_np(x, fs, half_lag=64, t_subsample=2):
    x = np.asarray(x, dtype=float); xa = _analytic_signal(x); N = xa.size
    L = int(max(8, min(int(half_lag), (N - 3)//2)))
    if L <= 0: raise ValueError("signal too short for PWVD")
    tstep = max(1, int(t_subsample)); t_idx = np.arange(L, N-L, tstep)
    while t_idx.size == 0 and L > 8: L=max(8,L//2); t_idx=np.arange(L,N-L,tstep)
    if t_idx.size == 0: raise ValueError("signal too short for PWVD (no valid t_idx)")
    M = 2*L + 1
    H = 0.5 - 0.5*np.cos(2*np.pi*np.arange(-L, L+1)/(2*L+1))
    def _reflect_indices(idx, N_):
        idx = np.asarray(idx, dtype=int)
        if N_ == 1: return np.zeros_like(idx)
        period = 2*(N_-1); k = np.mod(idx, period)
        return np.where(k <= N_-1, k, period - k)
    cols=[]
    for ti in t_idx:
        base = np.arange(ti-L, ti+L+1); ids = _reflect_indices(base, N)
        seg = xa[ids]; z_plus, z_minus = seg, seg.conj()[::-1]
        ac = H * z_plus * z_minus
        Spec = np.fft.fftshift(np.fft.fft(ac, n=4*M))
        cols.append(np.abs(Spec)**2)
    P = np.stack(cols, axis=1)
    F_full = np.fft.fftshift(np.fft.fftfreq(P.shape[0], d=1.0/fs))
    sel = F_full >= 0
    return F_full[sel], t_idx/fs, P[sel,:]

# =============================================================================
# 3. 数据生成与仿真接口
# =============================================================================
def _adapt_fault_kwargs(simulate_ecifull, mods: dict):
    sig = inspect.signature(simulate_ecifull)
    allow = set(sig.parameters.keys())
    out = {}
    def put(cands, val):
        for name in cands:
            if name in allow:
                out[name] = val; return True
        return False
    for k, v in (mods or {}).items():
        if k in allow:
            out[k] = v; continue
        if k == "tvc_rate_lim_deg_s":
            put(["tvc_rate_lim_deg_s","tvc_rate_limit_deg_s","tvc_rate_limit_dps","tvc_rate_dps"], v)
        elif k == "tvc_stick_t":
            put(["tvc_stick_t","tvc_stick_start","tvc_stick_start_s","tvc_stick_t0"], v)
        elif k == "tvc_stick_dt":
            put(["tvc_stick_dt","tvc_stick_duration","tvc_stick_duration_s"], v)
        elif k == "sensor_bias":
            put(["sensor_bias","sensor_bias_xyz","acc_bias_xyz"], v)
        elif k == "delay" and isinstance(v, dict):
            out["event_delay"] = v
    return out

def sim_residual_single(
    t_end: float = 100.0,
    dt: float = 0.05,
    seed: int = 42,
    noise_std: float = 0.15,
    **faults,
):
    """
    Run a single ECIFULL simulation and return (t, channels_dict),
    where channels_dict contains all phys6_eso channels.
    """
    safe_mods = _adapt_fault_kwargs(simulate_ecifull, faults)
    sim = simulate_ecifull(
        t_end=t_end,
        dt=dt,
        preset="nasaspaceflight",
        seed=int(seed),
        noise_std=noise_std,
        **safe_mods,
    )
    a_meas = sim["a_meas"]
    res_pred, _, _ = run_eso(a_meas[:, [0, 2]], dt=dt)
    r_eso_z = res_pred[:, 1]  # ESO z-axis residual

    # Extract all phys6_eso channels
    channels = {
        "ax": a_meas[:, 0],
        "az": a_meas[:, 2],
        "q_dyn": sim["q_dyn"],
        "n_load": sim["n_load"],
        "thrust": sim["thrust"],
        "fpa_deg": sim["fpa_deg"],
        "r_eso_z": r_eso_z,
    }

    t = np.arange(r_eso_z.size) * dt
    return t, channels

def _gen_one_sample(seed, mods, t_end=100.0, dt=0.05):
    """Generate one sample for scatter plot with improved features."""
    t, channels = sim_residual_single(t_end, dt=dt, seed=seed, **mods)
    r = channels["r_eso_z"]

    # STFT特征
    S = stft_feat(r, win_len=256, hop=64)
    e_low = float(np.sum(S[:10, :]))      # 低频能量
    e_mid = float(np.sum(S[10:30, :]))    # 中频能量
    e_high = float(np.sum(S[30:, :]))     # 高频能量

    # 时域统计特征
    r_std = float(np.std(r))              # 标准差
    r_peak = float(np.max(np.abs(r)))     # 峰值
    r_kurtosis = float(np.mean((r - np.mean(r))**4) / (r_std**4 + 1e-12))  # 峰度

    # 样本熵
    ent = float(sample_entropy(r, m=2, r=0.2*r_std+1e-12))

    return np.array([e_low, e_mid, e_high, r_std, r_peak, r_kurtosis, ent], dtype=float)

def gen_dataset_scatter(N=240, seed=42, workers=0):
    """Generate dataset for scatter plots."""
    rs = np.random.RandomState(seed)
    classes = [
        ("nominal",            {}),
        ("thrust_drop_15",     {"thrust_drop": 0.15}),
        ("tvc_rate_2dps",      {"tvc_rate_lim_deg_s": 2.0}),
        ("tvc_stick_90s_5s",   {"tvc_stick_t": 90.0, "tvc_stick_dt": 5.0}),
        ("sensor_bias_0p2",    {"sensor_bias": (0.2,0.0,0.0)}),
        ("delay_s3ign_1s",     {"delay": {"S2_sep": 2.0, "Fairing_jettison": 2.0}}),
    ]
    per = max(1, N // len(classes))
    feats, labels, seeds, mods_all = [], [], [], []
    for ci, (_, mods) in enumerate(classes):
        for _ in range(per):
            s = int(rs.randint(0, 1_000_000_000))
            seeds.append(s); mods_all.append(mods); labels.append(ci)
    
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_gen_one_sample, seeds[i], mods_all[i]) for i in range(len(labels))]
            for f in as_completed(futs):
                feats.append(f.result())
    else:
        for i in range(len(labels)):
            feats.append(_gen_one_sample(seeds[i], mods_all[i]))
            
    X = np.asarray(feats, dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    X = (X - mu) / sd
    return X, np.asarray(labels, dtype=int), classes

# =============================================================================
# 4. 绘图函数
# =============================================================================

def plot_fig3_2_timefreq(outdir, seed=42):
    """图3-2：时频图 (STFT + PWVD)

    参数与 diagnosis/features.py 中的 stft/pwvd 保持一致:
    - STFT: win_len=256, hop=64 (采样点数)
    - PWVD: win_len=256 (即 half_lag=128)
    """
    print("[Plot] Fig 3-2 Time-Frequency...")
    t, channels = sim_residual_single(t_end=100.0, seed=seed)
    r = channels["r_eso_z"]  # Extract ESO residual
    dt = t[1] - t[0] if len(t) > 1 else 0.05
    fs = 1.0 / dt

    # 与 features.py 一致的参数
    win_len = 256  # 采样点数
    hop = 64       # 采样点数
    half_lag = win_len // 2  # PWVD half_lag = 128

    # 转换为时间单位供 stft_np 使用
    win_sec = win_len * dt  # 12.8 秒
    hop_sec = hop * dt      # 3.2 秒

    # 统一的 dB 动态范围
    dr_db = 45.0

    # Waveform (图3-2a) - 只画残差，不画阈值（阈值在图3-7中展示）
    outpath = os.path.join(outdir, "fig3_2_wave.png")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, r, lw=LINE_WIDTH, color=DEFAULT_COLORS["nominal"])
    ax.set_title("ESO z 轴残差波形（名义工况）", fontsize=14)
    ax.set_xlabel(r"时间 $t$ (s)")
    ax.set_ylabel(r"ESO z 残差 $r_z$ (m/s²)")
    ax.legend(["残差 $r(t)$"], loc="best", frameon=True, edgecolor='black', fancybox=False)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    # STFT (图3-2b)
    outpath = os.path.join(outdir, "fig3_2_stft.png")
    f, tt, S = stft_np(r, fs=fs, win_sec=win_sec, hop_sec=hop_sec)
    SdB_stft, (vmin_stft, vmax_stft) = db_scale(S, dr_db=dr_db)
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    im = ax.imshow(SdB_stft, origin="lower", aspect="auto", extent=[tt[0], tt[-1], f[0], f[-1]],
                   cmap="viridis", vmin=vmin_stft, vmax=vmax_stft)
    plt.colorbar(im, ax=ax).set_label("功率密度 (dB)")
    ax.set_ylim(0, 10.0)  # fmax
    ax.set_title("STFT 时频图", fontsize=14)
    ax.set_xlabel(r"时间 $t$ (s)")
    ax.set_ylabel(r"频率 $f$ (Hz)")
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    # PWVD (图3-2c)
    outpath = os.path.join(outdir, "fig3_2_pwvd.png")
    f2, tt2, P = pwvd_np(r, fs=fs, half_lag=half_lag, t_subsample=2)
    P = tfr_smooth(P, sigma_t=1.2, sigma_f=1.0)
    SdB_pwvd, _ = db_scale(P, dr_db=dr_db)
    # 使用与 STFT 相同的颜色刻度范围，便于视觉对比
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    im = ax.imshow(SdB_pwvd, origin="lower", aspect="auto", extent=[tt2[0], tt2[-1], f2[0], f2[-1]],
                   cmap="viridis", vmin=vmin_stft, vmax=vmax_stft)
    plt.colorbar(im, ax=ax).set_label("功率密度 (dB)")
    ax.set_ylim(0, 10.0)
    ax.set_title("PWVD 时频图", fontsize=14)
    ax.set_xlabel(r"时间 $t$ (s)")
    ax.set_ylabel(r"频率 $f$ (Hz)")
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Fig 3-2 saved to {outdir}")

def plot_fig3_3_residuals(outdir):
    """图3-3：六类故障残差波形（聚焦故障窗口，统一量程，标出故障时间）"""
    print("[Plot] Fig 3-3 Residuals (Focused & Unified Y)...")

    # 定义每个工况最具区分性的通道
    # Format: (Title, FaultKwargs, XLim, MarkerInfo, BestChannel, ChannelLabel)
    cases = [
        # 名义工况：ESO残差
        ("a_名义工况残差状态", dict(), (40, 60), None, "r_eso_z", "ESO残差 r_z"),

        # 传感器偏置：ax加速度最明显（偏置在x轴）
        ("b_传感器偏置残差状态", dict(sensor_bias=(1.5,0.0,0.0)), (40, 60), None, "ax", "轴向加速度 a_x"),

        # 推力降级：推力信号最直接
        ("c_推力降级15%残差状态", dict(thrust_drop=0.15), (40, 60), None, "thrust", "推力 T"),

        # 事件延迟：ESO残差或动压变化明显
        ("d_事件延迟(S2分离+2s)残差状态",
         dict(event_delay={"S2_sep": 2.0, "Fairing_jettison": 2.0}),
         (155, 175),
         {"vlines": [161.0, 163.0], "labels": ["S2分离(名义)", "S2分离(延迟)"], "colors": ["red", "orange"]},
         "r_eso_z", "ESO残差 r_z"),

        # TVC速率限制：过载或航迹角变化明显
        ("e_TVC速率限制(2°/s)残差状态", dict(tvc_rate_lim_deg_s=2.0), (40, 60), None, "n_load", "过载 n"),

        # TVC卡滞：过载或航迹角最明显
        ("f_TVC卡滞(90s×5s)残差状态",
         dict(tvc_stick=(90.0, 5.0)),
         (85, 105),
         {"vlines": [90.0, 95.0], "labels": ["卡滞开始", "卡滞结束"], "colors": ["red", "green"]},
         "n_load", "过载 n"),
    ]

    # 生成名义工况参考数据（用于计算残差）
    print("[Info] Generating nominal reference data...")
    t_nom, channels_nom = sim_residual_single(t_end=200.0)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    for idx, case in enumerate(cases):
        # 解包参数
        tag, kw, xlim, markers, best_channel, ch_label = case

        # Simulate enough time to cover the xlim
        sim_t_end = max(200.0, xlim[1] + 10.0)
        t, channels = sim_residual_single(t_end=sim_t_end, **kw)

        # 计算该通道的残差（相对于名义工况）
        signal = channels[best_channel]
        signal_nom = channels_nom[best_channel][:len(signal)]  # 确保长度一致
        residual = signal - signal_nom

        # Slice data for plotting
        mask = (t >= xlim[0]) & (t <= xlim[1])
        t_slice = t[mask]
        r_slice = residual[mask]

        ax = axes[idx]
        ax.plot(t_slice, r_slice, antialiased=True, lw=1.2, label="残差")

        # 绘制故障时间标线
        if markers and "vlines" in markers:
            vlines = markers["vlines"]
            labels = markers.get("labels", [f"t={v}s" for v in vlines])
            colors = markers.get("colors", ["red"] * len(vlines))

            for v, label, color in zip(vlines, labels, colors):
                if xlim[0] <= v <= xlim[1]:  # 只画在当前时间窗口内的标线
                    ax.axvline(v, linestyle="--", color=color, alpha=0.6, lw=1.5, label=label)

        ax.set_title(tag, loc="left", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel(f"{ch_label} 残差")
        ax.set_xlim(xlim)

        # Apply individual Y-axis limit for each subplot (1.3x margin for clarity)
        if r_slice.size > 0:
            local_max = float(np.max(np.abs(r_slice)))
            if local_max > 0:
                ylim = 1.3 * local_max
                ax.set_ylim(-ylim, ylim)

        # 如果有标线，添加图例
        if markers and "vlines" in markers:
            ax.legend(loc="best", fontsize=8, framealpha=0.8)

    axes[-2].set_xlabel("时间 t (s)")
    axes[-1].set_xlabel("时间 t (s)")

    outpath = os.path.join(outdir, "fig3_3_residuals.png")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Fig 3-3 saved to {outpath}")

def plot_fig3_3_multichannel_part1(outdir, seed=42):
    """图3-3 多通道残差对比图（第1部分：前4个通道）
    展示 ax, az, q_dyn, n_load 四个通道在六种故障工况下的残差
    布局：4行6列，每行一个通道，每列一种故障类型
    """
    print("[Plot] Fig 3-3 Multichannel Residuals Part 1 (4 channels x 6 faults)...")

    # 定义6种故障工况和对应的时间窗口
    cases = [
        ("名义工况", {}, 50.0, 70.0),
        ("推力下降", {"thrust_drop": 0.15}, 50.0, 70.0),
        ("TVC速率受限", {"tvc_rate_lim_deg_s": 0.35}, 30.0, 50.0),
        ("TVC卡滞", {"tvc_stick": (30.0, 20.0)}, 25.0, 55.0),
        ("传感器偏差", {"sensor_bias": (1.5, 0.0, 0.0)}, 10.0, 30.0),
        ("事件延迟", {"event_delay": {"S2_sep": 2.0, "Fairing_jettison": 2.0}}, 158.0, 180.0),
    ]

    # 前4个通道及其标签
    channel_names = ["ax", "az", "q_dyn", "n_load"]
    channel_labels = ["ax (m/s²)", "az (m/s²)", "动压 q (Pa)", "过载 n"]

    # 生成名义工况参考数据
    print("[Info] Generating nominal reference data...")
    t_nom, channels_nom = sim_residual_single(t_end=200.0, seed=seed)

    # 自定义纵横比 23.5:13.5
    fig_width = 13.5  # inches
    fig_height = 23.5  # inches
    fig, axes = plt.subplots(8, 3, figsize=(fig_width, fig_height), constrained_layout=True)

    # 每两行对应一个通道，每行3个工况：行对 (0,1)->ax (6工况), (2,3)->az (6工况), ...
    for ch_block_idx, (ch_name, ch_label) in enumerate(zip(channel_names, channel_labels)):
        for fault_idx, (fault_name, fault_params, t_start, t_end) in enumerate(cases):
            # 计算该故障应该放在哪一行哪一列：前3个在第一行，后3个在第二行
            local_row = fault_idx // 3  # 0->0, 1->0, 2->0, 3->1, 4->1, 5->1
            local_col = fault_idx % 3   # 0->0, 1->1, 2->2, 3->0, 4->1, 5->2
            row_idx = ch_block_idx * 2 + local_row
            col_idx = local_col

            # 生成故障工况数据
            sim_t_end = max(200.0, t_end + 20.0)
            t, channels = sim_residual_single(t_end=sim_t_end, seed=seed, **fault_params)

            # 计算残差（故障 - 名义）
            signal = channels[ch_name]
            signal_nom = channels_nom[ch_name][:len(signal)]
            residual = signal - signal_nom

            # 截取时间窗口
            mask = (t >= t_start) & (t <= t_end)
            t_slice = t[mask]
            r_slice = residual[mask]

            ax = axes[row_idx, col_idx]
            ax.plot(t_slice, r_slice, lw=0.8, color='#1f77b4')
            ax.grid(True, linestyle='--', alpha=0.3, lw=0.5)
            ax.tick_params(labelsize=7)
            ax.set_title(fault_name, fontsize=7, pad=2)

            # 左侧第一列标注通道名（只在每个通道块的第一行第一列）
            if col_idx == 0 and local_row == 0:
                ax.set_ylabel(f"{ch_label}\n残差", fontsize=7)
            # 底部两行加X轴标签
            if row_idx >= 6:
                ax.set_xlabel("时间 (s)", fontsize=6)

            # 自动调整y轴范围
            if r_slice.size > 0:
                local_max = float(np.max(np.abs(r_slice)))
                if local_max > 1e-12:
                    ylim = 1.3 * local_max
                    ax.set_ylim(-ylim, ylim)

    outpath = os.path.join(outdir, "fig3_3_multichannel_part1.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Fig 3-3 Multichannel Part 1 saved to {outpath}")

def plot_fig3_3_multichannel_part2(outdir, seed=42):
    """图3-3 多通道残差对比图（第2部分：后3个通道）
    展示 thrust, fpa_deg, r_eso_z 三个通道在六种故障工况下的残差
    布局：3行6列，每行一个通道，每列一种故障类型
    """
    print("[Plot] Fig 3-3 Multichannel Residuals Part 2 (3 channels x 6 faults)...")

    # 定义6种故障工况和对应的时间窗口（与part1保持一致）
    cases = [
        ("名义工况", {}, 50.0, 70.0),
        ("推力下降", {"thrust_drop": 0.15}, 50.0, 70.0),
        ("TVC速率受限", {"tvc_rate_lim_deg_s": 0.35}, 30.0, 50.0),
        ("TVC卡滞", {"tvc_stick": (30.0, 20.0)}, 25.0, 55.0),
        ("传感器偏差", {"sensor_bias": (1.5, 0.0, 0.0)}, 10.0, 30.0),
        ("事件延迟", {"event_delay": {"S2_sep": 2.0, "Fairing_jettison": 2.0}}, 158.0, 180.0),
    ]

    # 后3个通道及其标签
    channel_names = ["thrust", "fpa_deg", "r_eso_z"]
    channel_labels = ["推力 T (N)", "弹道倾角 γ (deg)", "ESO z残差 (m/s²)"]

    # 生成名义工况参考数据
    print("[Info] Generating nominal reference data...")
    t_nom, channels_nom = sim_residual_single(t_end=200.0, seed=seed)

    # 自定义纵横比 23.5:13.5
    fig_width = 13.5  # inches
    fig_height = 23.5  # inches
    fig, axes = plt.subplots(6, 3, figsize=(fig_width, fig_height), constrained_layout=True)

    # 每两行对应一个通道，每行3个工况：行对 (0,1)->thrust (6工况), (2,3)->fpa_deg (6工况), (4,5)->r_eso_z (6工况)
    for ch_block_idx, (ch_name, ch_label) in enumerate(zip(channel_names, channel_labels)):
        for fault_idx, (fault_name, fault_params, t_start, t_end) in enumerate(cases):
            # 计算该故障应该放在哪一行哪一列：前3个在第一行，后3个在第二行
            local_row = fault_idx // 3  # 0->0, 1->0, 2->0, 3->1, 4->1, 5->1
            local_col = fault_idx % 3   # 0->0, 1->1, 2->2, 3->0, 4->1, 5->2
            row_idx = ch_block_idx * 2 + local_row
            col_idx = local_col

            # 生成故障工况数据
            sim_t_end = max(200.0, t_end + 20.0)
            t, channels = sim_residual_single(t_end=sim_t_end, seed=seed, **fault_params)

            # 计算残差（故障 - 名义）
            signal = channels[ch_name]
            signal_nom = channels_nom[ch_name][:len(signal)]
            residual = signal - signal_nom

            # 截取时间窗口
            mask = (t >= t_start) & (t <= t_end)
            t_slice = t[mask]
            r_slice = residual[mask]

            ax = axes[row_idx, col_idx]
            ax.plot(t_slice, r_slice, lw=0.8, color='#1f77b4')
            ax.grid(True, linestyle='--', alpha=0.3, lw=0.5)
            ax.tick_params(labelsize=7)
            ax.set_title(fault_name, fontsize=7, pad=2)

            # 左侧第一列标注通道名（只在每个通道块的第一行第一列）
            if col_idx == 0 and local_row == 0:
                ax.set_ylabel(f"{ch_label}\n残差", fontsize=7)
            # 底部两行加X轴标签
            if row_idx >= 4:
                ax.set_xlabel("时间 (s)", fontsize=6)

            # 自动调整y轴范围
            if r_slice.size > 0:
                local_max = float(np.max(np.abs(r_slice)))
                if local_max > 1e-12:
                    ylim = 1.3 * local_max
                    ax.set_ylim(-ylim, ylim)

    outpath = os.path.join(outdir, "fig3_3_multichannel_part2.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Fig 3-3 Multichannel Part 2 saved to {outpath}")

def plot_fig3_3_split_subplots(outdir, seed=42):
    """图3-3 多通道残差拆分图
    将7个通道×6种故障拆分为21个独立图像，每个图像为1行×2列布局
    每3张图展示同一通道在不同故障工况下的残差（与名义工况的差值）
    输出到 ch3_figures/fig3_3_split/ 子文件夹

    注意：
    - 传感器偏差故障只影响ax通道
    - 事件延迟故障在确定性仿真中残差为0，改用原始信号展示
    """
    print("[Plot] Fig 3-3 Split Subplots (21 separate images, 1x2 each)...")

    # 创建输出子目录
    split_outdir = os.path.join(outdir, "fig3_3_split")
    os.makedirs(split_outdir, exist_ok=True)

    # 定义6种故障工况和对应的优化时间窗口（扩大窗口使残差更明显）
    cases = [
        ("名义工况", {}, 30.0, 90.0),
        ("推力下降", {"thrust_drop": 0.15}, 20.0, 100.0),
        ("TVC速率受限", {"tvc_rate_lim_deg_s": 0.35}, 20.0, 100.0),
        ("TVC卡滞", {"tvc_stick": (30.0, 20.0)}, 25.0, 80.0),
        ("传感器偏差", {"sensor_bias": (1.5, 0.0, 0.0)}, 0.0, 50.0),
        ("事件延迟", {"event_delay": {"S2_sep": 2.0, "Fairing_jettison": 2.0}}, 155.0, 185.0),
    ]

    # 全部7个通道及其标签
    channel_names = ["ax", "az", "q_dyn", "n_load", "thrust", "fpa_deg", "r_eso_z"]
    channel_labels = [
        "ax (m/s²)", "az (m/s²)", "动压 q (Pa)", "过载 n",
        "推力 T (N)", "弹道倾角 γ (deg)", "ESO z残差 (m/s²)"
    ]
    channel_names_zh = [
        "加速度ax", "加速度az", "动压q", "过载n",
        "推力T", "弹道倾角γ", "ESO残差z"
    ]

    # 生成名义工况参考数据
    print("[Info] Generating nominal reference data...")
    t_nom, channels_nom = sim_residual_single(t_end=200.0, seed=seed)

    # 将6种故障分成3对（每对2种故障）
    fault_pairs = [(0, 1), (2, 3), (4, 5)]  # (名义,推力下降), (TVC速率,TVC卡滞), (传感器偏差,事件延迟)

    img_count = 0
    for ch_idx, (ch_name, ch_label, ch_zh) in enumerate(zip(channel_names, channel_labels, channel_names_zh)):
        for pair_idx, (fault_i, fault_j) in enumerate(fault_pairs):
            img_count += 1

            # 创建1行2列的图
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            for subplot_idx, fault_idx in enumerate([fault_i, fault_j]):
                fault_name, fault_params, t_start, t_end = cases[fault_idx]

                # 生成故障工况数据
                sim_t_end = max(200.0, t_end + 20.0)
                t, channels = sim_residual_single(t_end=sim_t_end, seed=seed, **fault_params)

                # 获取信号
                signal = channels[ch_name]
                signal_nom = channels_nom[ch_name][:len(signal)]

                # 判断是否需要使用原始信号（事件延迟残差为0，使用原始信号）
                is_event_delay = "event_delay" in fault_params

                if is_event_delay:
                    # 事件延迟：绘制原始信号
                    plot_signal = signal
                    ylabel_suffix = ""
                else:
                    # 其他故障：绘制残差（与名义的差值）
                    plot_signal = signal - signal_nom
                    ylabel_suffix = " 残差"

                # 截取时间窗口
                mask = (t >= t_start) & (t <= t_end)
                t_slice = t[mask]
                signal_slice = plot_signal[mask]

                ax = axes[subplot_idx]
                ax.plot(t_slice, signal_slice, lw=1.2, color='#1f77b4')
                ax.grid(True, linestyle='--', alpha=0.4, lw=0.6)
                ax.tick_params(labelsize=9)

                # 事件延迟故障添加特殊标注
                if is_event_delay:
                    ax.set_title(f"{fault_name} (原始信号)", fontsize=11, fontweight='bold')
                else:
                    ax.set_title(f"{fault_name}", fontsize=11, fontweight='bold')

                ax.set_xlabel("时间 t (s)", fontsize=10)
                ax.set_ylabel(f"{ch_label}{ylabel_suffix}", fontsize=10)

                # 自动调整y轴范围
                if signal_slice.size > 0:
                    y_min, y_max = signal_slice.min(), signal_slice.max()
                    y_range = y_max - y_min
                    if y_range > 1e-12:
                        margin = 0.15 * y_range
                        ax.set_ylim(y_min - margin, y_max + margin)
                    elif not is_event_delay:  # 残差为0的情况
                        ax.set_ylim(-0.1, 0.1)
                        ax.text(0.5, 0.5, '残差≈0', transform=ax.transAxes,
                               ha='center', va='center', fontsize=12, color='gray')

            # 设置总标题
            fig.suptitle(f"通道: {ch_zh} ({ch_label})", fontsize=12, fontweight='bold', y=1.02)
            plt.tight_layout()

            # 保存图像
            filename = f"fig3_3_{img_count:02d}_{ch_name}_pair{pair_idx+1}.png"
            outpath = os.path.join(split_outdir, filename)
            fig.savefig(outpath, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  [{img_count}/21] Saved {filename}")

    print(f"[Done] All 21 split figures saved to {split_outdir}")


def plot_fig3_7_residual_threshold(outdir, seed=42):
    """图3-7：ESO残差与自适应阈值"""
    print("[Plot] Fig 3-7 Residual Threshold...")

    # 运行名义工况仿真
    t, ch_nom = sim_residual_single(t_end=100.0, seed=seed)
    r_nom = ch_nom["r_eso_z"]

    # 故障注入：在t=50s时模拟传感器突变故障（阶跃扰动）
    t_fault_inject = 50.0
    fault_magnitude = 0.8  # m/s²
    i_fault = np.searchsorted(t, t_fault_inject)
    r_combined = r_nom.copy()
    r_combined[i_fault:] += fault_magnitude

    # 基于故障前数据计算自适应阈值
    win = 101
    mu, sigma = moving_mean_std(r_nom[:i_fault], win=win)
    noise_99 = np.percentile(np.abs(r_nom[:i_fault]), 99)
    thr_value = max(np.mean(np.abs(mu) + 2.5 * sigma), 1.15 * noise_99)
    thr = np.full_like(r_nom, thr_value)

    # 故障检测
    N_consecutive = 3
    exceed = np.abs(r_combined) > thr
    t_detect, i_detect = None, None
    count = 0
    for i in range(len(exceed)):
        if exceed[i]:
            count += 1
            if count >= N_consecutive:
                i_detect = i - N_consecutive + 1
                t_detect = t[i_detect]
                break
        else:
            count = 0

    # 计算诊断时延
    latency_ms = (t_detect - t_fault_inject) * 1000 if t_detect else None

    # 绘图
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, r_combined, lw=LINE_WIDTH, label=r"残差 $r(t)$", color=DEFAULT_COLORS["nominal"])
    ax.plot(t, thr, lw=LINE_WIDTH, label=r"阈值 $\pm T_{ad}$", color=DEFAULT_COLORS["fault"])
    ax.plot(t, -thr, lw=LINE_WIDTH, color=DEFAULT_COLORS["fault"])

    # 故障注入时刻
    ax.axvline(t_fault_inject, linestyle=":", color="gray", lw=1.5, label=f"故障注入 t={t_fault_inject:.0f}s")

    # 诊断触发时刻
    if t_detect is not None:
        ax.axvline(t_detect, linestyle="--", color="green", lw=2.0)
        ax.text(t_detect + 1, thr_value * 0.8, f"诊断触发\nt={t_detect:.2f}s\n时延={latency_ms:.0f}ms",
                fontsize=9, color="green", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.9))

    ax.set_title("ESO残差与自适应阈值", fontsize=14)
    ax.set_xlabel(r"时间 $t$ (s)")
    ax.set_ylabel(r"残差 $r$ (m/s²)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    outpath = os.path.join(outdir, "fig3_7_residual.png")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    print(f"[Done] Fig 3-7 saved to {outpath}")
    if t_detect is not None:
        print(f"       Diagnosis trigger: {t_detect:.2f}s")

def plot_scatter_figs(outdir, seed=42, n_samples=240, workers=8, selected_classes=None):
    """
    图3-4 (2D) 和 图3-8 (3D) 散点图
    使用LDA降维最大化类间距离
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    print(f"[Plot] Fig 3-4 & 3-8 Scatter Plots (workers={workers})...")

    # 生成数据集 (7维特征)
    X, y, classes = gen_dataset_scatter(N=n_samples, seed=seed, workers=workers)

    labels_zh = ["名义", "推力降级15%", "TVC速率限制", "TVC卡滞", "传感器偏置", "事件延迟"]
    if selected_classes is None:
        selected_classes = list(range(len(classes)))

    color_palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]
    markers = ['o', 's', '^', 'D', 'v', 'P']

    # LDA降维 (最多n_classes-1维)
    n_comp = min(3, len(classes) - 1)
    lda = LDA(n_components=n_comp)
    X_lda = lda.fit_transform(X, y)
    print(f"  LDA explained variance ratio: {lda.explained_variance_ratio_[:n_comp].sum()*100:.1f}%")

    # Fig 3-4: 2D Scatter (LD1 vs LD2)
    print("  Generating Fig 3-4 (2D scatter)...")
    fig, ax = plt.subplots(figsize=(10, 7))
    for ci in selected_classes:
        if ci >= len(classes): continue
        mask = (y == ci)
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], s=50, alpha=0.8,
                   label=labels_zh[ci], color=color_palette[ci],
                   marker=markers[ci], edgecolors='white', linewidths=0.5)

    ax.set_xlabel(r"低频能量 $e_1$（标准化）", fontsize=12)
    ax.set_ylabel(r"中频能量 $e_2$（标准化）", fontsize=12)
    ax.set_title("时频特征散点图", fontsize=14)
    ax.legend(loc="best", ncol=2, fontsize=10, frameon=True, edgecolor='black', fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    outpath = os.path.join(outdir, "fig3_4_timefreq_scatter.png")
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")

    # Fig 3-8: 3D Scatter (LD1, LD2, LD3)
    print("  Generating Fig 3-8 (3D scatter)...")
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    for ci in selected_classes:
        if ci >= len(classes): continue
        mask = (y == ci)
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], X_lda[mask, 2], s=40, alpha=0.8,
                   label=labels_zh[ci], color=color_palette[ci],
                   marker=markers[ci], edgecolors='white', linewidths=0.3)

    ax.set_xlabel(r"低频能量 $e_1$", fontsize=11)
    ax.set_ylabel(r"中频能量 $e_2$", fontsize=11)
    ax.set_zlabel(r"样本熵 $SampEn$", fontsize=11)
    ax.set_title("三维时频特征散点图", fontsize=14, pad=20)
    ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=True, edgecolor='black', fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    outpath = os.path.join(outdir, "fig3_8_timefreq_3d.png")
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")

    print(f"[Done] Scatter plots saved to {outdir}")

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Chapter 3 Figures")
    parser.add_argument("--outdir", type=str, default="./exports/ch3_figures", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--fig", type=str, help="Generate specific figure (3-2, 3-3, 3-3a, 3-3b, 3-4, 3-7, 3-8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for parallel processing")
    parser.add_argument("--n_samples", type=int, default=240, help="Number of samples for scatter plots")
    parser.add_argument("--selected_classes", type=str, default=None,
                        help="Comma-separated class indices for scatter plots (e.g., '0,1,3,5')")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup_style()

    # 解析 selected_classes 参数
    selected_classes = None
    if args.selected_classes:
        try:
            selected_classes = [int(x.strip()) for x in args.selected_classes.split(',')]
        except ValueError:
            print(f"[Warning] Invalid --selected_classes format: {args.selected_classes}")
            selected_classes = None

    if args.all or args.fig == "3-2":
        plot_fig3_2_timefreq(args.outdir, args.seed)
    if args.all or args.fig == "3-3":
        plot_fig3_3_residuals(args.outdir)
    if args.all or args.fig == "3-3a":
        plot_fig3_3_multichannel_part1(args.outdir, args.seed)
    if args.all or args.fig == "3-3b":
        plot_fig3_3_multichannel_part2(args.outdir, args.seed)
    if args.fig == "3-3s":
        plot_fig3_3_split_subplots(args.outdir, args.seed)
    if args.all or args.fig == "3-7":
        plot_fig3_7_residual_threshold(args.outdir, args.seed)
    if args.all or args.fig in ["3-4", "3-8"]:
        plot_scatter_figs(args.outdir, args.seed, n_samples=args.n_samples,
                         workers=args.workers, selected_classes=selected_classes)

    if not args.all and not args.fig:
        print("Please specify --all or --fig [3-2|3-3|3-3a|3-3b|3-3s|3-4|3-7|3-8]")

if __name__ == "__main__":
    main()
