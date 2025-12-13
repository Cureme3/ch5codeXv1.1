# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add path to find sim and diagnosis modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sim.eci_full import simulate_ecifull
from src.sim.kz1a_eci_core import build_timeline_kz1a, KZ1AConfig
from diagnosis.eso import run_eso
from diagnosis.features import extract_features_from_residual

def pca_reduce(X: np.ndarray, out_dim: int = 4):
    """
    对样本矩阵 X 做 PCA 降维，返回降维后的特征、均值和投影矩阵。
    X: (N, D)
    out_dim: 目标维度，必须 <= D
    return:
        Z: (N, out_dim)  降维后的特征
        mean: (D,)       原始特征均值，用于将来复现 PCA
        components: (D, out_dim)  PCA 投影矩阵（每列是一个主成分）
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    out_dim = max(1, min(out_dim, D))
    mean = X.mean(axis=0)
    X0 = X - mean
    # 使用 SVD 实现 PCA
    U, S, Vt = np.linalg.svd(X0, full_matrices=False)
    W = Vt[:out_dim].T   # (D, out_dim)
    Z = X0 @ W           # (N, out_dim)
    return Z, mean, W

def _get_fault_roi_time_range(t, cls_name, fault_params, window_sec):
    """
    根据故障类型和参数，返回一个 (t_roi_start, t_roi_end)（单位: s），
    只在这个时间段内做滑动窗口。
    t: 仿真时间轴 (1D ndarray)，单位秒
    cls_name: 类别名字符串，例如 "nominal", "thrust_drop_15", ...
    fault_params: 为该仿真构造 simulate_ecifull 时用的故障参数 dict
    window_sec: 特征窗口长度（秒）
    """
    t0_all = float(t[0])
    t1_all = float(t[-1])

    # 默认：全程（防止忘记配置）
    t0, t1 = t0_all, t1_all

    # 1) 推力降级 15%：从起飞后到三级关机前，中后段加速度差异最大
    if cls_name in ("nominal", "thrust_drop_15"):
        # S1+S2+S3 中段，避免起飞前和关机后的无信息片段
        t0 = max(t0_all, 20.0)
        t1 = min(t1_all, 260.0)

    # 2) TVC 速率限制：主要在大气层、气动载荷较大阶段显现
    elif cls_name == "tvc_rate_2dps":
        # 对照验证脚本，速率限制最明显大约在 40~120 s
        t0 = max(t0_all, 40.0)
        t1 = min(t1_all, 120.0)

    # 3) TVC 卡滞：用参数里的卡滞开始时间和持续时间做 ROI
    elif cls_name == "tvc_stick_90s_5s":
        # make_dataset_cache 里你在构造 sim_args 时已经把
        # tvc_stick=(t_start, dt) 写进 fault_params 里，直接复用
        t_stick, dt_fault = fault_params.get("tvc_stick", (90.0, 5.0))
        t_fault_start = float(t_stick)
        t_fault_end   = float(t_stick + dt_fault)
        # 在卡滞前后各多保留 1~2 个窗口长度
        pad = 2.0 * window_sec
        t0 = max(t0_all, t_fault_start - pad)
        t1 = min(t1_all, t_fault_end + pad)

    # 4) 传感器偏置：从注入时刻开始，ESO 残差会一直带偏
    elif cls_name == "sensor_bias_0p2":
        # 偏置在仿真一开始就存在，选中后段加速度较大的区间
        t0 = max(t0_all, 40.0)
        t1 = min(t1_all, 260.0)

    # 5) 事件延迟（S2_sep / Fairing_jettison 延迟）：
    elif cls_name == "delay_s3ign_1s":
        # 获取名义事件时间
        _, events_nom = build_timeline_kz1a(KZ1AConfig())
        t_s2_sep_nom = events_nom["S2_sep"]
        t_fair_nom = events_nom["Fairing_jettison"]
        
        # 注入时最大延迟约 3.0s
        max_delay = 3.0
        
        # ROI 覆盖两个事件
        t_start_ev = min(t_s2_sep_nom, t_fair_nom)
        t_end_ev = max(t_s2_sep_nom, t_fair_nom)
        
        t0 = max(t0_all, t_start_ev - 0.5 * window_sec)
        t1 = min(t1_all, t_end_ev + max_delay + 0.5 * window_sec)

    # 保证 ROI 至少容纳一个窗口
    if t1 - t0 < window_sec:
        mid = 0.5 * (t0 + t1)
        t0 = max(t0_all, mid - 0.5 * window_sec)
        t1 = min(t1_all, mid + 0.5 * window_sec)

    return t0, t1

import multiprocessing as mp

def _worker_job(job):
    """
    Worker function for multiprocessing.
    job: (idx, seed_i, cls_cfg, window_sec, hop_sec, spec_method)
    """
    idx, seed_i, (ci, name, params), window_sec, hop_sec, spec_method, channel_mode = job
    
    is_delay_cls = (name == "delay_s3ign_1s")
    
    # Re-initialize RNG for this worker
    rng = np.random.RandomState(seed_i)
    
    # Randomize parameters slightly for robustness (Copied from original loop)
    p = params.copy()
    
    # 1. Prepare arguments for simulate_ecifull
    sim_args = {
        "t_end": 350.0,
        "dt": 0.05,
        "noise_std": 0.15,
        "seed": rng.randint(0, 2**31 - 1),
        "preset": "nasaspaceflight"
    }

    # 2. Map Fault Parameters
    if "thrust_drop" in p: 
        # Note: simulate_ecifull applies drop globally, t_fault ignored in this wrapper
        sim_args["thrust_drop"] = rng.uniform(0.1, 0.2)
    
    if "tvc_rate_lim_deg_s" in p:
        sim_args["tvc_rate_lim_deg_s"] = rng.uniform(0.25, 0.45)
        
    if "tvc_stick_t" in p or "tvc_stick_dt" in p:
        t_start = p.get("tvc_stick_t", 30.0)
        dt_dur = p.get("tvc_stick_dt", 20.0)
        # Jitter
        t_start = rng.uniform(t_start - 5.0, t_start + 5.0)
        dt_dur = rng.uniform(dt_dur * 0.75, dt_dur * 1.25)
        sim_args["tvc_stick"] = (t_start, dt_dur)
        
    if "sensor_bias" in p:
        base = np.asarray(p["sensor_bias"], dtype=float).ravel()
        # Increase bias range: U(0.3, 0.6) relative scale effectively
        # Original request: bias ~ U(0.3g, 0.6g). 
        # The base param in 'classes' is (1.5, 0.0, 0.0) which is ~0.15g.
        # So we need a scale factor of approx 2.0 to 4.0 to reach 0.3g-0.6g.
        # Let's adjust scale to match user request: U(2.0, 4.0) * 1.5 m/s^2 ~= 3.0 - 6.0 m/s^2 (0.3-0.6g)
        scale = rng.uniform(2.0, 4.0)
        sim_args["sensor_bias"] = tuple(base * scale)
        
    if "delay" in p and isinstance(p["delay"], dict):
        # Jitter delays
        d_map = {}
        for k, v in p["delay"].items():
            if k in ["S2_sep", "Fairing_jettison"]:
                # User request: U(1.5, 3.0) for delay
                d_map[k] = rng.uniform(1.5, 3.0)
            else:
                d_map[k] = rng.uniform(0.5 * v, 1.5 * v)
        sim_args["event_delay"] = d_map

    # 3. Run Simulation
    sim = simulate_ecifull(**sim_args)
    a_meas = sim["a_meas"]
    t = sim["t"]
    
    # Feature Extraction (ESO -> Residual -> Sliding Window -> Features)
    # Use X and Z axes (indices 0 and 2)
    res_pred, _, _ = run_eso(a_meas[:, [0, 2]], dt=0.05)
    r_eso_x = res_pred[:, 0] # Residual channel X
    r_eso_z = res_pred[:, 1] # Residual channel Z
    
    # Prepare channels based on channel_mode
    def _extract_channels(sim_res, r_eso_z_val):
        a = sim_res["a_meas"]
        if channel_mode == "raw_z":
            return [a[:, 2]]
        elif channel_mode == "raw_xyz":
            return [a[:, 0], a[:, 1], a[:, 2]]
        elif channel_mode == "raw_xyz_eso":
            return [a[:, 0], a[:, 1], a[:, 2], r_eso_z_val]
        elif channel_mode == "phys6":
            return [a[:, 0], a[:, 2], sim_res["q_dyn"], sim_res["n_load"], sim_res["thrust"], sim_res["fpa_deg"]]
        elif channel_mode == "phys6_eso":
            return [a[:, 0], a[:, 2], sim_res["q_dyn"], sim_res["n_load"], sim_res["thrust"], sim_res["fpa_deg"], r_eso_z_val]
        else:
            return [a[:, 2]]

    channels = _extract_channels(sim, r_eso_z)
    
    # --- Weak Fault Filtering Logic ---
    # WEAK_FAULT_THRESH = 0.03 (3%)
    BASE_FAULT_THRESH = 0.03
    DELAY_FAULT_THRESH = 0.02 # Lower threshold for delay class
    
    current_thresh = DELAY_FAULT_THRESH if is_delay_cls else BASE_FAULT_THRESH
    
    # If this is a fault case (ci > 0), generate nominal reference
    channels_nom = None
    if ci > 0:
        # Generate nominal trajectory with SAME seed
        # Note: We must use the exact same seed and parameters as the fault case, 
        # BUT without the fault injection parameters.
        # The 'sim_args' dict already has the seed. We just need to remove fault keys.
        nom_args = sim_args.copy()
        for k in ["thrust_drop", "tvc_rate_lim_deg_s", "tvc_stick", "sensor_bias", "event_delay"]:
            nom_args.pop(k, None)
            
        sim_nom = simulate_ecifull(**nom_args)
        # Also need ESO for nominal if used
        res_pred_nom, _, _ = run_eso(sim_nom["a_meas"][:, [0, 2]], dt=0.05)
        r_eso_z_nom = res_pred_nom[:, 1]
        
        channels_nom = _extract_channels(sim_nom, r_eso_z_nom)

    # Sliding Window Logic
    dt = 0.05
    
    # --- ROI Logic ---
    t_roi0, t_roi1 = _get_fault_roi_time_range(t, name, sim_args, window_sec)
    
    # Convert to indices
    i_roi0 = max(0, int(np.floor(t_roi0 / dt)))
    i_roi1 = min(len(t), int(np.ceil(t_roi1 / dt)))
    
    # Slice all channels to ROI
    channels_roi = [ch[i_roi0:i_roi1] for ch in channels]
    if channels_nom:
        channels_nom_roi = [ch[i_roi0:i_roi1] for ch in channels_nom]
    else:
        channels_nom_roi = None
    
    win_pts = int(window_sec / dt)
    hop_pts = int(hop_sec / dt)
    n_pts = len(channels_roi[0])
    
    feats_list = []
    labels_list = []
    sim_ids_list = [] # Track sim_id (idx)
    
    n_filtered = 0
    n_total_wins = 0
    
    # If ROI is shorter than window, take the whole ROI as one window (fallback)
    if n_pts < win_pts:
        if n_pts > 0:
            indices = [0]
        else:
            indices = []
    else:
        indices = range(0, n_pts - win_pts + 1, hop_pts)
        
    n_total_wins = len(indices) if hasattr(indices, "__len__") else len(list(indices))
    # Re-create indices generator/list because len() might consume it if it was an iterator (range is safe)
    if not isinstance(indices, (list, range)):
         indices = range(0, n_pts - win_pts + 1, hop_pts)

    for start_idx in indices:
        # Determine end index, clamping to n_pts
        end_idx = start_idx + win_pts
        if end_idx > n_pts:
            end_idx = n_pts
            
        # Weak Fault Filtering (skip for delay and sensor_bias classes)
        is_sensor_bias_cls = (name == "sensor_bias_0p2")
        if channels_nom_roi and (not is_delay_cls) and (not is_sensor_bias_cls):
            # Calculate max relative RMS diff across channels
            max_delta = 0.0
            for k, ch_roi in enumerate(channels_roi):
                sig_win = ch_roi[start_idx : end_idx]
                nom_win = channels_nom_roi[k][start_idx : end_idx]
                
                rms_nom = np.sqrt(np.mean(nom_win**2))
                rms_diff = np.sqrt(np.mean((sig_win - nom_win)**2))
                
                delta = rms_diff / (rms_nom + 1e-9)
                if delta > max_delta:
                    max_delta = delta
            
            if max_delta < current_thresh:
                n_filtered += 1
                continue # Skip this window
        
        # Extract features for each channel and concatenate
        f_concat = []
        for ch_roi in channels_roi:
            sig_win = ch_roi[start_idx : end_idx]
            
            # Call feature extraction function
            f_ch = extract_features_from_residual(
                sig_win, 
                dt=dt, 
                spec_method=spec_method,
                use_tf_entropy=None, 
                use_sampen=True
            )
            f_concat.append(f_ch)
        
        f_vec = np.concatenate(f_concat)
        feats_list.append(f_vec)
        labels_list.append(ci)
        sim_ids_list.append(idx)
        
    return feats_list, labels_list, sim_ids_list, n_filtered, n_total_wins

def generate_dataset(
    n_samples=200, 
    seed=42,
    window_sec=20.0,
    hop_sec=10.0,
    spec_method="stft",
    n_jobs=1,
    channel_mode="raw_z",
    nominal_factor=1.0
):
    rng = np.random.RandomState(seed)
    
    # 6 Classes matching train_eval_classifier.py
    classes = [
        ("nominal",            {}),
        ("thrust_drop_15",     {"thrust_drop": 0.15, "t_fault": 50.0}), 
        ("tvc_rate_2dps",      {"tvc_rate_lim_deg_s": 0.35}),
        ("tvc_stick_90s_5s",   {"tvc_stick_t": 30.0, "tvc_stick_dt": 20.0}),
        ("sensor_bias_0p2",    {"sensor_bias": (1.5, 0.0, 0.0)}),
        ("delay_s3ign_1s",     {"delay": {"S2_sep": 2.0, "Fairing_jettison": 2.0}}),
    ]
    
    # Determine samples per class
    n_classes = len(classes)
    base_per_class = max(1, n_samples // n_classes)
    
    # Apply nominal_factor
    n_sims_per_class = []
    for i, (name, _) in enumerate(classes):
        if name == "nominal":
            n = int(base_per_class * nominal_factor)
        else:
            n = base_per_class
        n_sims_per_class.append(n)
    
    print(f"[Dataset] Target samples: {n_samples}")
    print(f"[Dataset] Class stats: " + ", ".join([f"{classes[i][0]}={n}" for i, n in enumerate(n_sims_per_class)]))
    
    jobs = []
    global_idx = 0
    for ci, (name, params) in enumerate(classes):
        n_to_gen = n_sims_per_class[ci]
        for _ in range(n_to_gen):
            seed_i = seed + global_idx
            jobs.append((global_idx, seed_i, (ci, name, params), window_sec, hop_sec, spec_method, channel_mode))
            global_idx += 1
            
    feats_all = []
    labels_all = []
    sim_ids_all = []
    
    total_filtered = 0
    total_windows = 0
    
    if n_jobs is None or n_jobs <= 0:
        n_jobs_eff = 1
    else:
        n_jobs_eff = min(n_jobs, mp.cpu_count())
        
    print(f"[parallel] n_jobs={n_jobs_eff}, total_sims={len(jobs)}")
    
    if n_jobs_eff <= 1:
        for job in jobs:
            f_list, l_list, s_list, n_filt, n_tot = _worker_job(job)
            feats_all.extend(f_list)
            labels_all.extend(l_list)
            sim_ids_all.extend(s_list)
            total_filtered += n_filt
            total_windows += n_tot
    else:
        with mp.Pool(processes=n_jobs_eff) as pool:
            for f_list, l_list, s_list, n_filt, n_tot in pool.imap_unordered(_worker_job, jobs, chunksize=1):
                feats_all.extend(f_list)
                labels_all.extend(l_list)
                sim_ids_all.extend(s_list)
                total_filtered += n_filt
                total_windows += n_tot
            
    X_raw = np.array(feats_all, dtype=float)
    y = np.array(labels_all, dtype=int)
    sim_ids = np.array(sim_ids_all, dtype=int)
    
    print(f"Weak Fault Filtering: {total_filtered}/{total_windows} windows dropped ({total_filtered/max(1, total_windows)*100:.1f}%)")
    
    # PCA Reduction (for metadata compatibility)
    out_dim = 4
    X_pca, pca_mean, pca_components = pca_reduce(X_raw, out_dim=out_dim)
    
    print(f"Total samples generated: {len(X_raw)}, raw_dim={X_raw.shape[1]}")
    return X_raw, y, pca_mean, pca_components, sim_ids

def plot_scatter_figs(X, y, outdir, channel_mode="phys6_eso", spec_method="stft"):
    """Generate 2D and 3D scatter plots from the dataset."""
    print("[Plot] Generating scatter plots...")
    os.makedirs(outdir, exist_ok=True)
    
    # Setup fonts
    from matplotlib.font_manager import fontManager
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
        r"/System/Library/Fonts/PingFang.ttc", r"/System/Library/Fonts/STHeiti Medium.ttc",
        r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    family_map = {
        "msyh.ttc": "Microsoft YaHei", "simhei.ttf": "SimHei",
        "PingFang": "PingFang SC", "STHeiti": "STHeiti",
        "NotoSansCJK": "Noto Sans CJK SC", "wqy-microhei": "WenQuanYi Micro Hei",
    }
    chosen_font = "DejaVu Sans"
    for p in candidates:
        if os.path.exists(p):
            try:
                fontManager.addfont(p)
                fn = os.path.basename(p).lower()
                for k, fam in family_map.items():
                    if k.lower() in fn:
                        chosen_font = fam; break
                if chosen_font != "DejaVu Sans": break
            except Exception:
                pass
    plt.rcParams['font.sans-serif'] = [chosen_font, "DejaVu Sans"]
    plt.rcParams['axes.unicode_minus'] = False
    
    # Determine feature indices for r_z channel
    # phys6_eso: [ax, az, q, n, thrust, gamma, r_z] -> r_z is index 6
    # Features per channel:
    # stft: [Low, Mid, High, SampEn, DC] (5 features)
    # pwvd: [Low, Mid, High, TFEnt, SampEn, DC] (6 features)
    
    if channel_mode == "phys6_eso":
        ch_idx = 6
    elif channel_mode == "raw_xyz_eso":
        ch_idx = 3
    elif channel_mode == "raw_z":
        ch_idx = 0
    else:
        print(f"[Plot] Warning: channel_mode '{channel_mode}' not fully supported for auto-plotting. Skipping.")
        return

    if spec_method == "pwvd":
        n_feat = 6
        # Indices relative to channel start: Low=0, Mid=1, SampEn=4
        idx_e1 = 0
        idx_e2 = 1
        idx_se = 4
    else: # stft
        n_feat = 5
        # Indices relative to channel start: Low=0, Mid=1, SampEn=3
        idx_e1 = 0
        idx_e2 = 1
        idx_se = 3
        
    start_idx = ch_idx * n_feat
    
    # Extract features
    try:
        e1 = X[:, start_idx + idx_e1]
        e2 = X[:, start_idx + idx_e2]
        se = X[:, start_idx + idx_se]
    except IndexError:
        print("[Plot] Error: Feature indices out of bounds. Skipping plots.")
        return

    # Normalize for visualization (optional, but usually good for scatter)
    # Actually, the X passed here is raw features. Let's normalize them just for plotting.
    # Or use the raw values? The original script normalized them.
    # Let's normalize.
    def norm(v):
        return (v - np.mean(v)) / (np.std(v) + 1e-12)
    
    e1_n = norm(e1)
    e2_n = norm(e2)
    se_n = norm(se)
    
    labels_zh = ["名义","推力降级15%","TVC速率限制","TVC卡滞","传感器偏置","事件延迟"]
    # Ensure y is int
    y = y.astype(int)
    
    # Fig 3-4: 2D Scatter
    plt.figure(figsize=(7.2, 4.8))
    for ci in range(len(labels_zh)):
        if ci > y.max(): break
        I = (y == ci)
        if np.sum(I) > 0:
            plt.scatter(e1_n[I], e2_n[I], s=12, alpha=0.75, label=labels_zh[ci])
    plt.xlabel("低频能量 e1（标准化）")
    plt.ylabel("中频能量 e2（标准化）")
    plt.title("时频特征散点图（e1 vs e2）")
    plt.legend(ncol=3)
    plt.tight_layout()
    outpath = os.path.join(outdir, "fig3_4_timefreq_scatter.png")
    plt.savefig(outpath)
    plt.close()
    print(f"[Plot] Saved {outpath}")
    
    # Fig 3-8: 3D Scatter
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111, projection='3d')
    for ci in range(len(labels_zh)):
        if ci > y.max(): break
        I = (y == ci)
        if np.sum(I) > 0:
            ax.scatter(e1_n[I], e2_n[I], se_n[I], s=10, alpha=0.7, label=labels_zh[ci])
    ax.set_xlabel("e1")
    ax.set_ylabel("e2")
    ax.set_zlabel("SampEn")
    plt.title("三维时频特征散点图")
    plt.legend(ncol=2)
    plt.tight_layout()
    outpath = os.path.join(outdir, "fig3_8_timefreq_3d.png")
    plt.savefig(outpath)
    plt.close()
    print(f"[Plot] Saved {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="D:\python projects\ch5codeAGv1.2\ch3codev1.1\exports\clf")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--window_sec", type=float, default=20.0, help="Sliding window length in seconds")
    ap.add_argument("--hop_sec", type=float, default=10.0, help="Sliding window hop in seconds")
    ap.add_argument("--spec_method", type=str, default="stft", choices=["stft", "pwvd"], help="Spectral method: stft or pwvd")
    ap.add_argument("--n_jobs", type=int, default=1, help="并行进程数（<= CPU 核心数），1 表示单进程")
    ap.add_argument("--channel_mode", type=str, default="raw_z", 
                    choices=["raw_z", "raw_xyz", "raw_xyz_eso", "phys6", "phys6_eso"], 
                    help="选择特征通道组合")
    ap.add_argument("--nominal_factor", type=float, default=1.0, help="名义工况样本量倍数")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cache = os.path.join(args.outdir, "cache")
    os.makedirs(cache, exist_ok=True)
    key = f"N{args.n_samples}_seed{args.seed}"
    path = os.path.join(cache, f"dataset_{key}.npz")

    X, y, pca_mean, pca_components, sim_ids = generate_dataset(
        n_samples=args.n_samples, 
        seed=args.seed,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        spec_method=args.spec_method,
        n_jobs=args.n_jobs,
        channel_mode=args.channel_mode,
        nominal_factor=args.nominal_factor
    )
    
    mu = X.mean(0)
    sd = X.std(0) + 1e-12
    
    print(f"Saving to {path} ...")
    np.savez_compressed(
        path, 
        X=X, 
        y=y, 
        mu=mu, 
        sd=sd,
        pca_mean=pca_mean,
        pca_components=pca_components,
        channel_mode=args.channel_mode,
        sim_ids=sim_ids,
        nominal_factor=args.nominal_factor
    )
    print("[OK] cache written:", path)
    print("Data Shape:", X.shape)
    
    # Plot scatter figures
    plot_outdir = os.path.join(os.path.dirname(args.outdir), "ch3_figures")
    plot_scatter_figs(X, y, plot_outdir, args.channel_mode, args.spec_method)

if __name__ == "__main__":
    mp.freeze_support() # For Windows support
    main()
