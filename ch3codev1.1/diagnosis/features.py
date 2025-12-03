# -*- coding: utf-8 -*-
import numpy as np
def stft(x, win_len, hop):
    x=np.asarray(x, dtype=float)
    n=len(x); w=np.hanning(win_len)
    frames=[]
    for s in range(0, n-win_len+1, hop):
        seg=x[s:s+win_len]*w
        spec=np.fft.rfft(seg)
        frames.append(np.abs(spec))
    return np.array(frames).T
def pwvd(x, win_len):
    x=np.asarray(x, dtype=float); n=len(x); L=win_len//2
    w=np.hanning(2*L+1); F=n; tfr=np.zeros((F,n))
    for t in range(n):
        Lp=min(L,t,n-1-t)
        for tau in range(-Lp, Lp+1):
            t1=t+tau; t2=t-tau
            tfr[(tau)%F,t] += w[tau+L]*x[t1]*x[t2]
    tfr=np.fft.fft(tfr, axis=0); tfr=np.abs(tfr)
    return tfr[:F//2, :]
def sample_entropy(x, m=2, r=None):
    x=np.asarray(x, dtype=float); n=len(x)
    if r is None: r=0.2*np.std(x)+1e-12
    def _phi(m):
        c=0
        for i in range(n-m):
            for j in range(i+1, n-m):
                if np.max(np.abs(x[i:i+m]-x[j:j+m]))<=r: c+=1
        return c/(n-m+1e-12)
    return -np.log((_phi(m+1)+1e-12)/(_phi(m)+1e-12))
def band_energy(spec, f_lo, f_hi, fs):
    F,T=spec.shape; freqs=np.linspace(0, fs/2, F)
    idx=(freqs>=f_lo)&(freqs<=f_hi); return np.sum(spec[idx,:], axis=0)

def extract_features_from_residual(
    signal: np.ndarray,
    dt: float,
    spec_method: str = "stft",
    use_tf_entropy: bool | None = None,
    use_sampen: bool = True,
) -> np.ndarray:
    """
    给定一段一维信号序列 signal (可以是残差，也可以是原始加速度) 及采样间隔 dt，提取特征。
    
    该函数用于第三章"喂入分类器前"的统一特征提取。
    spec_method 可以在命令行参数中切换，方便做 STFT vs PWVD 的对比实验。
    
    当 spec_method="pwvd" 且调用时未显式指定 use_tf_entropy 时，函数会默认启用时频熵。

    特征包括：
    1. 频带能量 (Low, Mid, High) - 基于 STFT 或 PWVD
    2. 样本熵 (Sample Entropy) - 可选
    3. 时频熵 (TF Entropy) - 可选
    4. 窗口均值 (DC 分量) - [新增]

    Parameters
    ----------
    signal : np.ndarray
        一维信号序列 (Residual or Raw Acceleration)。
    dt : float
        采样间隔 (s)。
    spec_method : str
        "stft" (默认) 或 "pwvd"。
    use_tf_entropy : bool | None
        是否计算时频熵 (Shannon Entropy of T-F distribution)。
        若为 None，则当 spec_method="pwvd" 时自动为 True，否则为 False。
    use_sampen : bool
        是否计算样本熵 (Sample Entropy)。

    Returns
    -------
    np.ndarray
        一维特征向量。
    """
    r = np.asarray(signal, dtype=float)
    fs = 1.0 / dt
    
    # Auto-enable TF Entropy for PWVD if not specified
    if use_tf_entropy is None:
        use_tf_entropy = (spec_method.lower() == "pwvd")
    
    # Ensure signal is long enough for the window
    win_len = 256
    if len(r) < win_len:
        # Pad with zeros to match win_len
        pad_width = win_len - len(r)
        r = np.pad(r, (0, pad_width), mode='constant')
    
    # 1. Time-Frequency Representation
    if spec_method == "pwvd":
        # PWVD
        spec = pwvd(r, win_len=win_len)
    else:
        # STFT (Default)
        spec = stft(r, win_len=win_len, hop=64)
        
    # Check if spec is valid
    if spec.ndim != 2:
        # Fallback if STFT failed to produce 2D array (e.g. extremely short signal)
        # Create a dummy spectrum to avoid crash, though padding should prevent this
        spec = np.zeros((win_len // 2 + 1, 1))
        
    # 2. Band Energy
    # Low: [0, fs/6], Mid: [fs/6, fs/3], High: [fs/3, fs/2]
    f_nyq = fs / 2.0
    f_split1 = fs / 6.0
    f_split2 = fs / 3.0
    
    E_low_seq = band_energy(spec, 0, f_split1, fs)
    E_mid_seq = band_energy(spec, f_split1, f_split2, fs)
    E_high_seq = band_energy(spec, f_split2, f_nyq, fs)
    
    E_low = float(np.sum(E_low_seq))
    E_mid = float(np.sum(E_mid_seq))
    E_high = float(np.sum(E_high_seq))
    
    feats = [E_low, E_mid, E_high]
    
    # 3. Time-Frequency Entropy (Optional)
    if use_tf_entropy:
        # P_norm = |Spec|^2 / sum(|Spec|^2)
        P = spec**2
        P_sum = np.sum(P)
        if P_sum > 1e-12:
            P_norm = P / P_sum
            # Shannon Entropy: -sum(p * log(p))
            # Add epsilon to avoid log(0)
            ent_tf = -np.sum(P_norm * np.log(P_norm + 1e-12))
        else:
            ent_tf = 0.0
        feats.append(float(ent_tf))
        
    # 4. Sample Entropy (Optional)
    if use_sampen:
        # Use existing sample_entropy function
        # r tolerance usually 0.2 * std
        se = sample_entropy(r, m=2, r=0.2 * np.std(r) + 1e-12)
        feats.append(float(se))

    # 5. DC Component (Window Mean)
    dc = float(np.mean(signal))
    feats.append(dc)
        
    return np.array(feats, dtype=float)
