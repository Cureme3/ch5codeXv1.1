# -*- coding: utf-8 -*-
"""
diagnosis.classifier — 提精度的简化训练（无 PSO）
提供：
- kmeans(X, K, iters=40, seed=0) -> (K, d)
- rbf_features(X, centers, sigma) -> (N, K)
- ridge_regression(Phi, Y, lam=1e-2) -> (M, C)
- ridge_regression_weighted(Phi, Y, w, lam) -> (M, C)
- simple_train_rbf(..., class_balance=True) -> (centers, sigma, lam, W, curve, (acc_te, (Itr, Ite)))
"""
from __future__ import annotations
import numpy as np

__all__ = [
    "kmeans",
    "rbf_features",
    "ridge_regression",
    "ridge_regression_weighted",
    "simple_train_rbf",
]

# -------------------------- KMeans --------------------------
def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = np.sum(A*A, axis=1, keepdims=True)
    B2 = np.sum(B*B, axis=1, keepdims=True).T
    return A2 + B2 - 2.0*(A @ B.T)

def kmeans(X: np.ndarray, K: int, iters: int=40, seed: int=0) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    assert 1 <= K <= N, "K must be in [1, N]"
    rs = np.random.RandomState(seed)
    C = X[rs.choice(N, size=K, replace=False)].copy()
    for _ in range(max(1, iters)):
        D = _pairwise_sqdist(X, C)          # (N,K)
        assign = np.argmin(D, axis=1)       # (N,)
        for k in range(K):
            I = (assign==k)
            if not np.any(I):
                j = int(rs.randint(0, N))
                C[k] = X[j] + 1e-6*rs.randn(d)
            else:
                C[k] = X[I].mean(axis=0)
    return C

# ---------------------- RBF 特征映射 ------------------------
def rbf_features(X: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    C = np.asarray(centers, dtype=float)
    s2 = float(sigma)**2 + 1e-12
    D = _pairwise_sqdist(X, C)              # (N,K)
    return np.exp(-0.5*D/s2)                # (N,K)

# ---------------------- 岭回归（闭式解） ----------------------
def ridge_regression(Phi: np.ndarray, Y: np.ndarray, lam: float=1e-2) -> np.ndarray:
    Phi = np.asarray(Phi, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim==1: Y = Y[:,None]
    M = Phi.shape[1]
    A = Phi.T @ Phi + lam*np.eye(M)
    B = Phi.T @ Y
    return np.linalg.solve(A, B)

def ridge_regression_weighted(Phi: np.ndarray, Y: np.ndarray, w: np.ndarray|None=None, lam: float=1e-2) -> np.ndarray:
    """样本加权（类别均衡用）。w 形状 (N,)；若 None 则退化为普通岭回归。"""
    Phi = np.asarray(Phi, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim==1: Y = Y[:,None]
    N, M = Phi.shape
    if w is None:
        return ridge_regression(Phi, Y, lam=lam)
    w = np.asarray(w, dtype=float).reshape(N,1)
    Phi_w = Phi * w
    A = Phi_w.T @ Phi + lam*np.eye(M)
    B = Phi_w.T @ Y
    return np.linalg.solve(A, B)

# ---------------------- 简化训练流水线 ----------------------
def _classwise_kmeans(X: np.ndarray, y: np.ndarray, Kc: int=3, iters: int=30, seed: int=0) -> np.ndarray:
    centers = []
    for c in range(int(np.max(y))+1):
        Xc = X[y==c]
        k  = min(Kc, max(1, len(Xc)))
        Cc = kmeans(Xc, K=k, iters=iters, seed=seed+c)
        centers.append(Cc)
    return np.vstack(centers)

def _robust_sigma(X: np.ndarray, centers: np.ndarray) -> float:
    D = np.linalg.norm(X[:,None,:] - centers[None,:,:], axis=2)  # (N,K)
    return float(np.median(np.min(D, axis=1)) + 1e-8)

def _class_weights(y: np.ndarray) -> np.ndarray:
    """返回每个样本的权重，按 1/freq 做均衡并归一化到均值=1。"""
    y = np.asarray(y, dtype=int)
    uniq, cnt = np.unique(y, return_counts=True)
    inv = {c: 1.0/max(1, n) for c, n in zip(uniq, cnt)}
    w = np.array([inv[int(c)] for c in y], dtype=float)
    return w / np.mean(w)

def _refine_sigma(Xtr, Ytr, centers, s0, steps=11, span=1.2, lam=1e-2, weights=None, verbose=True, Xval=None, Yval=None):
    """Refine sigma with optional validation set.

    If Xval/Yval provided, use validation error; otherwise use training error.
    """
    alphas = np.exp(np.linspace(-span, span, steps))
    best = 1.0; best_s = s0; curve = []
    ncls = int(np.max(Ytr)) + 1
    Ytr_oh = np.eye(ncls)[Ytr]

    use_val = (Xval is not None and Yval is not None)

    for i, a in enumerate(alphas, 1):
        s = float(s0 * a)
        Phi = rbf_features(Xtr, centers, s)
        W   = ridge_regression_weighted(Phi, Ytr_oh, w=weights, lam=float(lam))

        # Evaluate on validation set if available, else training set
        if use_val:
            Phi_eval = rbf_features(Xval, centers, s)
            pred = np.argmax(Phi_eval @ W, axis=1)
            err = float(np.mean(pred != Yval))
        else:
            pred = np.argmax(Phi @ W, axis=1)
            err = float(np.mean(pred != Ytr))

        if err <= best: best, best_s = err, s
        curve.append(best)
        if verbose:
            val_tag = "(val)" if use_val else "(train)"
            print(f"[refine σ] {i:02d}/{len(alphas)}  s={s:.4g}  best_err={best:.4f} {val_tag}")
    return best_s, np.asarray(curve, dtype=float)

def simple_train_rbf(
    X: np.ndarray, y: np.ndarray,
    per_class_k: int=6,
    refine_steps: int=11,
    refine_span: float=1.2,
    lam_grid=(1e-2, 5e-3, 2e-3, 1e-3, 8e-4, 5e-4),
    val_ratio: float=0.2,
    class_balance: bool=True,
    seed: int=0
):
    """返回: centers, sigma, lam, W, curve, (acc_te, (Itr, Ite))

    改进版: 使用 train/val/test 三路划分，验证集用于选择 (σ, λ)，
    curve 记录验证集上的累积最优误差。
    """
    rs = np.random.RandomState(seed)
    n  = len(y)
    I = np.arange(n); rs.shuffle(I)

    # 三路划分: 60% train, 20% val, 20% test
    n_test = int(0.2 * n)
    n_val = int(0.2 * n)
    n_train = n - n_test - n_val

    Itr = I[:n_train]
    Ival = I[n_train:n_train+n_val]
    Ite = I[n_train+n_val:]

    Xtr, Ytr = X[Itr], y[Itr]
    Xval, Yval = X[Ival], y[Ival]
    Xte, Yte = X[Ite], y[Ite]

    # 计算 RBF centers 和初始 sigma
    centers = _classwise_kmeans(Xtr, Ytr, Kc=per_class_k, iters=30, seed=seed)
    w_tr = _class_weights(Ytr) if class_balance else None
    s0 = _robust_sigma(Xtr, centers)

    # 构建 5×3 网格: σ_grid (5 values) × λ_grid (3 values) = 15 evaluations
    sigma_multipliers = [0.7, 0.85, 1.0, 1.15, 1.3]
    sigma_grid = [s0 * m for m in sigma_multipliers]
    # 从原 lam_grid 中选择 3 个有代表性的值
    if len(lam_grid) >= 3:
        lam_grid_small = [lam_grid[0], lam_grid[len(lam_grid)//2], lam_grid[-1]]
    else:
        lam_grid_small = list(lam_grid)

    print(f"[Grid Search] σ_grid={[f'{s:.3g}' for s in sigma_grid]}, λ_grid={[f'{l:.1e}' for l in lam_grid_small]}")

    # 网格搜索: 评估每个 (σ, λ) 组合，在验证集上计算误差
    ncls = int(np.max(y)) + 1
    Ytr_oh = np.eye(ncls)[Ytr]
    best_err = 1.0
    best_sigma = s0
    best_lam = lam_grid_small[0]
    curve = []  # 累积最优误差
    eval_count = 0

    for sig in sigma_grid:
        for lam in lam_grid_small:
            eval_count += 1
            # 在训练集上训练
            Phi_tr = rbf_features(Xtr, centers, sig)
            W_try = ridge_regression_weighted(Phi_tr, Ytr_oh, w=w_tr, lam=float(lam))

            # 在验证集上评估
            Phi_val = rbf_features(Xval, centers, sig)
            pred_val = np.argmax(Phi_val @ W_try, axis=1)
            err_val = float(np.mean(pred_val != Yval))

            # 更新最优
            if err_val < best_err:
                best_err = err_val
                best_sigma = sig
                best_lam = lam

            curve.append(best_err)
            print(f"[Grid {eval_count:02d}/15] σ={sig:.3g}, λ={lam:.1e}, val_err={err_val:.4f}, best={best_err:.4f}")

    # 用最优超参数在全部训练集上重训
    Phi_tr = rbf_features(Xtr, centers, best_sigma)
    W = ridge_regression_weighted(Phi_tr, Ytr_oh, w=w_tr, lam=float(best_lam))

    # 在测试集上最终评估
    pred_te = np.argmax(rbf_features(Xte, centers, best_sigma) @ W, axis=1)
    acc_te = float(np.mean(pred_te == Yte))

    return centers, float(best_sigma), float(best_lam), W, np.array(curve), (acc_te, (Itr, Ite))
