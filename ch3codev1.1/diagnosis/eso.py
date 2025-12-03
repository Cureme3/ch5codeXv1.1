# -*- coding: utf-8 -*-
import numpy as np
def eso_step(x, y, dt, L):
    A = np.array([[1,0,dt,0],
                  [0,1,0,dt],
                  [0,0,1,0],
                  [0,0,0,1]], dtype=float)
    C = np.array([[1,0,0,0],
                  [0,1,0,0]], dtype=float)
    x_pred = A @ x
    y_pred = C @ x_pred
    e = y - y_pred
    x_new = x_pred + (L @ e)
    return x_new, e
def run_eso(axz_meas, dt=0.01, L=None):
    n = axz_meas.shape[0]
    x=np.zeros(4)
    if L is None:
        L = np.array([[0.2,0.0],[0.0,0.2],[1.0,0.0],[0.0,1.0]], dtype=float)
    x_hist=np.zeros((n,4)); e_hist=np.zeros((n,2))
    for i in range(n):
        x, e = eso_step(x, axz_meas[i], dt, L)
        x_hist[i]=x; e_hist[i]=e
    return e_hist, x_hist[:,2:4], x_hist
