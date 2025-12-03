# ch4codexv1.1/src/opt/linearization.py

import numpy as np

def linearize_dynamics_3dof(dyn, t_ref, x_ref, u_ref):
    """
    计算 3-DOF 动力学的雅可比矩阵 A, B
    x: [rx, ry, rz, vx, vy, vz, m]
    u: [Tx, Ty, Tz]
    dx/dt = [v, (u/m + g), 0]
    """
    r = x_ref[0:3]
    m = x_ref[6]
    r_norm = np.linalg.norm(r)
    
    # --- 计算 A = df/dx ---
    A = np.zeros((7, 7))
    
    # dr/dv = I
    A[0:3, 3:6] = np.eye(3)
    
    # dv/dr = dg/dr (Gravity Gradient)
    # g = -mu * r / r^3
    # dg/dr = -mu/r^3 * (I - 3*r*r^T / r^2)
    mu = 3.986004418e14
    I3 = np.eye(3)
    r_rT = np.outer(r, r)
    A[3:6, 0:3] = -(mu / r_norm**3) * (I3 - 3 * r_rT / (r_norm**2))
    
    # dv/dm = -u / m^2 (推力加速度对质量的偏导)
    A[3:6, 6] = -u_ref / (m**2)
    
    # --- 计算 B = df/du ---
    B = np.zeros((7, 3))
    
    # dv/du = I / m
    B[3:6, 0:3] = I3 / m
    
    return A, B