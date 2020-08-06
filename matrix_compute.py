import numpy as np
import cupy as cp
import math
import time

from numba import jit, njit, cuda, prange, vectorize

from scipy.sparse.linalg import cg

import filter_constant as C

@njit
def get_patch(LR, xP, yP, zP):
    return LR[xP - C.FILTER_HALF: xP + (C.FILTER_HALF + 1),
              yP - C.FILTER_HALF: yP + (C.FILTER_HALF + 1),
              zP - C.FILTER_HALF: zP + (C.FILTER_HALF + 1)]

@njit
def get_gxyz(Lgx, Lgy, Lgz, xP, yP, zP):
    gx = Lgx[xP - C.GRAD_HALF: xP + (C.GRAD_HALF + 1),
             yP - C.GRAD_HALF: yP + (C.GRAD_HALF + 1),
             zP - C.GRAD_HALF: zP + (C.GRAD_HALF + 1)]
    gy = Lgy[xP - C.GRAD_HALF: xP + (C.GRAD_HALF + 1),
             yP - C.GRAD_HALF: yP + (C.GRAD_HALF + 1),
             zP - C.GRAD_HALF: zP + (C.GRAD_HALF + 1)]
    gz = Lgz[xP - C.GRAD_HALF: xP + (C.GRAD_HALF + 1),
             yP - C.GRAD_HALF: yP + (C.GRAD_HALF + 1),
             zP - C.GRAD_HALF: zP + (C.GRAD_HALF + 1)]
    return gx, gy, gz

@njit
def get_bucket(angle_p, angle_t, strength, coherence):
    return angle_p * C.Q_ANGLE_T * C.Q_COHERENCE * C.Q_STRENGTH + \
           angle_t * C.Q_COHERENCE * C.Q_STRENGTH + \
           strength * C.Q_COHERENCE + \
           coherence

def add_qv_jt(patchSa, xSa, Qa, Va, j, t):
    A = cp.array(patchSa)
    b = cp.array(xSa).reshape(-1, 1)

    Qa = cp.array(Qa)
    Va = cp.array(Va)

    Qa += cp.dot(A.T, A)
    Va += cp.dot(A.T, b)

    return Qa.get(), Va.get()

def compute_h(Q, V):
    h = np.zeros((C.Q_TOTAL, C.PIXEL_TYPE, C.FILTER_VOL))

    print("\rComputing H...   ")
    start = time.time()
    for j in range(Q_TOTAL):
        for t in range(PIXEL_TYPE):
            print(j * C.PIXEL_TYPE + t, "/", C.Q_TOTAL * C.PIXEL_TYPE, end='\r', flush=True)
            h[j, t] = cg(Q[j, t], V[j, t])[0]

    np.save(C.H_FILE, h)
    print('Computing H is off in {} minutes'.format((time.time() - start) // 60))