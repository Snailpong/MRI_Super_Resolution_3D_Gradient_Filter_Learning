import numpy as np
import math
from numba import jit, njit

import filter_constant as C

strength_func, coherence_func = None, None

def determine_geometric_func():
    global strength_func, coherence_func

    @njit
    def lambda1(lamb):
        return lamb[0]
    @njit
    def trace(lamb):
        trace = lamb[0]+lamb[1]+lamb[2]
        if trace < 0.0001:
            return 0
        elif trace > 0.001:
            return 2
        else:
            return 1
        # trace_q = np.searchsorted([0.0001, 0.001], trace)
        # return trace_q
    @njit
    def coh2(lamb):
        rl1, rl2 = math.sqrt(lamb[0]), math.sqrt(lamb[1])
        coh2 = (rl1 - rl2) / (rl1 + rl2 + 0.0001)
        return coh_q
    @njit
    def fa(lamb):
        fa= math.sqrt(((lamb[0]-lamb[1])**2+(lamb[1]-lamb[2])**2+(lamb[0]-lamb[2])**2) \
            / (max(lamb[0]**2+lamb[1]**2+lamb[2]**2,1e-30)) / 2)
        if fa < 0.05:
            return 0
        elif fa > 0.1:
            return 2
        else:
            return 1
        # fa_q = np.searchsorted([0.05, 0.1], fa)
        # return fa_q

    if C.FEATURE_TYPE == 'lambda1_coh2':
        strength_func, coherence_func = lambda1, coh2
    elif C.FEATURE_TYPE == 'lambda1_fa':
        strength_func, coherence_func = lambda1, fa
    elif C.FEATURE_TYPE == 'trace_coh2':
        strength_func, coherence_func = trace, coh2
    elif C.FEATURE_TYPE == 'trace_fa':
        strength_func, coherence_func = trace, fa

@njit
def get_lambda_angle(gx, gy, gz, weight):
    G = np.vstack((gx.ravel(), gy.ravel(), gz.ravel())).T
    x0 = np.dot(G.T, weight)
    x = np.dot(x0, G)
    [eigenvalues, eigenvectors] = np.linalg.eig(x)

    idx = eigenvalues.argsort()[::-1]
    lamb = eigenvalues[idx]
    [vx, vy, vz] = eigenvectors[:, idx[0]]

    # For angle
    angle_p = math.atan2(vy, vx)
    angle_t = math.acos(vz / math.sqrt((vx**2+vy**2+vz**2) + 1e-10))

    if angle_p < 0:
        angle_p += math.pi
        angle_t = math.pi - angle_t

    # Quantization
    angle_p = int(angle_p / math.pi * C.Q_ANGLE_P - 0.0001)
    angle_t = int(angle_t / math.pi * C.Q_ANGLE_T - 0.0001)

    return angle_p, angle_t, lamb

@njit
def geometric_quantitization(gx, gy, gz, weight):
    global strength_func, coherence_func

    angle_p, angle_t, lamb = get_lambda_angle(gx, gy, gz, weight)
    strength = strength_func(lamb)
    coherence = coherence_func(lamb)

    return angle_p, angle_t, strength, coherence

@njit
def hashtable(gx, gy, gz, weight):
    G = np.vstack((gx.ravel(), gy.ravel(), gz.ravel())).T
    x0 = np.dot(G.T, weight)
    x = np.dot(x0, G)
    [eigenvalues, eigenvectors] = np.linalg.eig(x)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # For angle
    angle_p = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_t = math.acos(eigenvectors[2, 0] / (np.linalg.norm(eigenvectors[:, 0]) + 0.0001))

    if angle_p < 0:
        angle_p += math.pi
        angle_t = math.pi - angle_t

    # For strength
    strength = eigenvalues[0]

    # For coherence
    lamda1 = math.sqrt(eigenvalues[0])
    lamda2 = math.sqrt(eigenvalues[1])
    coherence = (lamda1 - lamda2) / (lamda1 + lamda2 + 0.0001)

    if strength < 0.0001:
        strength = 0
    elif strength > 0.001:
        strength = 2
    else:
        strength = 1
    if coherence < 0.25:
        coherence = 0
    elif coherence > 0.5:
        coherence = 2
    else:
        coherence = 1

    return int(angle_p), int(angle_t), int(strength), int(coherence)


@njit
def get_features(gx, gy, gz, weight):
    G = np.vstack((gx.ravel(), gy.ravel(), gz.ravel())).T
    x0 = np.dot(G.T, weight)
    x = np.dot(x0, G)
    [eigenvalues, eigenvectors] = np.linalg.eig(x)

    idx = eigenvalues.argsort()[::-1]
    [l1, l2, l3] = eigenvalues[idx]
    [vx, vy, vz] = eigenvectors[:, idx[0]]

    # For angle
    angle_p = math.atan2(vy, vx)
    angle_t = math.acos(vz / math.sqrt((vx**2+vy**2+vz**2) + 1e-10))

    if angle_p < 0:
        angle_p += math.pi
        angle_t = math.pi - angle_t

    strength = l1
    fa = math.sqrt(((l1-l2)**2+(l2-l3)**2+(l1-l3)**2) / (max(l1**2+l2**2+l3**2,1e-30)) / 2)

    # Quantization
    angle_p = int(angle_p / math.pi * C.Q_ANGLE_P - 0.0001)
    angle_t = int(angle_t / math.pi * C.Q_ANGLE_T - 0.0001)

    if strength < 0.0001:
        strength = 0
    elif strength > 0.001:
        strength = 2
    else:
        strength = 1

    if fa < 0.05:
        fa = 0
    elif fa > 0.1:
        fa = 2
    else:
        fa = 1

    return angle_p, angle_t, int(strength), int(fa)

@njit
def get_features2(gx, gy, gz, weight):
    G = np.vstack((gx.ravel(), gy.ravel(), gz.ravel())).T
    x0 = np.dot(G.T, weight)
    x = np.dot(x0, G)
    [eigenvalues, eigenvectors] = np.linalg.eig(x)

    idx = eigenvalues.argsort()[::-1]
    [l1, l2, l3] = eigenvalues[idx]
    [vx, vy, vz] = eigenvectors[:, idx[0]]

    # For angle
    angle_p = math.atan2(vy, vx)
    angle_t = math.acos(vz / math.sqrt((vx**2+vy**2+vz**2) + 1e-10))

    if angle_p < 0:
        angle_p += math.pi
        angle_t = math.pi - angle_t

    trace = l1 + l2 + l3
    fa = math.sqrt(((l1-l2)**2+(l2-l3)**2+(l1-l3)**2)/(max(l1**2+l2**2+l3**2,1e-30))/2)

    # Quantization
    angle_p = int(angle_p / math.pi * C.Q_ANGLE_P - 0.0001)
    angle_t = int(angle_t / math.pi * C.Q_ANGLE_T - 0.0001)

    if trace < 0.0001:
        trace = 0
    elif trace > 0.001:
        trace = 2
    else:
        trace = 1

    if fa < 0.05:
        fa = 0
    elif fa > 0.1:
        fa = 2
    else:
        fa = 1

    return angle_p, angle_t, int(trace), int(fa)