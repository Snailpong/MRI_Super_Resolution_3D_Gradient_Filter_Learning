# -*- coding: utf-8 -*-

import numpy as np
import math
from numba import jit

from filterVariable import *

#@jit(nopython=True)
@jit
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
    #strength = eigenvalues.max() / (eigenvalues.sum() + 0.0001)
    strength = eigenvalues[0]

    # For coherence
    lamda1 = math.sqrt(eigenvalues[0])
    lamda2 = math.sqrt(eigenvalues[1])
    coherence = (lamda1 - lamda2) / (lamda1 + lamda2 + 0.0001)

    # Quantization
    angle_p = int(angle_p / math.pi * Qangle_p - 0.0001)
    angle_t = int(angle_t / math.pi * Qangle_t - 0.0001)
    # strength = int(strength * Qstrength - 0.0001)
    # coherence = int(coherence * Qcoherence - 0.0001)

    #print(eigenvalues.max())

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

    return angle_p, angle_t, int(strength), int(coherence)