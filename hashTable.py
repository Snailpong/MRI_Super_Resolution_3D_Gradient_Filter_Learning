# -*- coding: utf-8 -*-

import numpy as np
import cupy as cp
import math


def hashTable(gradient, Qangle_p, Qangle_t, Qstrength, Qcoherence):
    G = np.matrix((gradient[0].ravel(), gradient[1].ravel(), gradient[2].ravel())).T
    x = np.matmul(G.T, G)
    [eigenvalues, eigenvectors] = np.linalg.eig(x)
    eigensort_arg = np.argsort(eigenvalues)

    arg_first = eigensort_arg[2]
    arg_second = eigensort_arg[1]

    # For angle
    angle_p = math.atan2(eigenvectors[1, arg_first], eigenvectors[0, arg_first])
    if angle_p < 0:
        angle_p += math.pi

    angle_t = math.acos(eigenvectors[2, arg_first] / (np.linalg.norm(eigenvectors[:, arg_first]) + 0.0001))

    # For strength
    strength = eigenvalues.max() / (eigenvalues.sum() + 0.0001)

    # For coherence
    lamda1 = math.sqrt(eigenvalues.max())
    lamda2 = math.sqrt(eigenvalues[arg_second])
    coherence = np.abs((lamda1 - lamda2) / (lamda1 + lamda2 + 0.0001))

    # Quantization
    angle_p = int(angle_p / math.pi * Qangle_p - 0.0001)
    angle_t = int(angle_t / math.pi * Qangle_t - 0.0001)
    strength = int(strength * Qstrength - 0.0001)
    coherence = int(coherence * Qcoherence - 0.0001)

    return angle_p, angle_t, strength, coherence


def hashTable_cupy(gradient, Qangle_p, Qangle_t, Qstrength, Qcoherence):
    G = cp.array((gradient[0].ravel(), gradient[1].ravel(), gradient[2].ravel())).T
    x = cp.matmul(G.T, G)
    [eigenvalues, eigenvectors] = cp.linalg.eigh(x)

    # For angle
    angle_p = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    if angle_p < 0:
        angle_p += math.pi

    angle_t = math.acos(eigenvectors[2, 0] / (cp.linalg.norm(eigenvectors[:, 0]) + 0.0001))

    # For strength
    strength = eigenvalues.max() / (eigenvalues.sum() + 0.0001)

    # For coherence
    lamda1 = math.sqrt(eigenvalues.max())
    lamda2 = math.sqrt(max(eigenvalues.min(), 0))
    coherence = np.abs((lamda1 - lamda2) / (lamda1 + lamda2 + 0.0001))

    # Quantization
    angle_p = math.floor(angle_p / math.pi * Qangle_p - 0.0001)
    angle_t = math.floor(angle_t / math.pi * Qangle_t - 0.0001)
    strength = math.floor(strength * Qstrength - 0.0001)
    coherence = math.floor(coherence * Qcoherence - 0.0001)

    return int(angle_p), int(angle_t), int(strength), int(coherence)