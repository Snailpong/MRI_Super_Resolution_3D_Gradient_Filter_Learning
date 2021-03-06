import numpy as np
import cupy as cp
import time
from math import atan2, floor, pi, ceil, isnan, sqrt, acos, pi
import random
from numba import jit, njit

import filter_constant as C


# Quantization procedure to get the optimized strength and coherence boundaries
# @njit
def quantization_border(im, im_GX, im_GY, im_GZ, patchNumber, w, quantization, instance):
    H, W, D = im_GX.shape
    for i1 in range(C.PATCH_HALF, H - C.PATCH_HALF):
        for j1 in range(C.PATCH_HALF, W - C.PATCH_HALF):
            for k1 in range(C.PATCH_HALF, D - C.PATCH_HALF):

                if random.random() > 0.2 or np.any(im[i1, j1, k1] == 0):
                    continue

                idxg = (slice(i1 - C.GRADIENT_HALF, i1 + C.GRADIENT_HALF + 1),
                        slice(j1 - C.GRADIENT_HALF, j1 + C.GRADIENT_HALF + 1),
                        slice(k1 - C.GRADIENT_HALF, k1 + C.GRADIENT_HALF + 1))

                patchX = im_GX[idxg]
                patchY = im_GY[idxg]
                patchZ = im_GZ[idxg]
                strength, coherence = grad(patchX, patchY, patchZ, w)

                quantization[patchNumber, 0] = strength
                quantization[patchNumber, 1] = coherence
                patchNumber += 1
    return quantization, patchNumber

@njit
def get_lamda_u(l1, l2, l3):
    lamda = l1
    # lamda = l1 + l2 + l3
    # u = (sqrt(l1) - sqrt(l2)) / (sqrt(l1) + sqrt(l2) + 1e-16)
    u = (sqrt(l1) - sqrt(l2) - sqrt(l3)) / (sqrt(l1) + sqrt(l2) + sqrt(l3))
    # u = sqrt(((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / (2 * (l1 ** 2 + l2 ** 2 + l3 ** 2)))
    # u = l1 / sqrt(l2 ** 2 + l3 ** 2)
    return lamda, u


@njit
def get_hash(patchX, patchY, patchZ, weight, stre, cohe):
    G = np.vstack((patchX.ravel(), patchY.ravel(), patchZ.ravel())).T
    x = G.T @ weight @ G
    w, v = np.linalg.eig(x)

    index = w.argsort()[::-1]
    [l1, l2, l3] = w[index]
    v = v[:, index]

    angle_p = atan2(v[1, 0], v[0, 0])
    angle_t = acos(v[2, 0] / (sqrt((v[0, 0]) ** 2 + v[1, 0] ** 2 + v[2, 0] ** 2) + 1e-16))

    if angle_p < 0:
        angle_p = angle_p + pi
        angle_t = pi - angle_t

    angle_p = min(max(int(angle_p / (pi / C.Q_ANGLE_P)), 0), C.Q_ANGLE_P - 1)
    angle_t = min(max(int(angle_t / (pi / C.Q_ANGLE_T)), 0), C.Q_ANGLE_T - 1)

    lamda, u = get_lamda_u(l1, l2, l3)

    lamda = np.searchsorted(stre, lamda)
    u = np.searchsorted(cohe, u)

    return angle_p, angle_t, lamda, u

@njit
def grad(patchX, patchY, patchZ, weight):
    G = np.vstack((patchX.ravel(), patchY.ravel(), patchZ.ravel())).T
    x = G.T @ weight @ G
    w, v = np.linalg.eig(x)

    index = w.argsort()[::-1]
    [l1, l2, l3] = w[index]

    lamda, u = get_lamda_u(l1, l2, l3)

    return lamda, u

def init_buckets(Q_TOTAL):
    patchS = [[] for j in range(C.Q_TOTAL)]
    xS = [[] for j in range(C.Q_TOTAL)]
    return patchS, xS

def chunk(lst, size):
    return list(map(lambda x: lst[x * size:x * size + size], list(range(0, ceil(len(lst) / size)))))


def train_qv(im_LR, im_HR, w, stre, cohe, Q, V, count):
    H, W, D = im_HR.shape
    im_GX, im_GY, im_GZ = np.gradient(im_LR)  # Calculate the gradient images

    xyz_range = [[x, y, z] for x in range(C.PATCH_HALF, H - C.PATCH_HALF)
                    for y in range(C.PATCH_HALF, W - C.PATCH_HALF)
                    for z in range(C.PATCH_HALF, D - C.PATCH_HALF)]
    sample_range = random.sample(xyz_range, len(xyz_range) // C.SAMPLE_RATE)
    point_list = chunk(sample_range, len(sample_range) // C.TRAIN_DIV + 1)

    for sample_idx, point_list1 in enumerate(point_list):
        print('\r{} / {}'.format(sample_idx + 1, len(point_list)), end='', flush=True)
        timer = time.time()
        patchS, xS = init_buckets(C.Q_TOTAL)

        for i1, j1, k1 in point_list1:

            # if np.any(im_HR[i1 + C.PATCH_HALF, j1 + C.PATCH_HALF, k1 + C.PATCH_HALF] == 0):
            if im_HR[i1, j1, k1] == 0:
                continue

            idxp = (slice(i1 - C.PATCH_HALF, i1 + C.PATCH_HALF + 1),
                    slice(j1 - C.PATCH_HALF, j1 + C.PATCH_HALF + 1),
                    slice(k1 - C.PATCH_HALF, k1 + C.PATCH_HALF + 1))

            idxg = (slice(i1 - C.GRADIENT_HALF, i1 + C.GRADIENT_HALF + 1),
                    slice(j1 - C.GRADIENT_HALF, j1 + C.GRADIENT_HALF + 1),
                    slice(k1 - C.GRADIENT_HALF, k1 + C.GRADIENT_HALF + 1))

            patch = im_LR[idxp]
            patchX = im_GX[idxg]
            patchY = im_GY[idxg]
            patchZ = im_GZ[idxg]

            angle_p, angle_t, lamda, u = get_hash(patchX, patchY, patchZ, w, stre, cohe)
            j = int(angle_p * C.Q_STRENGTH * C.Q_COHERENCE * C.Q_ANGLE_T + angle_t * C.Q_STRENGTH * C.Q_COHERENCE + lamda * C.Q_COHERENCE + u)

            patch1 = patch.reshape(-1)
            x1 = im_HR[i1, j1, k1]

            patchS[j].append(patch1)
            xS[j].append(x1)
            count[j] += 1

        print('\t{} s'.format(((time.time() - timer) * 1000 // 10) / 100), end='', flush=True)
        timer = time.time()

        for j in range(C.Q_TOTAL):
            if len(xS[j]) != 0:
                A = cp.array(patchS[j])
                b = cp.array(xS[j]).reshape(-1, 1)
                Q[j] += cp.dot(A.T, A)
                V[j] += cp.dot(A.T, b).reshape(-1)
                # Qa = cp.array(Q[j])
                # Va = cp.array(V[j])
                #
                # Qa += cp.dot(A.T, A)
                # Va += cp.dot(A.T, b).reshape(-1)
                #
                # Q[j] = Qa.get()
                # V[j] = Va.get()

        print('\tqv {} s'.format((time.time() - timer) * 100 // 10 / 10), end='', flush=True)

    return Q, V, count