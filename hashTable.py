import numpy as np
import cupy as cp
import time
from math import atan2, floor, pi, ceil, isnan, sqrt, acos, pi
import random
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
        rl1, rl2 = sqrt(lamb[0]), sqrt(lamb[1])
        coh2 = (rl1 - rl2) / (rl1 + rl2 + 0.0001)
        return coh2
    @njit
    def fa(lamb):
        fa= sqrt(((lamb[0]-lamb[1])**2+(lamb[1]-lamb[2])**2+(lamb[0]-lamb[2])**2) \
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


# Quantization procedure to get the optimized strength and coherence boundaries
@njit
def quantization_border(im, im_GX, im_GY, im_GZ, patchNumber, quantization):
    H, W, D = im_GX.shape
    for i1 in range(H - 2 * C.PATCH_HALF):
        print(i1, patchNumber)
        for j1 in range(W - 2 * C.PATCH_HALF):
            for k1 in range(D - 2 * C.PATCH_HALF):

                if random.random() > 0.2 or im[i1 + C.PATCH_HALF, j1 + C.PATCH_HALF, k1 + C.PATCH_HALF] == 0:
                    continue

                idx1 = (slice(i1+1, (i1 + 2 * C.GRADIENT_HALF + 2)), slice(j1+1, (j1 + 2 * C.GRADIENT_HALF + 2)),
                        slice(k1+1, (k1 + 2 * C.GRADIENT_HALF + 2)))

                patchX = im_GX[idx1]
                patchY = im_GY[idx1]
                patchZ = im_GZ[idx1]
                strength, coherence = grad(patchX, patchY, patchZ)

                quantization[patchNumber, 0] = strength
                quantization[patchNumber, 1] = coherence
                patchNumber += 1
    return quantization, patchNumber


@njit
def get_hash(gx, gy, gz, stre, cohe):
    G = np.vstack((gx.ravel(), gy.ravel(), gz.ravel())).T
    x = G.T @ C.G_WEIGHT @ G
    w, v = np.linalg.eig(x)

    index = w.argsort()[::-1]
    [l1, l2, l3] = np.abs(w[index])
    v = np.real(v[:, index])

    angle_p = atan2(v[1, 0], v[0, 0])
    angle_t = acos(v[2, 0] / (sqrt((v[0, 0]) ** 2 + v[1, 0] ** 2 + v[2, 0] ** 2) + 1e-16))

    if angle_p < 0:
        angle_p = angle_p + pi
        angle_t = pi - angle_t

    angle_p = min(max(int(angle_p / (pi / C.Q_ANGLE_P)), 0), C.Q_ANGLE_P - 1)
    angle_t = min(max(int(angle_t / (pi / C.Q_ANGLE_T)), 0), C.Q_ANGLE_T - 1)

    lamda = l1
    u = (sqrt(l1) - sqrt(l2)) / (sqrt(l1) + sqrt(l2) + 1e-16)

    # lamda = l1 + l2 + l3
    # u = sqrt(((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / (2 * (l1 ** 2 + l2 ** 2 + l3 ** 2)))

    lamda = np.searchsorted(stre, lamda)
    u = np.searchsorted(cohe, u)

    return angle_p, angle_t, lamda, u


@njit
def grad(patchX, patchY, patchZ):
    G = np.vstack((patchX.ravel(), patchY.ravel(), patchZ.ravel())).T
    x = G.T @ C.G_WEIGHT @ G
    w, v = np.linalg.eig(x)

    index = w.argsort()[::-1]
    [l1, l2, l3] = w[index]

    # lamda = l1 + l2 + l3
    # u = sqrt(((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / (2 * (l1 ** 2 + l2 ** 2 + l3 ** 2)))

    lamda = l1
    u = (sqrt(l1) - sqrt(l2)) / (sqrt(l1) + sqrt(l2) + 1e-16)

    return lamda, u


def train_qv(im_LR, im_HR, stre, cohe, Q, V):
    im_GX, im_GY, im_GZ = np.gradient(im_LR)  # Calculate the gradient images
    for t in range(C.R ** 3):
        Q, V, mark = train_qv_type(t, im_LR, im_HR, im_GX, im_GY, im_GZ, stre, cohe, Q, V)
    return Q, V, mark


def init_buckets():
    patchS = [[] for j in range(C.Q_TOTAL)]
    xS = [[] for j in range(C.Q_TOTAL)]
    return patchS, xS


def train_qv_type(t, im_LR, im_HR, im_GX, im_GY, im_GZ, stre, cohe, Q, V):
    H, W, D = im_HR.shape
    xd = (t // (C.R * C.R)) % C.R
    yd = (t // C.R) % C.R
    zd = t % C.R

    patchS, xS = init_buckets()
    timer = time.time()

    for i1 in range(xd, H - 2 * C.PATCH_HALF, C.R):
        print('\r{} / {}    {} / {}    {} s'.format(t+1, C.R ** 3, i1 // C.R, (H - 2 * C.PATCH_HALF) // C.R, ((time.time() - timer) * 1000 // 10) / 100), end='')
        timer = time.time()

        for j1 in range(yd, W - 2 * C.PATCH_HALF, C.R):
            for k1 in range(zd, D - 2 * C.PATCH_HALF, C.R):

                if random.random() > C.SAMPLE_RATE or im_HR[i1 + C.PATCH_HALF, j1 + C.PATCH_HALF, k1 + C.PATCH_HALF] == 0:
                    continue

                idx1 = (slice(i1, (i1 + 2 * C.PATCH_HALF + 1)), slice(j1, (j1 + 2 * C.PATCH_HALF + 1)),
                        slice(k1, (k1 + 2 * C.PATCH_HALF + 1)))

                patch = im_LR[idx1]

                idx2 = (slice(i1+1, (i1 + 2 * C.GRADIENT_HALF + 2)), slice(j1+1, (j1 + 2 * C.GRADIENT_HALF + 2)),
                        slice(k1+1, (k1 + 2 * C.GRADIENT_HALF + 2)))

                gx = im_GX[idx2]
                gy = im_GY[idx2]
                gz = im_GZ[idx2]

                angle_p, angle_t, lamda, u = get_hash(gx, gy, gz, stre, cohe)
                j = int(angle_p * C.Q_STRENGTH * C.Q_COHERENCE * C.Q_ANGLE_T + angle_t * C.Q_STRENGTH * C.Q_COHERENCE + lamda * C.Q_COHERENCE + u)

                patch1 = patch.reshape(-1)
                x1 = im_HR[i1 + C.PATCH_HALF, j1 + C.PATCH_HALF, k1 + C.PATCH_HALF]

                patchS[j].append(patch1)
                xS[j].append(x1)

    timer = time.time()

    for j in range(C.Q_TOTAL):
        if len(xS[j]) != 0:
            A = cp.array(patchS[j])
            b = cp.array(xS[j]).reshape(-1, 1)
            Qa = cp.array(Q[j])
            Va = cp.array(V[j])

            Qa += cp.dot(A.T, A)
            Va += cp.dot(A.T, b).reshape(-1)

            Q[j] = Qa.get()
            V[j] = Va.get()

    print('   qv {} s'.format((time.time() - timer) * 100 // 10 / 10), end='')

    return Q, V


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
    angle_p = atan2(vy, vx)
    angle_t = acos(vz / sqrt((vx**2+vy**2+vz**2) + 1e-10))

    if angle_p < 0:
        angle_p += pi
        angle_t = pi - angle_t

    # Quantization
    angle_p = int(angle_p / pi * C.Q_ANGLE_P - 0.0001)
    angle_t = int(angle_t / pi * C.Q_ANGLE_T - 0.0001)

    return angle_p, angle_t, lamb


@njit
def geometric_quantitization(gx, gy, gz, weight):
    global strength_func, coherence_func

    angle_p, angle_t, lamb = get_lambda_angle(gx, gy, gz, weight)
    strength = strength_func(lamb)
    coherence = coherence_func(lamb)

    return angle_p, angle_t, strength, coherence