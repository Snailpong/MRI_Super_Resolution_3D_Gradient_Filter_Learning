import os
import glob
import time
import pickle

import numpy as np
import cupy as cp
from numba import jit, prange

import nibabel as nib

from crop_black import crop_black
from filter_constant import *
from filter_func import *
from get_lr import *
from hashtable import hashtable
from matrix_compute import *

# Construct an empty matrix Q, V uses the corresponding LR and HR
if os.path.isfile('./arrays/Q.npy') and os.path.isfile('./arrays/V.npy'):
    print('Importing exist arrays...', end=' ', flush=True)
    Q = np.load("./arrays/Q.npy")
    V = np.load("./arrays/V.npy")
    with open('./arrays/finished_files.pkl', 'rb') as f:
        finished_files = pickle.load(f)
    print('Done', flush=True)
    
else:
    Q = np.zeros((Q_TOTAL, PIXEL_TYPE, FILTER_VOL, FILTER_VOL))
    V = np.zeros((Q_TOTAL, PIXEL_TYPE, FILTER_VOL, 1))
    finished_files = []

dataDir="./train/*"

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]

# Preprocessing normalized Gaussian matrix W for hashkey calculation
weight = get_normalized_gaussian()

start = time.time()

for idx, file in enumerate(fileList):
    fileName = file.split('/')[-1].split('\\')[-1]
    print('[' + str(idx+1), '/', str(len(fileList)) + ']\t', fileName)

    # Load NIfTI Image
    HR = nib.load(file).dataobj[:, :-1, :]

    # Normalized to [0, 1]
    HR = HR / np.max(HR)

    # Using Image domain
    print('Making LR...', end='', flush=True)
    #LR = get_lr_kspace(HR)
    LR = get_lr_interpolation(HR)

    # Dog-Sharpening
    print('\rSharpening...', end='', flush=True)
    HR = dog_sharpener(HR)

    [Lgx, Lgy, Lgz] = np.gradient(LR)

    [x_use, y_use, z_use] = crop_black(HR)
    print("x: ", x_use, "y: ", y_use, "z: ", z_use)

    xRange = range(max(FILTER_HALF, x_use[0] - FILTER_HALF), min(LR.shape[0] - FILTER_HALF, x_use[1] + FILTER_HALF))
    yRange = prange(max(FILTER_HALF, y_use[0] - FILTER_HALF), min(LR.shape[1] - FILTER_HALF, y_use[1] + FILTER_HALF))
    zRange = prange(max(FILTER_HALF, z_use[0] - FILTER_HALF), min(LR.shape[2] - FILTER_HALF, z_use[1] + FILTER_HALF))

    # xRange = range(FILTER_HALF, LR.shape[0] - FILTER_HALF)
    # yRange = prange(FILTER_HALF, LR.shape[1] - FILTER_HALF)
    # zRange = prange(FILTER_HALF, LR.shape[2] - FILTER_HALF)

    start = time.time()

    patchS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]
    xS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]


    # Iterate over each pixel
    for xP in xRange:

        print('\r{} / {}    last {} s '.format(xP - xRange[0], xRange[-1] - xRange[0], '%.3f' % (time.time() - start)), end='', flush=True)
        start = time.time()

        for yP in yRange:
            for zP in zRange:
                patch = LR[xP - FILTER_HALF: xP + (FILTER_HALF + 1), yP - FILTER_HALF: yP + (FILTER_HALF + 1),
                        zP - FILTER_HALF: zP + (FILTER_HALF + 1)]

                if not np.any(patch):
                        continue

                gx = Lgx[xP - FILTER_HALF: xP + (FILTER_HALF + 1), yP - FILTER_HALF: yP + (FILTER_HALF + 1),
                        zP - FILTER_HALF: zP + (FILTER_HALF + 1)]
                gy = Lgy[xP - FILTER_HALF: xP + (FILTER_HALF + 1), yP - FILTER_HALF: yP + (FILTER_HALF + 1),
                        zP - FILTER_HALF: zP + (FILTER_HALF + 1)]
                gz = Lgz[xP - FILTER_HALF: xP + (FILTER_HALF + 1), yP - FILTER_HALF: yP + (FILTER_HALF + 1),
                        zP - FILTER_HALF: zP + (FILTER_HALF + 1)]

                # Computational characteristics
                angle_p, angle_t, strength, coherence = hashtable(gx, gy, gz, weight)

                # Compressed vector space
                j = angle_p * Q_ANGLE_T * Q_COHERENCE * Q_STRENGTH + angle_t * Q_COHERENCE * Q_STRENGTH + strength * Q_COHERENCE + coherence
                t = xP % 2 * 4 + yP % 2 * 2 + zP % 2
                
                pk = patch.reshape((-1))
                x = HR[xP, yP, zP]

                patchS[j][t].append(pk)
                xS[j][t].append(x)

        # Compute Q, V in 10 times
        if xP % 10 == 9 or xP == xRange[-1]:
            print('\t Computing Q, V... ', end='', flush=True)
            for j in prange(Q_TOTAL):
                for t in prange(PIXEL_TYPE):
                    if len(xS[j][t]) != 0:
                        A = cp.array(patchS[j][t])
                        b = cp.array(xS[j][t]).reshape(-1, 1)

                        Q[j, t] += cp.dot(A.T, A).get()
                        V[j, t] += cp.dot(A.T, b).get()

            patchS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]
            xS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]
            print('\r', ' ' * 30, 'last QV', '%.3f' % (time.time() - start), 's', end='', flush=True)

    finished_files.append(file.split('/')[-1].split('\\')[-1])
    np.save("./arrays/Q", Q)
    np.save("./arrays/V", V)
    with open('./arrays/finished_files.pkl', 'wb') as f:
        finished_files = pickle.load(f)


if str(type(Q)) == '<class \'cupy.core.core.ndarray\'>':
    Q = Q.get()
    V = V.get()


compute_h(Q, V)

