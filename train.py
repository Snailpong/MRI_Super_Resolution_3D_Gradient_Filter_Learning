import glob
import os
import pickle
import time
import random

import cupy as cp
import numpy as np
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
    if fileName in finished_files:
        continue
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

    x_range = range(max(FILTER_HALF, x_use[0] - FILTER_HALF), min(LR.shape[0] - FILTER_HALF, x_use[1] + FILTER_HALF))
    y_range = range(max(FILTER_HALF, y_use[0] - FILTER_HALF), min(LR.shape[1] - FILTER_HALF, y_use[1] + FILTER_HALF))
    z_range = range(max(FILTER_HALF, z_use[0] - FILTER_HALF), min(LR.shape[2] - FILTER_HALF, z_use[1] + FILTER_HALF))

    xyz_range = [(x,y,z) for x in x_range for y in y_range for z in z_range]
    sample_range = random.sample(xyz_range, len(xyz_range) // TRAIN_DIV)
    split_range = list(chunks(sample_range, len(sample_range) // TRAIN_STP - 1))

    start = time.time()

    patchS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]
    xS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]


    # Iterate over each pixel
    for ix, point in enumerate(split_range):

        start = time.time()

        for xP, yP, zP in point:
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

        print('\r{} / {}    last {} s '.format(ix, TRAIN_STP, '%.3f' % (time.time() - start)), end='', flush=True)

        # Compute Q, V
        for j in prange(Q_TOTAL):
            for t in prange(PIXEL_TYPE):
                if len(xS[j][t]) != 0:
                    A = cp.array(patchS[j][t])
                    b = cp.array(xS[j][t]).reshape(-1, 1)

                    Q[j, t] += cp.dot(A.T, A).get()
                    V[j, t] += cp.dot(A.T, b).get()

        patchS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]
        xS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]
        print('   QV', '%.3f' % (time.time() - start), 's', end='', flush=True)

    finished_files.append(file.split('/')[-1].split('\\')[-1])
    np.save("./arrays/Q", Q)
    np.save("./arrays/V", V)
    with open('./arrays/finished_files.pkl', 'wb') as f:
        pickle.dump(finished_files, f)


if str(type(Q)) == '<class \'cupy.core.core.ndarray\'>':
    Q = Q.get()
    V = V.get()


compute_h(Q, V)

