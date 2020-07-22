import os
import glob
import time

import numpy as np
from numba import jit, prange

import scipy.sparse as sparse
from scipy.sparse.linalg import cg
from scipy.ndimage import zoom

import nibabel as nib
import matplotlib.pyplot as plt
import cv2
#from scipy import sparse

from hashTable import hashtable
from getMask import crop_black
from util import *
from filterVariable import *

# Construct an empty matrix Q, V uses the corresponding LR and HR, h is the filter, three hashmaps are Angle, Strength, Coherence, t
Q = np.zeros((Q_total, pixel_type, filter_volume, filter_volume))
V = np.zeros((Q_total, pixel_type, filter_volume, 1))
h = np.zeros((Q_total, pixel_type, filter_volume))

dataDir="./train/*"
dataLRDir="./train_low/*"

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]
fileLRList = [file for file in glob.glob(dataLRDir) if file.endswith(".nii.gz")]

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weight = gaussian_3d((filter_length,filter_length,filter_length))
weight = np.diag(weight.ravel())
weight = np.array(weight, dtype=np.float32)

start = time.time()

for idx, file in enumerate(fileList):
    print('\n[' + str(idx+1), '/', str(len(fileList)) + ']\t', file)

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
    #HR = dog_sharpener(HR)

    [Lgx, Lgy, Lgz] = np.gradient(LR)

    [x_use, y_use, z_use] = crop_black(LR)
    print("x: ", x_use, "y: ", y_use, "z: ", z_use)

    xRange = range(max(filter_half, x_use[0] - filter_half), min(LR.shape[0] - filter_half, x_use[1] + filter_half))
    yRange = range(max(filter_half, y_use[0] - filter_half), min(LR.shape[1] - filter_half, y_use[1] + filter_half))
    zRange = range(max(filter_half, z_use[0] - filter_half), min(LR.shape[2] - filter_half, z_use[1] + filter_half))

    total_yz = (yRange[-1] - yRange[0]) * (zRange[-1] - zRange[0])
    start = time.time()


    # Iterate over each pixel
    for xP in xRange:
        jtS = np.zeros((total_yz, 2), np.int16)
        xS = np.zeros((total_yz), np.float32)
        patchS = np.zeros((total_yz, filter_volume))


        print('{} / {}, last {} s '.format(xP - xRange[0], xRange[-1] - xRange[0], time.time() - start), end='', flush=True)
        start = time.time()
        times = 0

        for yP in prange(max(filter_half, y_use[0] - filter_half), min(LR.shape[1] - filter_half, y_use[1] + filter_half)):
            for zP in prange(max(filter_half, z_use[0] - filter_half), min(LR.shape[2] - filter_half, z_use[1] + filter_half)):
                patch = LR[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]

                if not np.any(patch):
                        continue

                gx = Lgx[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]
                gy = Lgy[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]
                gz = Lgz[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]

                # Computational characteristics
                angle_p, angle_t, strength, coherence = hashtable(gx, gy, gz, weight)

                # Compressed vector space
                j = angle_p * Qangle_t * Qcoherence * Qstrength + angle_t * Qcoherence * Qstrength + strength * Qcoherence + coherence
                t = xP % 2 * 4 + yP % 2 * 2 + zP % 2
                
                pk = patch.reshape((-1))
                x = HR[xP, yP, zP]

                jtS[times] = np.array([j, t])
                xS[times] = x
                patchS[times] = pk

                times += 1

        print('{}  \r'.format(time.time() - start), end='', flush=True)

        for i in prange(times):
            j, t = jtS[i]
            patchT = patchS[i].reshape(1, -1)

            #Q[j, t] += np.dot(patchT.T, patchT)
            ata_add(patchT, Q[j, t])
            V[j, t] += np.dot(patchT.T, xS[i])



if str(type(Q)) == '<class \'cupy.core.core.ndarray\'>':
    Q = Q.get()
    V = V.get()

np.save("./Q", Q)
np.save("./V", V)


print("\nComputing H...")
# Set the train step

for j in range(Q_total):
    for t in range(pixel_type):
        print(j, "/", Q_total, end='\r', flush=True)
        h[j,t] = sparse.linalg.cg(Q[j,t], V[j,t])[0]

print('Train is off in {} minutes'.format((time.time() - start) // 60))
np.save('./filter_array/lowR4', h)