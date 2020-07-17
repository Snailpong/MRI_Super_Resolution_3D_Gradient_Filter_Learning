import os
import glob

import numpy as np
import cupy as cp
from numba import jit

import scipy.sparse as sparse
from scipy.sparse.linalg import cg
from scipy.ndimage import zoom

import nibabel as nib
import matplotlib.pyplot as plt
import cv2
#from scipy import sparse

from hashTable import hashtable, hashtable_cupy
from getMask import crop_black
from util import *
from filterVariable import *

# Construct an empty matrix Q, V uses the corresponding LR and HR, h is the filter, three hashmaps are Angle, Strength, Coherence, t
# Q = cp.zeros((Q_total, filter_volume, filter_volume))
# V = cp.zeros((Q_total, filter_volume, 1))

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


for idx, file in enumerate(fileList):
    print(idx+1, "/", len(fileList), "\t", file)

    # Load NIfTI Image
    mat_file = nib.load(file)
    mat = np.array(mat_file.dataobj)
    mat = mat[:, :-1, :]

    # Normalized to [0, 1]
    HR = mat / np.max(mat)

    # Dog-Sharpening
    print("Sharpening...", end='', flush=True)
    HR = dog_sharpener(HR)

    # Using k-space domain
    #mat_file2 = np.array(nib.load(fileLRList[idx]).dataobj)
    #LR = mat_file2 / np.max(mat)

    # Downscale (bicububic interpolation)
    print("\rMaking LR...", end='', flush=True)
    downscaled_LR = zoom(HR, 0.5, order=2)

    # Upscale (bilinear interpolation)
    LR = zoom(downscaled_LR, 2, order=1)

    [Lgx, Lgy, Lgz] = np.gradient(LR)

    # Using Cupy
    # HR = np.array(HR)
    # LR = np.array(LR)

    [x_use, y_use, z_use] = crop_black(LR)
    print("x: ", x_use, "y: ", y_use, "z: ", z_use)

    # xRange = range(max(filter_half, x_use[0] - filter_half), min(LR.shape[0] - filter_half, x_use[1] + filter_half))
    # yRange = range(max(filter_half, y_use[0] - filter_half), min(LR.shape[1] - filter_half, y_use[1] + filter_half))
    # zRange = range(max(filter_half, z_use[0] - filter_half), min(LR.shape[2] - filter_half, z_use[1] + filter_half))

    # xRange = range(80,180)
    # yRange = range(105,205)
    # zRange = range(80,180)

    xRange = range(60,200)
    yRange = range(85,225)
    zRange = range(60,200)


    # Iterate over each pixel
    for xP in xRange:
        for yP in yRange:

            print('\r', xP - xRange[0], "/", xRange[-1] - xRange[0], '\t',
                yP - yRange[0], "/", yRange[-1] - yRange[0], end='', flush=True)

            for zP in zRange:
                patch = LR[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]

                # if(cp.max(LR) < 0.03):
                # continue

                gx = Lgx[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                     zP - filter_half: zP + (filter_half + 1)]
                gy = Lgy[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                     zP - filter_half: zP + (filter_half + 1)]
                gz = Lgz[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                     zP - filter_half: zP + (filter_half + 1)]

                # Computational characteristics
                angle_p, angle_t, strength, coherence = hashtable([gx, gy, gz], weight)

                # Compressed vector space
                j = angle_p * Qangle_t * Qcoherence * Qstrength + angle_t * Qcoherence * Qstrength + strength * Qcoherence + coherence
                t = xP % 2 * 4 + yP % 2 * 2 + zP % 2
                
                #A = cp.array(patch.ravel())
                A = np.matrix(patch.ravel())
                x = HR[xP, yP, zP]

                # Save the corresponding HashMap
                Q[j, t] += np.dot(A.T, A)
                V[j, t] += np.dot(A.T, x)

    #print(tT)

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
        # Train 8 * 24 * 3 * 3 filters for each pixel type and image feature
        #h[j] = cg(Q[j], V[j])[0]
        h[j,t] = sparse.linalg.cg(Q[j,t], V[j,t])[0]

print("Train is off")
np.save("./filter_array/lowR4", h)