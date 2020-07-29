import os
import glob
import time

import numpy as np
from numba import jit, njit, prange

import nibabel as nib

from crop_black import crop_black
from filter_constant import *
from filter_func import *
from get_lr import *
from hashtable import hashtable
from matrix_compute import *


dataDir="./test/*"

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]

# Preprocessing normalized Gaussian matrix W for hashkey calculation
weight = get_normalized_gaussian()

h = np.load("./arrays/lowR4.npy")
h = np.array(h)

for idx, file in enumerate(fileList):
    print(idx+1, "/", len(fileList), "\t", file)

    # Load NIfTI Image
    mat_file = nib.load(file)
    mat = np.array(mat_file.dataobj)
    mat = mat[:, :-1, :]
    HR = mat

    # Make LR
    print('Making LR...', end='', flush=True)
    #LR = get_lr_kspace(HR)
    LR = get_lr_interpolation(HR)

    ni_img = nib.Nifti1Image(LR, np.eye(4))
    nib.save(ni_img, str(idx) + 'LR.nii.gz')

    [x_use, y_use, z_use] = crop_black(HR)
    print("x: ", x_use, "y: ", y_use, "z: ", z_use)


    [Lgx, Lgy, Lgz] = np.gradient(LR)

    LR = np.array(LR)
    LRDirect = np.zeros((LR.shape[0], LR.shape[1], LR.shape[2]))

    xRange = range(max(FILTER_HALF, x_use[0] - FILTER_HALF), min(LR.shape[0] - FILTER_HALF, x_use[1] + FILTER_HALF))
    yRange = prange(max(FILTER_HALF, y_use[0] - FILTER_HALF), min(LR.shape[1] - FILTER_HALF, y_use[1] + FILTER_HALF))
    zRange = prange(max(FILTER_HALF, z_use[0] - FILTER_HALF), min(LR.shape[2] - FILTER_HALF, z_use[1] + FILTER_HALF))


    start = time.time()

    for xP in xRange:
        print('\r{} / {}, last {} s '.format(xP - xRange[0], xRange[-1] - xRange[0], time.time() - start), end='', flush=True)
        start = time.time()

        for yP in yRange:
            for zP in zRange:
                patch = LR[xP - FILTER_HALF : xP + (FILTER_HALF + 1), yP - FILTER_HALF : yP + (FILTER_HALF + 1),
                        zP - FILTER_HALF : zP + (FILTER_HALF + 1)]

                if not np.any(patch):
                    continue

                gx = Lgx[xP - FILTER_HALF: xP + (FILTER_HALF + 1), yP - FILTER_HALF: yP + (FILTER_HALF + 1),
                        zP - FILTER_HALF: zP + (FILTER_HALF + 1)]
                gy = Lgy[xP - FILTER_HALF: xP + (FILTER_HALF + 1), yP - FILTER_HALF: yP + (FILTER_HALF + 1),
                        zP - FILTER_HALF: zP + (FILTER_HALF + 1)]
                gz = Lgz[xP - FILTER_HALF: xP + (FILTER_HALF + 1), yP - FILTER_HALF: yP + (FILTER_HALF + 1),
                    zP - FILTER_HALF: zP + (FILTER_HALF + 1)]

                [angle_p, angle_t, strength, coherence] = hashtable(gx, gy, gz, weight)

                j = angle_p * Q_ANGLE_T * Q_COHERENCE * Q_STRENGTH + angle_t * Q_COHERENCE * Q_STRENGTH + strength * Q_COHERENCE + coherence
                t = xP % 2 * 4 + yP % 2 * 2 + zP % 2
                A = patch.reshape((1, -1))
                hh = h[j][t].reshape((1, -1))
                LRDirect[xP][yP][zP] = max(np.matmul(hh, A.T)[0, 0], 0)

        

            
    ni_img = nib.Nifti1Image(LRDirect, np.eye(4))
    nib.save(ni_img, str(idx) + 'outputt2.nii.gz')

    HR_Blend = blend_image(LR, LRDirect, BLEND_THRESHOLD)
    ni_img = nib.Nifti1Image(HR_Blend, np.eye(4))
    nib.save(ni_img, str(idx) + 'outputt3.nii.gz')

print("Test is off")