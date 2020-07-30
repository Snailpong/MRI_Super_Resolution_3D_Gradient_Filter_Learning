import os
import glob
import time

import numpy as np
from numba import jit, njit, prange

import nibabel as nib

from crop_black import *
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

for idx, file in enumerate(fileList):
    print(idx+1, "/", len(fileList), "\t", file)

    # Load NIfTI Image
    mat_file = nib.load(file)
    HR = np.array(mat_file.dataobj)[:, :-1, :]
    HR = clipped_hr(HR)
    HR_max = HR.max()

    # Make LR
    print('Making LR...', end='', flush=True)
    #LR = get_lr_kspace(HR)
    LR = get_lr_interpolation(HR)

    ni_img = nib.Nifti1Image(LR, np.eye(4))
    nib.save(ni_img, str(idx) + 'LR.nii.gz')

    [Lgx, Lgy, Lgz] = np.gradient(LR)

    LR = np.array(LR)
    LRDirect = np.zeros((LR.shape[0], LR.shape[1], LR.shape[2]))

    xRange, yRange, zRange = get_range(HR)

    start = time.time()

    for xP in xRange:
        print('\r{} / {}, last {} s '.format(xP - xRange[0], xRange[-1] - xRange[0], time.time() - start), end='', flush=True)
        start = time.time()

        for yP in yRange:
            for zP in zRange:
                patch = LR[xP - FILTER_HALF: xP + (FILTER_HALF + 1), yP - FILTER_HALF: yP + (FILTER_HALF + 1),
                    zP - FILTER_HALF: zP + (FILTER_HALF + 1)]

                if not np.any(patch):
                    continue

                gx = Lgx[xP - GRAD_HALF: xP + (GRAD_HALF + 1), yP - GRAD_HALF: yP + (GRAD_HALF + 1),
                    zP - GRAD_HALF: zP + (GRAD_HALF + 1)]
                gy = Lgy[xP - GRAD_HALF: xP + (GRAD_HALF + 1), yP - GRAD_HALF: yP + (GRAD_HALF + 1),
                        zP - GRAD_HALF: zP + (GRAD_HALF + 1)]
                gz = Lgz[xP - GRAD_HALF: xP + (GRAD_HALF + 1), yP - GRAD_HALF: yP + (GRAD_HALF + 1),
                        zP - GRAD_HALF: zP + (GRAD_HALF + 1)]

                [angle_p, angle_t, strength, coherence] = hashtable(gx, gy, gz, weight)

                j = angle_p * Q_ANGLE_T * Q_COHERENCE * Q_STRENGTH + angle_t * Q_COHERENCE * Q_STRENGTH + strength * Q_COHERENCE + coherence
                t = xP % FACTOR * (FACTOR ** 2) + yP % FACTOR * FACTOR + zP % FACTOR

                AT = patch.reshape((1, -1))
                hh = h[j][t].reshape((-1, 1))
                LRDirect[xP][yP][zP] = np.dot(AT, hh)

    LRDirect = np.clip(LRDirect, 0, HR_max)
    ni_img = nib.Nifti1Image(LRDirect, np.eye(4))
    nib.save(ni_img, str(idx) + 'outputt2.nii.gz')

    HR_Blend = blend_image(LR, LRDirect, BLEND_THRESHOLD)
    ni_img = nib.Nifti1Image(HR_Blend, np.eye(4))
    nib.save(ni_img, str(idx) + 'outputt3.nii.gz')

print("Test is off")