import os
import glob

import numpy as np
from numba import jit

import scipy.sparse as sparse
from scipy.sparse.linalg import cg
from scipy.ndimage import zoom

import nibabel as nib
import matplotlib.pyplot as plt
import cv2

from hashTable import hashtable
from filterVariable import *
from getMask import getMask_test, crop_black
from util import *


dataDir="./test/*"
dataLRDir="./test_low/*"

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]
fileLRList = [file for file in glob.glob(dataLRDir) if file.endswith(".nii.gz")]

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weight = gaussian_3d((filter_length,filter_length,filter_length))
weight = np.diag(weight.ravel())
weight = np.array(weight, dtype=np.float32)

h = np.load("./filter_array/lowR4.npy")
h = np.array(h)

for idx, file in enumerate(fileList):
    print(idx+1, "/", len(fileList), "\t", file)

    #LR = np.array(nib.load(file).dataobj)


    # Load NIfTI Image
    mat_file = nib.load(file)
    mat = np.array(mat_file.dataobj)
    mat = mat[:, :-1, :]
    HR = mat

    # Dog-Sharpening
    #print("Sharpening...", end='', flush=True)
    #HR = dog_sharpener(HR)

    # Using k-space domain
    # mat_file2 = np.array(nib.load(fileLRList[idx]).dataobj)
    # LR = mat_file2 / np.max(mat)

    # Using Image domain
    print('Making LR...', end='', flush=True)
    LR = get_lr_kspace(HR)

    ni_img = nib.Nifti1Image(LR, np.eye(4))
    nib.save(ni_img, str(idx) + 'LR.nii.gz')



    [Lgx, Lgy, Lgz] = np.gradient(LR)

    LR = np.array(LR)
    LRDirect = np.zeros((LR.shape[0], LR.shape[1], LR.shape[2]))

    [x_use, y_use, z_use] = crop_black(LR)
    print("x: ", x_use, "y: ", y_use, "z: ", z_use)

    xRange = range(max(filter_half, x_use[0] - filter_half), min(LR.shape[0] - filter_half, x_use[1] + filter_half))
    yRange = range(max(filter_half, y_use[0] - filter_half), min(LR.shape[1] - filter_half, y_use[1] + filter_half))
    zRange = range(max(filter_half, z_use[0] - filter_half), min(LR.shape[2] - filter_half, z_use[1] + filter_half))

    # xRange = range(80,180)
    # yRange = range(105,205)
    # zRange = range(80,180)

    # xRange = range(60,200)
    # yRange = range(85,225)
    # zRange = range(60,200)

    for xP in xRange:
        for yP in yRange:

            print(xP - xRange[0], "/", xRange[-1] - xRange[0], '\t',
                  yP - yRange[0], "/", yRange[-1] - yRange[0], end='\r', flush=True)

            for zP in zRange:
                patch = LR[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
                        zP - filter_half : zP + (filter_half + 1)]

                if not np.any(patch):
                    continue

                gx = Lgx[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]
                gy = Lgy[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]
                gz = Lgz[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                    zP - filter_half: zP + (filter_half + 1)]

                [angle_p, angle_t, strength, coherence] = hashtable(gx, gy, gz, weight)

                j = angle_p * Qangle_t * Qcoherence * Qstrength + angle_t * Qcoherence * Qstrength + strength * Qcoherence + coherence
                t = xP % 2 * 4 + yP % 2 * 2 + zP % 2
                A = patch.reshape(1, -1)
                hh = h[j][t].reshape(1, -1)
                LRDirect[xP][yP][zP] = max(np.matmul(hh, A.T)[0, 0], 0)

            
    ni_img = nib.Nifti1Image(LRDirect, np.eye(4))
    nib.save(ni_img, str(idx) + 'outputt2_gg.nii.gz')

    HR_Blend = blend_image(LR.get(), LRDirect, 5)
    ni_img = nib.Nifti1Image(HR_Blend, np.eye(4))
    nib.save(ni_img, str(idx) + 'outputt3_gg.nii.gz')

print("Test is off")