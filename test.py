import os
import glob

import numpy as np
import cupy as cp

import scipy.sparse as sparse
from scipy.sparse.linalg import cg

import nibabel as nib
import matplotlib.pyplot as plt
import cv2

from hashTable import hashTable
from filterVariable import *
from getMask import getMask_test, cropBlack


dataDir="./test/*"
dataLRDir="./test_low/*"

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]
fileLRList = [file for file in glob.glob(dataLRDir) if file.endswith(".nii.gz")]

h = np.load("./filter_array/lowR4.npy")
h = cp.array(h)

for idx, file in enumerate(fileLRList):
    print(idx+1, "/", len(fileList), "\t", file)

    LR = np.array(nib.load(file).dataobj)

    [Lgx, Lgy, Lgz] = np.gradient(LR)

    LR = cp.array(LR)
    LRDirect = cp.zeros((LR.shape[0], LR.shape[1], LR.shape[2]))

    #[x_use, y_use, z_use] = crop_black(LR)
    #print("x: ", x_use, "y: ", y_use, "z: ", z_use)

    #xRange = range(max(filter_half, x_use[0] - filter_half), min(LR.shape[0] - filter_half, x_use[1] + filter_half))
    #yRange = range(max(filter_half, y_use[0] - filter_half), min(LR.shape[1] - filter_half, y_use[1] + filter_half))
    #zRange = range(max(filter_half, z_use[0] - filter_half), min(LR.shape[2] - filter_half, z_use[1] + filter_half))

    xRange = range(100, 200)
    yRange = range(100, 200)
    zRange = range(100, 200)

    for xP in xRange:
        for yP in yRange:

            print(xP - xRange[0], "/", xRange[-1] - xRange[0], '\t',
                  yP - yRange[0], "/", yRange[-1] - yRange[0], end='\r', flush=True)

            for zP in zRange:
                try:
                    patch = LR[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
                            zP - filter_half : zP + (filter_half + 1)]
                    gx = Lgx[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                            zP - filter_half: zP + (filter_half + 1)]
                    gy = Lgy[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                            zP - filter_half: zP + (filter_half + 1)]
                    gz = Lgz[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]

                    [angle_p, angle_t, strength, coherence] = hashtable([gx, gy, gz])

                    j = angle_p * Qangle_t * Qcoherence * Qstrength + angle_t * Qcoherence * Qstrength + strength * Qcoherence + coherence
                    A = patch.reshape(1, -1)

                    hh = h[j].reshape(1, -1)
                    LRDirect[xP][yP][zP] = max(cp.matmul(hh, A.T)[0, 0], 0)
                except:
                    pass

    LRDirect = LRDirect.get()
            
    ni_img = nib.Nifti1Image(LRDirect, np.eye(4))
    nib.save(ni_img, str(idx) + 'outputt2_gg.nii.gz')

print("Test is off")