import numpy as np
#import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import cupy as cp
import os
import glob

from hashTable import hashTable
from cropBlack import cropBlack
from filterVariable import *


dataDir="./test/*"
dataLRDir="./test_low/*"

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]
fileLRList = [file for file in glob.glob(dataLRDir) if file.endswith(".nii.gz")]

h = np.load("./filter_array/lowR4.npy")
h = cp.array(h)

for idx, file in enumerate(fileLRList):
    print(idx+1, "/", len(fileList), "\t", file)

    LR = np.array(nib.load(file).dataobj)

    [x_use, y_use, z_use] = cropBlack(LR)
    print("x: ", x_use, "y: ", y_use, "z: ", z_use)

    xRange = range(x_use[0] + filter_half, x_use[1] - filter_half)
    yRange = range(y_use[0] + filter_half, y_use[1] - filter_half)
    zRange = range(z_use[0] + filter_half, z_use[1] - filter_half)
    #zRange = range(125, 130)

    [Lgx, Lgy, Lgz] = np.gradient(LR)

    LR = cp.array(LR)
    LRDirect = cp.zeros((LR.shape[0], LR.shape[1], LR.shape[2]))
    for xP in xRange:
        #print(xP - filter_half, "/", (LR.shape[0] - 2 * filter_half - 1), end='\r', flush=True)
        for yP in yRange:

            print(xP - xRange[0], "/", xRange[-1] - xRange[0], '\t', 
                yP - yRange[0], "/", yRange[-1] - yRange[0], end='\r', flush=True)

            for zP in zRange:
                patch = LR[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
                        zP - filter_half : zP + (filter_half + 1)]
                gx = Lgx[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]
                gy = Lgy[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]
                gz = Lgz[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                    zP - filter_half: zP + (filter_half + 1)]

                [angle_p, angle_t, strength, coherence] = hashTable([gx, gy, gz], Qangle_p, Qangle_t, Qstrength, Qcoherence)

                j = angle_p * Qangle_t * Qcoherence * Qstrength + angle_t * Qcoherence * Qstrength + strength * Qcoherence + coherence
                A = patch.reshape(1, -1)

                # Calculate pixel type
                t = (xP % 2) * 4 + (yP % 2) * 2 + zP % 2

                hh = h[j, t].reshape(1, -1)
                LRDirect[xP][yP][zP] = cp.matmul(hh, A.T)[0, 0]

    LRDirect = LRDirect.get()
            
    ni_img = nib.Nifti1Image(LRDirect, np.eye(4))
    nib.save(ni_img, str(idx) + 'outputt2_gg.nii.gz')

print("Test is off")