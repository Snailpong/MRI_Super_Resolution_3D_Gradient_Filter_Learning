import numpy as np
#import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import cupy as cp
import os
import glob

from hashTable import hashTable
from filterVariable import *
from getMask import getMask_test


dataDir="./train/*"
dataLRDir="./train_low/*"

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]
fileLRList = [file for file in glob.glob(dataLRDir) if file.endswith(".nii.gz")]

h = np.load("./filter_array/lowR4.npy")
h = cp.array(h)

for idx, file in enumerate(fileLRList):
    print(idx+1, "/", len(fileList), "\t", file)

    LR = np.array(nib.load(file).dataobj)
    points = getMask_test(file, LR, filter_half)

    [Lgx, Lgy, Lgz] = np.gradient(LR)

    LR = cp.array(LR)
    LRDirect = cp.zeros((LR.shape[0], LR.shape[1], LR.shape[2]))

    point_size = len(points)

    for idxp, [xP, yP, zP] in enumerate(points):
        print(idxp, "/", point_size, end='\r')
        try:
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

            hh = h[j].reshape(1, -1)
            LRDirect[xP][yP][zP] = max(cp.matmul(hh, A.T)[0, 0], 0)
        except:
            pass

    LRDirect = LRDirect.get()
            
    ni_img = nib.Nifti1Image(LRDirect, np.eye(4))
    nib.save(ni_img, str(idx) + 'outputt2_gg.nii.gz')

print("Test is off")