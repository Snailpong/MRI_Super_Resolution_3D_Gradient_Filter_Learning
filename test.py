import numpy as np
#import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import cupy as cp
import os

from hashTable import hashTable
from cupyfunc import gradient
from cropBlack import cropBlack
from filterVariable import *


dataDir="./train"
dataLRDir="./train_low"

fileName = 'T1w_acpc_dc_restore_brain_100307.nii.gz'

LR = np.array(nib.load(dataLRDir + '/' + fileName).dataobj)
HR = np.array(nib.load(dataDir + '/' + fileName).dataobj)

h = np.load("lowR4.npy")
h = cp.array(h)
# h = np.load("Filters.npy")

[x_use, y_use, z_use] = cropBlack(LR)
print("x: ", x_use, "y: ", y_use, "z: ", z_use)

[Lgx, Lgy, Lgz] = np.gradient(LR)

# Upscaling
#LR = cv2.resize(mat,(0,0),fx=2,fy=2)

def make_high_resolution(LR, h):
    LR = cp.array(LR)
    LRDirect = cp.zeros((LR.shape[0], LR.shape[1], LR.shape[2]))
    for xP in range(x_use[0] + filter_half, x_use[1] - filter_half):
        #print(xP - filter_half, "/", (LR.shape[0] - 2 * filter_half - 1), end='\r', flush=True)
        for yP in range(y_use[0] + filter_half, y_use[1] - filter_half):

            print(xP - filter_half, "/", (LR.shape[0] - 2 * filter_half - 1), '\t', yP - filter_half, "/", (LR.shape[1] - 2 * filter_half - 1), end='\r', flush=True)

            #for zP in range(z_use[0] + filter_half+50, z_use[1] - filter_half-50):
            for zP in range(120, 140):
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
    return cp.asnumpy(LRDirect)

LRDirect = make_high_resolution(LR.copy(), h)
        
        
print("Test is off")


ni_img = nib.Nifti1Image(LRDirect, np.eye(4))
nib.save(ni_img, 'outputt2_gg.nii.gz')
        
# Show the result
#mat = cv2.imread("./train/a.jpg")
#mat = cv2.imread("./train/alp2.jpg")
#mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)

fig, axes = plt.subplots(ncols=2)
axes[0].imshow(LR[:, :, 130])
axes[0].set_title('ORIGIN')


#LR = cv2.resize(mat,(0,0),fx=2,fy=2)
#LRDirectImage = LR
#LRDirectImage[:,:,2] = LRDirect
axes[1].imshow(LRDirect[:, :, 130])
axes[1].set_title('RAISR')

fig.savefig("./fig3.png")
