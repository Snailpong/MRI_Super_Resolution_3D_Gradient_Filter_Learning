import os
import glob
import numpy as np
import cv2
import cupy as cp
#import scipy as scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import cg
import nibabel as nib
import matplotlib.pyplot as plt
#from scipy import sparse

from hashTable import hashTable, hashTable_cupy
from getMask import getMask
from filterVariable import *

# Construct an empty matrix Q, V uses the corresponding LR and HR, h is the filter, three hashmaps are Angle, Strength, Coherence, t
Q = cp.zeros((Q_total, filter_volume, filter_volume))
V = cp.zeros((Q_total, filter_volume, 1))
h = np.zeros((Q_total, filter_volume))

dataDir="./train/*"
dataLRDir="./train_low/*"

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]
fileLRList = [file for file in glob.glob(dataLRDir) if file.endswith(".nii.gz")]

for idx, file in enumerate(fileList):
    print(idx+1, "/", len(fileList), "\t", file)

    mat_file = nib.load(file)
    mat = np.array(mat_file.dataobj)

    # Scale to 0-1
    HR = mat / np.max(mat)
    HR = cp.array(HR)

    points = getMask(file, HR, filter_half)

    mat_file2 = np.array(nib.load(fileLRList[idx]).dataobj)
    LR = mat_file2 / np.max(mat)
    [Lgx, Lgy, Lgz] = np.gradient(LR)

    LR = cp.array(LR)
    
    print("Training...")

    tT = cp.zeros((Q_total))

    # Iterate over each pixel
    point_size = len(points)
    for idxp, [xP, yP, zP] in enumerate(points):
        print(idxp, "/", point_size, end='\r')
        try:
            # Take patch
            patch = LR[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
                    zP - filter_half : zP + (filter_half + 1)]
            gx = Lgx[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
                    zP - filter_half : zP + (filter_half + 1)]
            gy = Lgy[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
                    zP - filter_half : zP + (filter_half + 1)]
            gz = Lgz[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
                zP - filter_half : zP + (filter_half + 1)]

            # Computational characteristics
            [angle_p, angle_t, strength, coherence] = hashTable([gx, gy, gz], Qangle_p, Qangle_t, Qstrength, Qcoherence)

            # Compressed vector space
            j = angle_p * Qangle_t * Qcoherence * Qstrength + angle_t * Qcoherence * Qstrength + strength * Qcoherence + coherence
            A = patch.reshape(1, -1)
            x = HR[xP][yP][zP]

            tT[j] += 1

            # Save the corresponding HashMap
            Q[j] += A * A.T
            V[j] += A.T * x
        except:
            pass

    #print(tT)

Q = Q.get()
V = V.get()

np.save("./Q", Q)
np.save("./V", V)


print("\nComputing H...")
# Set the train step

for j in range(Q_total):
    print(j, "/", Q_total, end='\r', flush=True)
    # Train 8 * 24 * 3 * 3 filters for each pixel type and image feature
    h[j] = cg(Q[j], V[j])[0]
    #h[j,t] = sparse.linalg.cg(Q[j,t],V[j,t])[0]

print("Train is off")
np.save("./filter_array/lowR4", h)