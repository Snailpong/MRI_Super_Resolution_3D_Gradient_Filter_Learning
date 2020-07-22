import os
import glob
import numpy as np
import cv2
import math
import scipy.linalg as ln
from numba import jit, cuda, prange
import nibabel as nib

import time
import random
import pickle

from hashTable import hashtable
from getMask import getMask, crop_black
from filterVariable import *
from util import *

RANGES = 68000

# Construct an empty matrix Q, V uses the corresponding LR and HR, h is the filter, three hashmaps are Angle, Strength, Coherence, t
# Q = cp.zeros((Q_total, filter_volume, filter_volume))
# V = cp.zeros((Q_total, filter_volume, 1))
# h = np.zeros((Q_total, filter_volume))

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weight = gaussian_3d((filter_length,filter_length,filter_length))
weight = np.diag(weight.ravel())
weight = np.array(weight, dtype=np.float32)

dataDir="./train/*"
dataLRDir="./train_low/*"

h = np.load("./filter_array/lowR4.npy")

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]
fileLRList = [file for file in glob.glob(dataLRDir) if file.endswith(".nii.gz")]

file = fileList[0]
mat_file = nib.load(file)
mat = np.array(mat_file.dataobj)[:, :-1, :]

scale_max = np.max(mat)

# Scale to 0-1
HR = mat / scale_max

print("\rMaking LR...", end='', flush=True)
start = time.time()
LR = get_lr_interpolation(HR)
print('lr time: ', time.time() - start)
start = time.time()

#HR = cp.array(HR)

print("Sharpening...", end='', flush=True)
#HR = dog_sharpener(HR)
print('sharpening time: ', time.time() - start)

[Lgx, Lgy, Lgz] = np.gradient(LR)


jtS = np.zeros((RANGES, 2), np.int16)
xS = np.zeros((RANGES), np.float32)
patchS = np.zeros((RANGES, filter_volume))


Q = np.zeros((Q_total, pixel_type, filter_volume, filter_volume))
V = np.zeros((Q_total, pixel_type, filter_volume, 1))
LRDirect = np.zeros((LR.shape[0], LR.shape[1], LR.shape[2]))


# sigma=0.85
# alpha=1.414
# r=15
# ksize=(3,3,3)
# input = HR

# G1 = gaussian_3d_blur(input, ksize, sigma)
# Ga1 = gaussian_3d_blur(input, ksize, sigma*alpha)
# D1 = add_weight(G1, 1+r, Ga1, -r, 0)
#
# G2 = gaussian_3d_blur(Ga1, ksize, sigma)
# Ga2 = gaussian_3d_blur(Ga1, ksize, sigma*alpha)
# D2 = add_weight(G2, 1+r, Ga2, -r, 0)
#
# G3 = gaussian_3d_blur(Ga2, ksize, sigma)
# Ga3 = gaussian_3d_blur(Ga2, ksize, sigma * alpha)
# D3 = add_weight(G3, 1+r, Ga3, -r, 0)
# result = D3

# result = dog_sharpener(input)
# print(result)
# print(result.shape)
#
#
# fig = plt.figure()
# fig.add_subplot(1, 2, 1)

# plt.imshow(grayorigin, cmap='gray', interpolation='none')
# fig.add_subplot(1, 4, 2)
# plt.imshow(LR, cmap='gray', interpolation='none')

# plt.imshow(HR[100:200, 100:200, 130], cmap='gray', interpolation='none')
# fig.add_subplot(1, 2, 2)
# plt.imshow(result[100:200, 100:200, 130], cmap='gray', interpolation='none')
#
# plt.show()










#mat_file2 = np.array(nib.load(fileLRList[0]).dataobj)


#LR = mat_file2 / np.max(mat_file2)
# LR = mat_file2
# LR = LR / scale_max
# [Lgx, Lgy, Lgz] = np.gradient(LR)
#
# [x_use, y_use, z_use] = cropBlack(LR)
# print("x: ", x_use, "y: ", y_use, "z: ", z_use)

xRange = range(60,200)
yRange = range(85,225)
zRange = range(60,200)

#LR = cp.array(LR)

check1 = 0
check2 = 0
check3 = 0
check4 = 0
check5 = 0
check6 = 0

@cuda.jit
def ata_add2(A, B):
    i, j = cuda.grid(2)
    if i < A.shape[1] and j < A.shape[1]:
        B[i, j] += A[0, i] * A[0, j]


#@jit(nopython=True, parallel=True)
#@cuda.jit
def addNumba(A, B):
    A += B

# tT = cp.zeros((Q_total))
#
# Iterate over each pixel
for i in range(RANGES):
    #print(np.max(LR[xP]))
    
    xP = np.random.choice(xRange, 1)[0]
    yP = np.random.choice(yRange, 1)[0]
    zP = np.random.choice(zRange, 1)[0]
    # xP = i // 50 + 120
    # yP = i % 50 + 140
    # zP = i % 17 + 120

    start = time.time()
    print(i, "/", RANGES, end='\r', flush=True)

    # Take patch

    patch = LR[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
            zP - filter_half : zP + (filter_half + 1)]
    gx = Lgx[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
            zP - filter_half : zP + (filter_half + 1)]
    gy = Lgy[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
            zP - filter_half : zP + (filter_half + 1)]
    gz = Lgz[xP - filter_half : xP + (filter_half + 1), yP - filter_half : yP + (filter_half + 1),
        zP - filter_half : zP + (filter_half + 1)]

    check1 += time.time() - start
    

    if not np.any(patch):
        continue

    start = time.time()

    # Hashtable
    # G = np.vstack((gx.ravel(), gy.ravel(), gz.ravel())).T
    # x0 = np.dot(G.T, weight)
    # x = np.dot(x0, G)

    

    # [eigenvalues, eigenvectors] = np.linalg.eig(x)

    # idx = eigenvalues.argsort()[::-1]
    # eigenvalues = eigenvalues[idx]
    # eigenvectors = eigenvectors[:, idx]

    # # For angle
    # angle_p = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    # angle_t = math.acos(eigenvectors[2, 0] / (np.linalg.norm(eigenvectors[:, 0]) + 0.0001))

    # if angle_p < 0:
    #     angle_p += math.pi
    #     angle_t = math.pi - angle_t

    # # For strength
    # strength = eigenvalues[0]

    # # For coherence
    # lamda1 = math.sqrt(eigenvalues[0])
    # lamda2 = math.sqrt(eigenvalues[1])
    # coherence = (lamda1 - lamda2) / (lamda1 + lamda2 + 0.0001)

    # # Quantization
    # angle_p = int(angle_p / math.pi * Qangle_p - 0.0001)
    # angle_t = int(angle_t / math.pi * Qangle_t - 0.0001)

    # if strength < 0.0001:
    #     strength = 0
    # elif strength > 0.001:
    #     strength = 2
    # else:
    #     strength = 1
    # if coherence < 0.25:
    #     coherence = 0
    # elif coherence > 0.5:
    #     coherence = 2
    # else:
    #     coherence = 1

    angle_p, angle_t, strength, coherence = hashtable(gx, gy, gz, weight)

    if i != 0:
        check2 += time.time() - start
    start = time.time()

    
    # Compressed vector space
    j = angle_p * Qangle_t * Qcoherence * Qstrength + angle_t * Qcoherence * Qstrength + strength * Qcoherence + coherence
    t = xP % 2 * 4 + yP % 2 * 2 + zP % 2



    #A = np.matrix(patch.ravel())
    #A = patch.reshape(1, -1)
    pk = patch.reshape(-1)
    hh = h[j, t].reshape(1, -1)

    x = HR[xP, yP, zP]

    check3 += time.time() - start
    start = time.time()


    jtS[i] = np.array([j, t])
    xS[i] = x

    check4 += time.time() - start
    start = time.time()


    patchS[i] = pk
    

    
    


    # Save the corresponding HashMap
    # Ac = cp.array(A)
    # ATA = cp.dot(Ac.T, Ac)
    # ATA = ATA.get()

    #Ac = cp.array(A)
    #ATA = np.dot(A.T, A)
    #ATA = dotNumba(A.T, A)
    #ATA = np.zeros((1331,1331))
    #matmul(A.T, A, ATA)
    #ATA = ATA.get()
    #matmul_add(A.T, A, Q[j, t])

    #Q[j, t] += np.dot(A.T, A)
    #ata_add(A, Q[j, t])

    #ata_add_cuda_all(A, Q[j, t])

    #addNumba(Q[j, t], ATA)
    #Q[j, t] += ATA
    #

    check5 += time.time() - start
    

    

    #V[j, t] += np.dot(A.T, x)

    

#print(tT)

start = time.time()

for i in prange(RANGES):
    j, t = jtS[i]
    patchT = patchS[i].reshape(1, -1)

    #Q[j, t] += np.dot(patchT.T, patchT)
    ata_add(patchT, Q[j, t])
    V[j, t] += np.dot(patchT.T, xS[i])
        

check6 += time.time() - start


print("check1:", check1)
print("check2:", check2)
print("check3:", check3)
print("check4:", check4)
print("check5:", check5)
print("check6:", check6)
