import os
import numpy as np
#import cv2
#import scipy as scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import cg
import nibabel as nib
import matplotlib.pyplot as plt
#from scipy import sparse

from hashTable import hashTable
from cropBlack import cropBlack
from filterVariable import *

crop_pass = 40

Q_total = Qangle_t * Qangle_p * Qstrength * Qcoherence
filter_volume = filter_length ** 3

def process_file():
    HR = mat

    mat_max = np.max(mat)
    mat = mat / mat_max

    # Fuzzy LR
    # Upscaling
    # cv2.GaussianBlur()
    # LR = cv2.GaussianBlur(LR,(0,0),2);

    fig = plt.figure()

    img = np.transpose(mat, (1, 0, 2))

    fig.add_subplot(2, 3, 1)
    plt.imshow(img[:, :, 130])

    img = np.fft.fftshift(img)

    imgfft = np.fft.fftn(img)
    imgfft = np.fft.fftshift(imgfft)

    fig.add_subplot(2, 3, 2)
    plt.imshow(20 * np.log(abs(imgfft[:, :, 0])))  # magnitude

    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2

    imgfft_cut = np.array(imgfft)
    imgfft_cut[x_center - crop_pass: x_center + crop_pass,
    y_center - crop_pass: y_center + crop_pass, z_center - crop_pass: z_center + crop_pass] = 0
    imgfft = imgfft - imgfft_cut

    fig.add_subplot(2, 3, 3)  # substracted magnitude
    plt.imshow(20 * np.log(abs(imgfft[:, :, 130])))
    imgifft = np.fft.ifftn(imgfft)

    fig.add_subplot(2, 3, 4)  # ifft
    plt.imshow(abs(imgifft[:, :, 0]))

    fig.add_subplot(2, 3, 5)  # 원래와 비교한 것
    plt.imshow(img[:, :, 0] - np.abs(imgifft[:, :, 0]))

    fig.add_subplot(2, 3, 6)
    imgifft = np.concatenate((imgifft[x_center:, :, :], imgifft[:x_center, :, :]), axis=0)
    imgifft = np.concatenate((imgifft[:, y_center:, :], imgifft[:, :y_center, :]), axis=1)
    imgifft = np.concatenate((imgifft[:, :, z_center:], imgifft[:, :, :z_center]), axis=2)

    LR = np.abs(imgifft)
    LR = LR / np.max(LR) * mat_max

    plt.imshow(LR[:, :, 130])

    LR = np.transpose(LR, (1, 0, 2))
    LR = LR.astype('float32')

    print(HR[x_center - crop_pass: x_center + crop_pass, y_center - crop_pass: y_center + crop_pass, 130])
    print(LR[x_center - crop_pass: x_center + crop_pass, y_center - crop_pass: y_center + crop_pass, 130])

    print(mat_max)

    print(np.min(LR))
    print(np.max(LR))

    plt.show()

    ni_img = nib.Nifti1Image(LR, np.eye(4))
    nib.save(ni_img, 'output.nii.gz')


# Construct an empty matrix Q, V uses the corresponding LR and HR, h is the filter, three hashmaps are Angle, Strength, Coherence, t
Q = np.zeros((Q_total, 8, filter_volume, filter_volume))
V = np.zeros((Q_total, 8, filter_volume, 1))
h = np.zeros((Q_total, 8, filter_volume))

dataDir="./train_3d"
dataLRDir="./train_3d_low"

fileName = 'T1w_acpc_dc_restore_brain_100307.nii.gz'

fileList = [dataDir + '/' + fileName]
fileLRList = [dataLRDir + '/' + fileName]

"""
fileList = []
fileLRList = []
for parent,dirnames,filenames in os.walk(dataDir):
    for filename in filenames:
        fileList.append(os.path.join(parent, filename))

for parent,dirnames,filenames in os.walk(dataLRDir):
    for filename in filenames:
        fileLRList.append(os.path.join(parent, filename))
"""


ct = 0

for file in fileList:
    print("HashMap of %s"%file)
    # mat = cv2.imread(file)
    mat_file = nib.load(file)
    mat = np.array(mat_file.dataobj)

    

    # Scale to 0-1
    HR = mat / np.max(mat)

    mat_file2 = np.array(nib.load(fileLRList[ct]).dataobj)
    LR = mat_file2 / np.max(mat)
    [Lgx, Lgy, Lgz] = np.gradient(LR)

    [x_use, y_use, z_use] = cropBlack(LR)
    print("x: ", x_use, "y: ", y_use, "z: ", z_use)

    ct += 1

    # Set the train map
    # Iterate over each pixel
    for xP in range(x_use[0] + filter_half+100, x_use[1] - filter_half-100):
        for yP in range(y_use[0] + filter_half+100, y_use[1] - filter_half-100):

            print(xP - x_use[0] - filter_half, "/", (x_use[1] - 2 * filter_half - x_use[0]), '\t',
                yP - y_use[0] - filter_half, "/", (y_use[1] - 2 * filter_half - y_use[0]), end='\r', flush=True)

            for zP in range(z_use[0] + filter_half+100, z_use[1] - filter_half-100):
                # Take patch
                patch = LR[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]
                gx = Lgx[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]
                gy = Lgy[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                        zP - filter_half: zP + (filter_half + 1)]
                gz = Lgz[xP - filter_half: xP + (filter_half + 1), yP - filter_half: yP + (filter_half + 1),
                    zP - filter_half: zP + (filter_half + 1)]

                # Computational characteristics
                [angle_t, angle_p, strength, coherence] = hashTable([gx, gy, gz], Qangle_t, Qangle_p, Qstrength, Qcoherence)

                # Compressed vector space
                j = angle_t * Qangle_p * Qcoherence * Qstrength + angle_p * Qcoherence * Qstrength + strength * Qcoherence + coherence
                A = patch.reshape(1, -1)
                x = HR[xP][yP][zP]

                # Calculate pixel type
                t = (xP % 2) * 4 + (yP % 2) * 2 + zP % 2

                # Save the corresponding HashMap
                Q[j, t] += A * A.T
                V[j, t] += A.T * x

                #print(V[j, t].T)


# Set the train step
for t in range(8):
    for j in range(Qangle_t * Qangle_p * Qstrength * Qcoherence):
        # Train 8 * 24 * 3 * 3 filters for each pixel type and image feature
        h[j, t] = cg(Q[j, t], V[j, t])[0]
        #h[j,t] = sparse.linalg.cg(Q[j,t],V[j,t])[0]

print("Train is off")
np.save("./lowR4", h)
