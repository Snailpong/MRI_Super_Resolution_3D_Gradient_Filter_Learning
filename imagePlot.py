import os
import glob

import numpy as np
import cupy as cp
from numba import jit

import scipy.sparse as sparse
from scipy.sparse.linalg import cg
from scipy.ndimage import zoom

import nibabel as nib
import matplotlib.pyplot as plt
import cv2
#from scipy import sparse

from hashTable import hashtable, hashtable_cupy
from getMask import crop_black
from util import *
from filterVariable import *


file = "test/T1w_acpc_dc_restore_brain_101410.nii.gz"
file2 = "result/071907_0outputt3_gg.nii.gz"


# Load NIfTI Image
HR = nib.load(file).dataobj[:, :-1, :]
LR = get_lr_kspace(HR)
#LR = dog_sharpener(HR)
Result = nib.load(file2).dataobj[:, :-1, :]

HR = np.flip(HR.T, 0)
LR = np.flip(LR.T, 0)
Result = np.flip(Result.T, 0)

fig = plt.figure()
fig.add_subplot(3, 3, 1)
plt.imshow(HR[130, :, :], cmap='gray')
fig.add_subplot(3, 3, 2)
plt.imshow(LR[130, :, :], cmap='gray')
fig.add_subplot(3, 3, 3)
plt.imshow(Result[130, :, :], cmap='gray')

fig.add_subplot(3, 3, 4)
plt.imshow(HR[:, 155, :], cmap='gray')
fig.add_subplot(3, 3, 5)
plt.imshow(LR[:, 155, :], cmap='gray')
fig.add_subplot(3, 3, 6)
plt.imshow(Result[:, 155, :], cmap='gray')

fig.add_subplot(3, 3, 7)
plt.imshow(HR[:, :, 130], cmap='gray')
fig.add_subplot(3, 3, 8)
plt.imshow(LR[:, :, 130], cmap='gray')
fig.add_subplot(3, 3, 9)
plt.imshow(Result[:, :, 130], cmap='gray')

plt.show()
