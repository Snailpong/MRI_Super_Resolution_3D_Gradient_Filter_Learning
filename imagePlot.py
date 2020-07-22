import os
import glob
import math

import numpy as np
from numba import jit

import scipy.sparse as sparse
from scipy.sparse.linalg import cg
from scipy.ndimage import zoom

import nibabel as nib
import matplotlib.pyplot as plt
import cv2
#from scipy import sparse

from hashTable import hashtable
from getMask import crop_black
from util import *
from filterVariable import *


file = "test/T1w_acpc_dc_restore_brain_101410.nii.gz"
file2 = "result/071907_0outputt3_gg.nii.gz"
#file2 = "0outputt3_gg.nii.gz"


# Load NIfTI Image
HR = nib.load(file).dataobj[:, :-1, :]
LR = get_lr_kspace(HR)
#LR = get_lr_interpolation(HR)
#LR = dog_sharpener(HR)
Result = nib.load(file2).dataobj[:, :, :]
LR_max = np.max(LR)

print(np.max(HR))
print(np.max(LR))
print(np.max(Result))

print(np.min(HR))
print(np.min(LR))
print(np.min(Result))

clip(Result, 0, 1200)
LR = LR * (np.max(HR) / np.max(LR))
Result = Result * (np.max(HR) / np.max(Result))

psmel = 10 * math.log10((np.max(HR) ** 2 / np.mean(np.square(np.subtract(HR, LR)))))
psmer = 10 * math.log10(np.max(HR) ** 2 / np.mean(np.square(np.subtract(HR, Result))))

print(psmel, psmer)


HR = np.flip(HR.T, 0)
LR = np.flip(LR.T, 0)
Result = np.flip(Result.T, 0)

fig, ax = plt.subplots(3, 3)
[axi.set_axis_off() for axi in ax.ravel()]

ax[0, 0].imshow(HR[130, :, :], cmap='gray')
ax[0, 1].imshow(LR[130, :, :], cmap='gray')
ax[0, 2].imshow(Result[130, :, :], cmap='gray')

ax[1, 0].imshow(HR[:, 155, :], cmap='gray')
ax[1, 1].imshow(LR[:, 155, :], cmap='gray')
ax[1, 2].imshow(Result[:, 155, :], cmap='gray')


ax[2, 0].imshow(HR[:, :, 130], cmap='gray')
ax[2, 1].imshow(LR[:, :, 130], cmap='gray')
ax[2, 2].imshow(Result[:, :, 130], cmap='gray')


fig.tight_layout()
plt.show()
