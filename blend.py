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

from hashTable import hashtable
from filterVariable import *
from getMask import getMask_test, crop_black
from util import *


file = "test/T1w_acpc_dc_restore_brain_101410.nii.gz"
file2 = "0outputt2_gg.nii.gz"


# Load NIfTI Image
HR = nib.load(file).dataobj[:, :-1, :]
LR = get_lr_kspace(HR)

LRDirect = nib.load(file2).dataobj[:, :-1, :]

# Dog-Sharpening
#print("Sharpening...", end='', flush=True)
#HR = dog_sharpener(HR)

# Using k-space domain
# mat_file2 = np.array(nib.load(fileLRList[idx]).dataobj)
# LR = mat_file2 / np.max(mat)

# Downscale (bicububic interpolation)
# print("\rMaking LR...", end='', flush=True)
# downscaled_LR = zoom(HR, 0.5, order=2)

# Upscale (bilinear interpolation)
# LR = zoom(downscaled_LR, 2, order=1)

HR_Blend = blend_image(LR, LRDirect, 5)
ni_img = nib.Nifti1Image(HR_Blend, np.eye(4))
nib.save(ni_img, '0outputt3_gg.nii.gz')

