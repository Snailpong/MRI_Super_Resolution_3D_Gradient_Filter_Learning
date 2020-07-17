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


file = "0LR.nii.gz"
file2 = "0outputt2_gg.nii.gz"


# Load NIfTI Image
LR = nib.load(file).dataobj[:, :-1, :]
LR_empty = LR.copy()
LR_empty[60:200, 85:225, 60:200] = 0
LR = LR - LR_empty
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

HR_Blend = blend_image(LR, LRDirect)
ni_img = nib.Nifti1Image(HR_Blend, np.eye(4))
nib.save(ni_img, '0outputt3_gg.nii.gz')

