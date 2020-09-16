import os
import glob
import time

import numpy as np
import cupy as cp
from numba import jit, njit, prange

import nibabel as nib

import filter_constant as C

from crop_black import *
from filter_constant import *
from filter_func import *
from get_lr import *
from hashtable import *
from matrix_compute import *

from skimage.measure import compare_psnr

C.argument_parse()
determine_geometric_func()

C.R = 4
C.USE_PIXEL_TYPE = False

def make_image(im_LR, im_GX, im_GY, im_GZ, w, stre, cohe, h):
    H = im_LR.shape[0]
    result_image = im_LR.copy()

    timer = time.time()
    for i1 in range(H - 2 * PATCH_HALF):
        print('\r{} / {}    {} s'.format(i1, H - 2 * PATCH_HALF, ((time.time() - timer) * 100 // 10) / 10), end='')
        timer = time.time()
        result_image = make_image_yz(i1, result_image, im_LR, im_GX, im_GY, im_GZ, w, stre, cohe, h)

    result_image = np.clip(result_image, 0, 1)

    return result_image

# @njit(parallel=True)
def make_image_yz(i1, result_image, im_LR, im_GX, im_GY, im_GZ, w, stre, cohe, h):
    H, W, D = im_LR.shape
    for j1 in prange(W - 2 * PATCH_HALF):
        for k1 in prange(D - 2 * PATCH_HALF):
            idx1 = (slice(i1, (i1 + 2 * PATCH_HALF + 1)), slice(j1, (j1 + 2 * PATCH_HALF + 1)),
                    slice(k1, (k1 + 2 * PATCH_HALF + 1)))
            patch = im_LR[idx1]

            if im_LR[i1 + C.GRADIENT_HALF, j1 + C.GRADIENT_HALF, k1 + C.GRADIENT_HALF] == 0:
                    continue

            idx2 = (slice(i1+1, (i1 + 2 * GRADIENT_HALF + 2)), slice(j1+1, (j1 + 2 * GRADIENT_HALF + 2)),
                    slice(k1+1, (k1 + 2 * GRADIENT_HALF + 2)))

            patchX = im_GX[idx2]
            patchY = im_GY[idx2]
            patchZ = im_GZ[idx2]

            angle_p, angle_t, lamda, u = get_hash(patchX, patchY, patchZ, w, stre, cohe)
            j = int(angle_p * Q_STRENGTH * Q_COHERENCE * Q_ANGLE_T + angle_t * Q_STRENGTH * Q_COHERENCE + lamda * Q_COHERENCE + u)
            t = (i1 % R) * R * R + (j1 % R) * R + k1 % R

            patch1 = patch.ravel()
            result_image[i1 + PATCH_HALF, j1 + PATCH_HALF, k1 + PATCH_HALF] = np.dot(patch1, h[j])

    return result_image

current_hour = time.strftime('%m%d%H', time.localtime(time.time()))
# current_hour = '080413'

testPath = './test'
file_list = make_dataset(testPath)

# Preprocessing normalized Gaussian matrix W for hashkey calculation
G_WEIGHT = get_normalized_gaussian()

h = np.load('{}.npy'.format(C.H_FILE))
h = np.array(h, dtype=np.float32)

with open("./arrays/Qfactor_str" + str(R), "rb") as sp:
    stre = pickle.load(sp)

with open("./arrays/Qfactor_coh" + str(R), "rb") as cp:
    cohe = pickle.load(cp)

for file_idx, file in enumerate(file_list):
    filestart = time.time()

    file_name = file.split('\\')[-1].split('.')[0]
    print('\r', end='')
    print('' * 60, end='')
    print('\rProcessing ' + str(file_idx + 1) + '/' + str(len(file_list)) + ' image (' + file_name + ')')

    raw_image = nib.load(file).dataobj
    crop_image = mod_crop(raw_image, C.R)
    clipped_image = clip_image(crop_image)
    slice_area = crop_slice(clipped_image, PATCH_SIZE // 2, C.R)

    im_blank_LR = get_lr(clipped_image) / clipped_image.max()  # Prepare the cheap-upscaling images
    im_LR = im_blank_LR[slice_area]
    im_GX, im_GY, im_GZ = np.gradient(im_LR)  # Calculate the gradient images
    im_HR = clipped_image[slice_area] / clipped_image.max()

    im_result = make_image(im_LR, im_GX, im_GY, im_GZ, G_WEIGHT, stre, cohe, h)
    # im_blending = Blending2(im_LR, im_result)
    im_blending = blend_image3(im_LR, im_result, 5)
    # im_blending = Backprojection(imL, im_blending, 50) #Optional: Backprojection, which can slightly improve PSNR, especilly for large upscaling factor.
    im_blending = np.clip(im_blending, 0, 1)

    output_img = np.zeros(raw_image.shape)
    output_img[slice_area] = im_result
    output_img = output_img * clipped_image.max()
    ni_img = nib.Nifti1Image(output_img, np.eye(4))
    nib.save(ni_img, './result/{}_{}_result.nii.gz'.format(file_name, current_hour))

    output_img2 = np.zeros(raw_image.shape)
    output_img2[slice_area] = im_blending
    output_img2 = output_img2 * clipped_image.max()
    ni_img2 = nib.Nifti1Image(output_img2, np.eye(4))
    nib.save(ni_img2, './result/{}_{}_result_blend.nii.gz'.format(file_name, current_hour))

    print(compare_psnr(im_HR, im_LR))
    print(compare_psnr(im_HR, im_result))
    print(compare_psnr(im_HR, im_blending))

    area = np.nonzero(im_HR)

    print(compare_psnr(im_HR[area], im_LR[area]))
    print(compare_psnr(im_HR[area], im_result[area]))
    print(compare_psnr(im_HR[area], im_blending[area]))



    if file_idx == 0:
        break



print("Test is off")