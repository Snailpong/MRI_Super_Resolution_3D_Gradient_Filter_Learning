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

from skimage.measure import compare_psnr, compare_ssim

C.argument_parse()


def make_image(im_LR, im_GX, im_GY, im_GZ, w, stre, cohe, h):
    H = im_LR.shape[0]
    result_image = im_LR.copy()
    # im_LR = np.array(im_LR, dtype=np.float64)

    timer = time.time()
    for i1 in range(C.PATCH_HALF, H - C.PATCH_HALF):
        print('\r{} / {}    {} s'.format(i1, H - C.PATCH_HALF, ((time.time() - timer) * 100 // 10) / 10), end='')
        timer = time.time()
        result_image = make_image_yz(i1, result_image, im_LR, im_GX, im_GY, im_GZ, w, stre, cohe, h)

    result_image = np.clip(result_image, 0, 1)

    return result_image


@njit
def make_image_yz(i1, result_image, im_LR, im_GX, im_GY, im_GZ, w, stre, cohe, h):
    H, W, D = im_LR.shape
    
    for j1 in range(C.PATCH_HALF, W - C.PATCH_HALF):
        for k1 in range(C.PATCH_HALF, D - C.PATCH_HALF):
            idxp = (slice(i1 - C.PATCH_HALF, i1 + C.PATCH_HALF + 1),
                    slice(j1 - C.PATCH_HALF, j1 + C.PATCH_HALF + 1),
                    slice(k1 - C.PATCH_HALF, k1 + C.PATCH_HALF + 1))
            patch = im_LR[idxp]

            if im_LR[i1, j1, k1] == 0:
                continue

            np.where(patch == 0, patch[C.PATCH_HALF, C.PATCH_HALF, C.PATCH_HALF], patch)

            # if np.any(patch == 0):
            #     patch[np.where(patch == 0)] = patch[PATCH_HALF, PATCH_HALF, PATCH_HALF]

            idxg = (slice(i1 - C.GRADIENT_HALF, i1 + C.GRADIENT_HALF + 1),
                    slice(j1 - C.GRADIENT_HALF, j1 + C.GRADIENT_HALF + 1),
                    slice(k1 - C.GRADIENT_HALF, k1 + C.GRADIENT_HALF + 1))

            patchX = im_GX[idxg]
            patchY = im_GY[idxg]
            patchZ = im_GZ[idxg]

            angle_p, angle_t, lamda, u = get_hash(patchX, patchY, patchZ, w, stre, cohe)
            j = int(angle_p * C.Q_STRENGTH * C.Q_COHERENCE * C.Q_ANGLE_T + angle_t * C.Q_STRENGTH * C.Q_COHERENCE + lamda * C.Q_COHERENCE + u)

            patch1 = patch.ravel()
            result_image[i1, j1, k1] = np.dot(patch1, h[j])

    return result_image


current_hour = time.strftime('%m%d%H', time.localtime(time.time()))
result_dir = './result/{}/'.format(current_hour)
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

file_list = make_dataset('./test')

# Preprocessing normalized Gaussian matrix W for hashkey calculation
G_WEIGHT = get_normalized_gaussian()

h = np.load('./arrays/h_{}.npy'.format(C.R))
h = np.array(h, dtype=np.float32)

with open("./arrays/Qfactor_str" + str(C.R), "rb") as sp:
    stre = pickle.load(sp)

with open("./arrays/Qfactor_coh" + str(C.R), "rb") as cp:
    cohe = pickle.load(cp)

print(stre, cohe)
filestart = time.time()

for file_idx, file in enumerate(file_list):
    file_name = file.split('\\')[-1].split('.')[0]
    print('\r', end='')
    print('' * 60, end='')
    print('\rProcessing ' + str(file_idx + 1) + '/' + str(len(file_list)) + ' image (' + file_name + ')' + str(time.time() - filestart))
    filestart = time.time()

    raw_image = nib.load(file).get_fdata()
    crop_image = mod_crop(raw_image, C.R)
    clipped_image = clip_image(crop_image)
    slice_area = crop_slice(clipped_image, C.PATCH_SIZE // 2, C.R)

    im_blank_LR = get_lr(clipped_image) / clipped_image.max()  # Prepare the cheap-upscaling images
    im_LR = im_blank_LR[slice_area]
    im_GX, im_GY, im_GZ = np.gradient(im_LR)  # Calculate the gradient images

    im_GX[np.where(im_LR == 0)] = 0
    im_GY[np.where(im_LR == 0)] = 0
    im_GZ[np.where(im_LR == 0)] = 0

    im_HR = clipped_image[slice_area] / clipped_image.max()

    im_result = make_image(im_LR, im_GX, im_GY, im_GZ, G_WEIGHT, stre, cohe, h)
    # im_blending = Blending2(im_LR, im_result)
    # im_blending = blend_image(im_LR, im_result)
    # im_blending = blend_image3(im_LR, im_result, 3.5)
    # im_blending = Backprojection(imL, im_blending, 50) #Optional: Backprojection, which can slightly improve PSNR, especilly for large upscaling factor.
    # im_blending = np.clip(im_blending, 0, 1)

    output_img = np.zeros(raw_image.shape)
    output_img[slice_area] = im_result
    output_img = output_img * clipped_image.max()
    ni_img = nib.Nifti1Image(output_img, np.eye(4))
    nib.save(ni_img, '{}/{}_result.nii.gz'.format(result_dir, file_name))

    # output_img2 = np.zeros(raw_image.shape)
    # output_img2[slice_area] = im_blending
    # output_img2 = output_img2 * clipped_image.max()
    # ni_img2 = nib.Nifti1Image(output_img2, np.eye(4))
    # nib.save(ni_img2, '{}/{}_result_blend.nii.gz'.format(result_dir, file_name))

    print()
    print(compare_psnr(im_HR, im_LR), compare_psnr(im_HR, im_result))
    print(compare_ssim(im_HR, im_LR), compare_ssim(im_HR, im_result))
    # area = np.nonzero(im_HR)
    # print(compare_psnr(im_HR, im_LR), compare_psnr(im_HR, im_blending))

    # downscaled_lr = zoom(clipped_image, 1.0 / C.R, order=2, prefilter=False)
    # lr2 = np.clip(zoom(downscaled_lr, C.R, order=1), 0, clipped_image.max()) / clipped_image.max()
    # lr2[np.where(clipped_image == 0)] = 0

    # lr3 = np.clip(zoom(downscaled_lr, C.R, order=2), 0, clipped_image.max()) / clipped_image.max()
    # lr3[np.where(clipped_image == 0)] = 0

    # print(compare_psnr(im_HR, lr2[slice_area]), compare_psnr(im_HR, lr3[slice_area]))
    # print()

    # downscaled_lr = zoom(clipped_image, 1.0 / C.R, order=2)
    # lr2 = np.clip(zoom(downscaled_lr, C.R, order=1), 0, clipped_image.max()) / clipped_image.max()
    # lr2[np.where(clipped_image == 0)] = 0

    # lr3 = np.clip(zoom(downscaled_lr, C.R, order=2), 0, clipped_image.max()) / clipped_image.max()
    # lr3[np.where(clipped_image == 0)] = 0

    # print(compare_psnr(im_HR, lr2[slice_area]), compare_psnr(im_HR, lr3[slice_area]))
    # print()

    #if file_idx == 0:
        #break



print("Test is off")