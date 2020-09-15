import numpy as np
import random

import filter_constant as C
from util import *


def mod_crop(im):
    H, W, D = im.shape
    size0 = H - H % C.R
    size1 = W - W % C.R
    size2 = D - D % C.R

    out = im[0:size0, 0:size1, 0:size2]

    return out


def get_point_list_pixel_type(array):
    sampled_list = [[] for j in range(C.PIXEL_TYPE)]
    for xP, yP, zP in array:
        t = xP % C.R * (C.R ** 2) + yP % C.R * C.R + zP % C.R
        sampled_list[t].append([xP, yP, zP])
    return sampled_list


def get_sampled_point_list(array):
    [x_range, y_range, z_range] = crop_slice(array)

    xyz_range = [[x, y, z] for x in x_range for y in y_range for z in z_range]
    sample_range = random.sample(xyz_range, len(xyz_range) // C.TRAIN_DIV)
    sampled_list = get_point_list_pixel_type(sample_range)
    #split_range = list(chunks(sample_range, len(sample_range) // TRAIN_STP - 1))

    return sampled_list


def crop_slice(array):
    for i in range(C.PATCH_HALF, array.shape[0] - C.PATCH_HALF):
        if not np.all(array[i, :, :] == 0):
            x_use1 = i - C.PATCH_HALF
            x_use1 = x_use1 - (x_use1 % C.R)
            break
    for i in reversed(range(C.PATCH_HALF, array.shape[0] - C.PATCH_HALF)):
        if not np.all(array[i, :, :] == 0):
            x_use2 = i + C.PATCH_HALF
            break
    for i in range(C.PATCH_HALF, array.shape[1] - C.PATCH_HALF):
        if not np.all(array[:, i, :] == 0):
            y_use1 = i - C.PATCH_HALF
            y_use1 = y_use1 - (y_use1 % C.R)
            break
    for i in reversed(range(C.PATCH_HALF, array.shape[1] - C.PATCH_HALF)):
        if not np.all(array[:, i, :] == 0):
            y_use2 = i + C.PATCH_HALF
            break
    for i in range(C.PATCH_HALF, array.shape[2] - C.PATCH_HALF):
        if not np.all(array[:, :, i] == 0):
            z_use1 = i - C.PATCH_HALF
            z_use1 = z_use1 - (z_use1 % C.R)
            break
    for i in reversed(range(C.PATCH_HALF, array.shape[2] - C.PATCH_HALF)):
        if not np.all(array[:, :, i] == 0):
            z_use2 = i + C.PATCH_HALF
            break

    area = (slice(x_use1, x_use2), slice(y_use1, y_use2), slice(z_use1, z_use2))
    return area
