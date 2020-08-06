import numpy as np
from numba import jit, prange
import random

import filter_constant as C
from util import *

def get_point_list_pixel_type(array):
    sampled_list = [[] for j in range(C.PIXEL_TYPE)]
    for xP, yP, zP in array:
        t = xP % C.FACTOR * (C.FACTOR ** 2) + yP % C.FACTOR * C.FACTOR + zP % C.FACTOR
        sampled_list[t].append([xP, yP, zP])
    return sampled_list

def get_sampled_point_list(array):
    [x_range, y_range, z_range] = get_range(array)

    xyz_range = [[x,y,z] for x in x_range for y in y_range for z in z_range]
    sample_range = random.sample(xyz_range, len(xyz_range) // C.TRAIN_DIV)
    sampled_list = get_point_list_pixel_type(sample_range)
    #split_range = list(chunks(sample_range, len(sample_range) // TRAIN_STP - 1))

    return sampled_list


def get_range(array):
    [x_use, y_use, z_use] = crop_black(array)

    x_range = range(max(C.FILTER_HALF, x_use[0] - C.FILTER_HALF), min(array.shape[0] - C.FILTER_HALF, x_use[1] + C.FILTER_HALF))
    y_range = range(max(C.FILTER_HALF, y_use[0] - C.FILTER_HALF), min(array.shape[1] - C.FILTER_HALF, y_use[1] + C.FILTER_HALF))
    z_range = range(max(C.FILTER_HALF, z_use[0] - C.FILTER_HALF), min(array.shape[2] - C.FILTER_HALF, z_use[1] + C.FILTER_HALF))

    return x_range, y_range, z_range

def crop_black(array):
    array = array.copy()

    if str(type(array)) == '<class \'cupy.core.core.ndarray\'>':
        array = array.get()

    x_use, y_use, z_use = [], [], []

    for i in range (array.shape[0]):
        if np.all(array[i,:,:] == 0) == False:
            x_use.append(i)
            break
    for i in reversed(range(array.shape[0])):
        if np.all(array[i,:,:] == 0) == False:
            x_use.append(i + 1)
            break

    for i in range (array.shape[1]):
        if np.all(array[:,i,:] == 0) == False:
            y_use.append(i)
            break
    for i in reversed(range(array.shape[1])):
        if np.all(array[:,i,:] == 0) == False:
            y_use.append(i + 1)
            break
    
    for i in range (array.shape[2]):
        if np.all(array[:,:,i] == 0) == False:
            z_use.append(i)
            break
    for i in reversed(range(array.shape[2])):
        if np.all(array[:,:,i] == 0) == False:
            z_use.append(i + 1)
            break

    return x_use, y_use, z_use

def cropping(array, x_range, y_range, z_range):
    return array[x_range[0]:x_range[-1], y_range[0]:y_range[-1], z_range[0]:z_range[-1]]