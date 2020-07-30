import numpy as np
import random

from filter_constant import *

def get_sampled_point_list(array):
    [x_range, y_range, z_range] = get_range(array)

    xyz_range = [(x,y,z) for x in x_range for y in y_range for z in z_range]
    sample_range = random.sample(xyz_range, len(xyz_range) // TRAIN_DIV)
    split_range = list(chunks(sample_range, len(sample_range) // TRAIN_STP - 1))

    return split_range


def get_range(array):
    [x_use, y_use, z_use] = crop_black(array)

    x_range = range(max(FILTER_HALF, x_use[0] - FILTER_HALF), min(LR.shape[0] - FILTER_HALF, x_use[1] + FILTER_HALF))
    y_range = range(max(FILTER_HALF, y_use[0] - FILTER_HALF), min(LR.shape[1] - FILTER_HALF, y_use[1] + FILTER_HALF))
    z_range = range(max(FILTER_HALF, z_use[0] - FILTER_HALF), min(LR.shape[2] - FILTER_HALF, z_use[1] + FILTER_HALF))

    return x_range, y_range, z_range

def crop_black(array):
    array = array.copy()

    if str(type(array)) == '<class \'cupy.core.core.ndarray\'>':
        array = array.get()

    #print ('zero ratio:', np.count_nonzero(array==0)*100 / ((array.shape[0])*(array.shape[1])*(array.shape[2])))

    x_use=[]
    y_use=[]
    z_use=[]

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