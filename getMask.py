import numpy as np
import os
import pickle
import nibabel as nib

def getMask(fileName, HR, filter_half):
    fileName = fileName.split('\\')[-1].split('.')[0]

    if os.path.isfile('./mask_array/' + fileName + '.lst'):
        with open('./mask_array/' + fileName + '.lst', 'rb') as f:
            mask = pickle.load(f)
        return mask

    print('Making mask array...')

    mask = []

    [x_use, y_use, z_use] = cropBlack(HR)
    print("x: ", HR.shape[0], '->', x_use, "y: ", HR.shape[1], '->', y_use, "z: ", HR.shape[2], '->', z_use)

    for xP in range(max(x_use[0], filter_half), min(x_use[1], HR.shape[0] - filter_half)):
        print(xP - x_use[0] + 1, '/', x_use[1] - x_use[0], end='\r')
        for yP in range(max(y_use[0], filter_half), min(y_use[1], HR.shape[1] - filter_half)):
            for zP in range(max(z_use[0], filter_half), min(z_use[1], HR.shape[2] - filter_half)):
                if HR[xP][yP][zP] != 0:
                    mask.append([xP, yP, zP])

    with open('./mask_array/' + fileName + '.lst', 'wb') as f:
        pickle.dump(mask, f)


    return mask

def getMask_test(fileName, LR, filter_half):
    fileName = fileName.split('\\')[-1].split('.')[0]
    if fileName.startswith('LR_'):
        fileName = fileName[3:]
    if os.path.isfile('./mask_array/' + fileName + '.lst'):
        with open('./mask_array/' + fileName + '.lst', 'rb') as f:
            mask = pickle.load(f)
        return mask

    print('Making mask array...')

    mask = []

    filePath = './test/' + fileName + '.nii.gz'
    HR = nib.load(filePath).get_fdata()

    [x_use, y_use, z_use] = cropBlack(HR)
    print("x: ", HR.shape[0], '->', x_use, "y: ", HR.shape[1], '->', y_use, "z: ", HR.shape[2], '->', z_use)

    for xP in range(max(x_use[0], filter_half), min(x_use[1], HR.shape[0] - filter_half)):
        print(xP - x_use[0] + 1, '/', x_use[1] - x_use[0], end='\r')
        for yP in range(max(y_use[0], filter_half), min(y_use[1], HR.shape[1] - filter_half)):
            for zP in range(max(z_use[0], filter_half), min(z_use[1], HR.shape[2] - filter_half)):
                if HR[xP][yP][zP] != 0:
                    mask.append([xP, yP, zP])

    with open('./mask_array/' + fileName + '.lst', 'wb') as f:
        pickle.dump(mask, f)


    return mask

def crop_black(array):
    #print('original Data shape is ' + str(array.shape) + ' .')
    array = array.copy()

    if str(type(array)) == '<class \'cupy.core.core.ndarray\'>':
        array = array.get()

    array = array.round(out=array).astype(np.uint8)

    idx = []
    x_use = [0, array.shape[0]]
    for i in range(array.shape[0]):
        if np.max(array[i, :, :]) == 0:
            idx.append(i)
    for i in range(len(idx) - 1):
        if (idx[i + 1] - idx[i]) != 1:
            x_use[0] = idx[i] + 1
            x_use[1] = idx[i + 1]

    idx2 = []
    y_use = [0, array.shape[1]]
    for i in range(array.shape[1]):
        if np.max(array[:, i, :]) == 0:
            idx2.append(i)
    for i in range(len(idx2) - 1):
        if (idx2[i + 1] - idx2[i]) != 1:
            y_use[0] = idx2[i] + 1
            y_use[1] = idx2[i + 1]

    idx3 = []
    z_use = [0, array.shape[2]]
    for i in range(array.shape[2]):
        if np.max(array[:, :, i]) == 0:
            idx3.append(i)
    for i in range(len(idx3) - 1):
        if (idx3[i + 1] - idx3[i]) != 1:
            z_use[0] = idx3[i] + 1
            z_use[1] = idx3[i + 1]

    return x_use, y_use, z_use
