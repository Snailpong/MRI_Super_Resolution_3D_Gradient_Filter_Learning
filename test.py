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

C.argument_parse()

current_hour = time.strftime('%m%d%H', time.localtime(time.time()))
# current_hour = '080413'

fileList = [file for file in glob.glob(C.TEST_GLOB)]

# Preprocessing normalized Gaussian matrix W for hashkey calculation
G_WEIGHT = get_normalized_gaussian()

h = np.load('{}.npy'.format(C.H_FILE))

for idx, file in enumerate(fileList):
    filestart = time.time()

    fileName = file.split('/')[-1].split('\\')[-1]
    fileNumber = fileName.split('_')[-1].split('.')[0]
    print('\r{} / {}\t{}'.format(idx+1, len(fileList), fileName))

    # Load NIfTI Image
    mat_file = nib.load(file)
    HR = np.array(mat_file.dataobj)[:, :-1, :]
    HR = clipped_hr(HR)
    HR_max = HR.max()

    # Make LR
    print('Making LR...', end='', flush=True)
    LR = get_lr(HR)
    LRDirect = np.zeros((LR.shape[0], LR.shape[1], LR.shape[2]))

    [Lgx, Lgy, Lgz] = np.gradient(LR)
    xRange, yRange, zRange = get_range(HR)

    # ni_img = nib.Nifti1Image(LR, np.eye(4))
    # nib.save(ni_img, str(idx) + 'LR.nii.gz')

    start = time.time()

    for xP in xRange:
        print('\r{} / {}, last {} s '.format(xP - xRange[0], xRange[-1] - xRange[0], '%.1f' % (time.time() - start)), end='', flush=True)
        start = time.time()

        for yP in yRange:
            for zP in zRange:
                patch = get_patch(LR, xP, yP, zP)

                if not np.any(patch):
                    continue

                gx, gy, gz = get_gxyz(Lgx, Lgy, Lgz, xP, yP, zP)

                [angle_p, angle_t, strength, coherence] = hashtable(gx, gy, gz, G_WEIGHT)
                # [angle_p, angle_t, strength, coherence] = get_features2(gx, gy, gz, G_WEIGHT)

                j = angle_p * Q_ANGLE_T * Q_COHERENCE * Q_STRENGTH + angle_t * Q_COHERENCE * Q_STRENGTH + strength * Q_COHERENCE + coherence
                t = xP % FACTOR * (FACTOR ** 2) + yP % FACTOR * FACTOR + zP % FACTOR

                AT = patch.reshape((1, -1))
                hh = h[j][t].reshape((-1, 1))
                LRDirect[xP][yP][zP] = np.dot(AT, hh)

    LRDirect = np.clip(LRDirect, 0, HR_max)
    HR_Blend = blend_image(LR, LRDirect, C.BLEND_THRESHOLD)
    # ni_img = nib.Nifti1Image(LRDirect, np.eye(4))
    # nib.save(ni_img, str(idx) + 'outputt2.nii.gz')
    
    ni_img = nib.Nifti1Image(HR_Blend, np.eye(4))
    nib.save(ni_img, '{}{}_{}_outputt.nii.gz'.format(C.RESULT_DIR, current_hour, fileNumber))
    print(' ' * 30, 'last', '%.1f' % ((time.time() - filestart) / 60), 'min', end='', flush=True)

print("Test is off")