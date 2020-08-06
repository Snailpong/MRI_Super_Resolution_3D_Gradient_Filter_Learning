
import glob
import time
import random

import cupy as cp
import numpy as np
from numba import jit, prange

import nibabel as nib

import filter_constant as C

from crop_black import *
from filter_func import *
from get_lr import *
from hashtable import *
from matrix_compute import *
from util import *

C.argument_parse()

Q, V, finished_files = load_files()

fileList = [file for file in glob.glob(C.TRAIN_GLOB)]

# Preprocessing normalized Gaussian matrix W for hashkey calculation
G_WEIGHT = get_normalized_gaussian()

start = time.time()

for idx, file in enumerate(fileList):
    filestart = time.time()

    fileName = file.split('/')[-1].split('\\')[-1]
    if fileName in finished_files:
        continue

    print('\r[{} / {}]   '.format(idx+1, C.TRAIN_FILE_MAX), fileName)

    HR = nib.load(file).dataobj[:, :-1, :]  # Load NIfTI Image
    HR = normalization_hr(HR)               # Normalized to [0, 1]

    print('Making LR...', end='', flush=True)
    LR = get_lr(HR)

    print('\rSharpening...', end='', flush=True)    # Dog-Sharpening
    HR = dog_sharpener(HR)

    print('\rSampling...', end='', flush=True)
    [Lgx, Lgy, Lgz] = np.gradient(LR)
    sampled_list = get_sampled_point_list(HR)

    for t, points in enumerate(sampled_list):

        print('\r{} / {}'.format(t + 1, C.PIXEL_TYPE), end='', flush=True)
        start = time.time()
        patchS, xS = init_buckets()

        for point_idx in prange(len(points)):
            xP, yP, zP = points[point_idx]
            patch = get_patch(LR, xP, yP, zP)

            if not np.any(patch):
                continue

            gx, gy, gz = get_gxyz(Lgx, Lgy, Lgz, xP, yP, zP)

            # Computational characteristics
            angle_p, angle_t, strength, coherence = hashtable(gx, gy, gz, G_WEIGHT)
            # angle_p, angle_t, strength, coherence = get_features2(gx, gy, gz, G_WEIGHT)
            j = get_bucket(angle_p, angle_t, strength, coherence)
            
            patchS[j].append(patch.reshape(-1))
            xS[j].append(HR[xP, yP, zP])

        print('\r{} / {}    last {} s '.format(t + 1, C.PIXEL_TYPE, '%.1f' % (time.time() - start)), end='', flush=True)
        start = time.time()

        # Compute Q, V
        for j in range(C.Q_TOTAL):
            if len(xS[j]) != 0:
                Q[j, t], V[j, t] = add_qv_jt(patchS[j], xS[j], Q[j, t], V[j, t], j, t)

        print('   QV', '%.1f' % (time.time() - start), 's', end='', flush=True)

    print(' ' * 23, 'last', '%.1f' % ((time.time() - filestart) / 60), 'min', end='', flush=True)

    finished_files.append(fileName)
    ask_save_qv(Q, V, finished_files)

    if idx != -1 and idx >= C.TRAIN_FILE_MAX - 1:
        break

save_qv(Q, V, finished_files)
compute_h(Q, V)