
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
determine_geometric_func()

Q, V, finished_files = load_files()

fileList = [file for file in glob.glob(C.TRAIN_GLOB)]
C.TRAIN_FILE_MAX = min(C.TRAIN_FILE_MAX, len(fileList))

# Preprocessing normalized Gaussian matrix W for hashkey calculation
G_WEIGHT = get_normalized_gaussian()

start = time.time()

for idx, file in enumerate(fileList):
    filestart = time.time()

    fileName = file.split('/')[-1].split('\\')[-1]
    if fileName in finished_files:
        continue

    print('\r[{} / {}]    {}'.format(idx+1, C.TRAIN_FILE_MAX, fileName))

    HR = nib.load(file).dataobj[:, :-1, :]  # Load NIfTI Image
    HR = normalization_hr(HR)               # Normalized to [0, 1]

    print('Making LR...', end='', flush=True)
    LR = get_lr(HR)

    print('\rSharpening...', end='', flush=True)    # Dog-Sharpening
    # HR = dog_sharpener(HR)

    [Lgx, Lgy, Lgz] = np.gradient(LR)
    sampled_list = get_sampled_point_list(HR)

    for split_idx, points in enumerate(sampled_list):
        print('\r{} / {}'.format(split_idx + 1, C.PIXEL_TYPE), end='', flush=True)
        start = time.time()
        patchS, xS = init_buckets()

        if C.USE_PIXEL_TYPE:
            t = split_idx
        else:
            t = 0

        for point_idx in prange(len(points)):
            xP, yP, zP = points[point_idx]
            patch = get_patch(LR, xP, yP, zP)

            if not np.any(patch):
                continue

            gx, gy, gz = get_gxyz(Lgx, Lgy, Lgz, xP, yP, zP)

            # Computational characteristics
            angle_p, angle_t, strength, coherence = hashtable(gx, gy, gz, G_WEIGHT)
            # angle_p, angle_t, strength, coherence = get_features(gx, gy, gz, weight)

            # Compressed vector space
            j = angle_p * C.Q_ANGLE_T * C.Q_COHERENCE * C.Q_STRENGTH + angle_t * C.Q_COHERENCE * C.Q_STRENGTH + strength * C.Q_COHERENCE + coherence
            
            pk = patch.reshape(-1)
            x = HR[xP, yP, zP]

            patchS[j].append(pk)
            xS[j].append(x)

            
        print('\r{} / {}    last {} s '.format(split_idx + 1, C.PIXEL_TYPE, '%.3f' % (time.time() - start)), end='', flush=True)
        start = time.time()

        for j in range(C.Q_TOTAL):
                if len(xS[j]) != 0:
                    time11 = time.time()
                    A = cp.array(patchS[j])
                    b = cp.array(xS[j]).reshape(-1, 1)
                    Qa = cp.array(Q[j, t])
                    Va = cp.array(V[j, t])

                    Qa += cp.dot(A.T, A)
                    Va += cp.dot(A.T, b)

                    Q[j, t] = Qa.get()
                    V[j, t] = Va.get()

        
        print('   QV', '%.3f' % (time.time() - start), 's', end='', flush=True)

    finished_files.append(fileName)
    
    print(' ' * 23, 'last', '%.1f' % ((time.time() - filestart) / 60), 'min', end='', flush=True)

    finished_files.append(fileName)
    ask_save_qv(Q, V, finished_files)

save_qv(Q, V, finished_files)
compute_h(Q, V)