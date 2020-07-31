import glob
import time
import random

import cupy as cp
import numpy as np
from numba import jit, prange

import nibabel as nib

from crop_black import *
from filter_constant import *
from filter_func import *
from get_lr import *
from hashtable import *
from matrix_compute import *
from util import *

Q, V, finished_files = load_files()

dataDir="./train/*"

fileList = [file for file in glob.glob(dataDir) if file.endswith(".nii.gz")]

# Preprocessing normalized Gaussian matrix W for hashkey calculation
weight = get_normalized_gaussian()

start = time.time()

for idx, file in enumerate(fileList):
    filestart = time.time()

    fileName = file.split('/')[-1].split('\\')[-1]
    if fileName in finished_files:
        continue

    print('\r[' + str(idx+1), '/', str(len(fileList)) + ']   ', fileName)

    HR = nib.load(file).dataobj[:, :-1, :]  # Load NIfTI Image
    HR = normalization_hr(HR)               # Normalized to [0, 1]

    print('Making LR...', end='', flush=True)
    #LR = get_lr_kspace(HR)         # Using Frequency domain
    LR = get_lr_interpolation(HR)   # Using Image domain

    # Dog-Sharpening
    print('\rSharpening...', end='', flush=True)
    HR = dog_sharpener(HR)

    print('\rSampling...', end='', flush=True)
    [Lgx, Lgy, Lgz] = np.gradient(LR)
    sampled_list = get_sampled_point_list(HR)

    for t, points in enumerate(sampled_list):

        print('\r{} / {}'.format(t + 1, PIXEL_TYPE), end='', flush=True)
        start = time.time()
        patchS, xS = init_buckets()

        for point_idx in prange(len(points)):
            xP, yP, zP = points[point_idx]
            patch = LR[xP - FILTER_HALF: xP + (FILTER_HALF + 1), yP - FILTER_HALF: yP + (FILTER_HALF + 1),
                    zP - FILTER_HALF: zP + (FILTER_HALF + 1)]

            if not np.any(patch):
                    continue

            gx = Lgx[xP - GRAD_HALF: xP + (GRAD_HALF + 1), yP - GRAD_HALF: yP + (GRAD_HALF + 1),
                    zP - GRAD_HALF: zP + (GRAD_HALF + 1)]
            gy = Lgy[xP - GRAD_HALF: xP + (GRAD_HALF + 1), yP - GRAD_HALF: yP + (GRAD_HALF + 1),
                    zP - GRAD_HALF: zP + (GRAD_HALF + 1)]
            gz = Lgz[xP - GRAD_HALF: xP + (GRAD_HALF + 1), yP - GRAD_HALF: yP + (GRAD_HALF + 1),
                    zP - GRAD_HALF: zP + (GRAD_HALF + 1)]

            # Computational characteristics
            angle_p, angle_t, strength, coherence = hashtable(gx, gy, gz, weight)
            # angle_p, angle_t, strength, coherence = get_features(gx, gy, gz, weight)

            # Compressed vector space
            j = angle_p * Q_ANGLE_T * Q_COHERENCE * Q_STRENGTH + angle_t * Q_COHERENCE * Q_STRENGTH + strength * Q_COHERENCE + coherence
            
            pk = patch.reshape(-1)
            x = HR[xP, yP, zP]

            patchS[j].append(pk)
            xS[j].append(x)

            
        print('\r{} / {}    last {} s '.format(t + 1, PIXEL_TYPE, '%.3f' % (time.time() - start)), end='', flush=True)
        start = time.time()
        # check1 = 0
        # check2 = 0
        # check3 = 0
        # check4 = 0
        # Compute Q, V

        for j in range(Q_TOTAL):
            if len(xS[j]) != 0:
                # time11 = time.time()
                A = cp.array(patchS[j])
                b = cp.array(xS[j]).reshape(-1, 1)
                Qa = cp.array(Q[j, t])
                Va = cp.array(V[j, t])
                # check1 += time.time() - time11
                # time11 = time.time()

                Qa += cp.dot(A.T, A)
                Va += cp.dot(A.T, b)

                # check2 += time.time() - time11
                # time11 = time.time()

                Q[j, t] = Qa.get()
                V[j, t] = Va.get()

                # check3 += time.time() - time11
                # time11 = time.time()
                # check4 += time.time() - time11

        # print('\n', check1, check2, check3, check4)

        
        print('   QV', '%.3f' % (time.time() - start), 's', end='', flush=True)

    finished_files.append(fileName)
    
    print(' ' * 23, 'last', '%.1f' % ((time.time() - filestart) / 60), 'min', end='', flush=True)

    ask_save_qv(Q, V, finished_files)

    if(idx == 9):
        break
    

if str(type(Q)) == '<class \'cupy.core.core.ndarray\'>':
    Q = Q.get()
    V = V.get()

save_qv(Q, V, finished_files)
compute_h(Q, V)

