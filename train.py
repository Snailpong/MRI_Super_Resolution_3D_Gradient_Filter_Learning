import glob
import time
import random

import cupy as cp
import numpy as np

import nibabel as nib

import filter_constant as C

from crop_black import *
from filter_func import *
from get_lr import *
from hashtable import *
from matrix_compute import *
from util import *

C.argument_parse()

Q, V, finished_files, count = load_files()

stre = np.zeros((C.Q_STRENGTH - 1))  # Strength boundary
cohe = np.zeros((C.Q_COHERENCE - 1)) # Coherence boundary

file_list = make_dataset('./train')
C.TRAIN_FILE_MAX = min(C.TRAIN_FILE_MAX, len(file_list))

# Preprocessing normalized Gaussian matrix W for hashkey calculation
G_WEIGHT = get_normalized_gaussian()

instance = 5000000
patchNumber = 0
quantization = np.zeros((instance, 2))
for file_idx, image in enumerate(file_list):
    print('\r', end='')
    print('' * 60, end='')
    print('\r Quantization: Processing '+ image.split('\\')[-1] + str(instance) + ' patches (' + str(100*patchNumber/instance) + '%)')

    raw_image = np.array(nib.load(image).get_fdata(), dtype=np.float32)
    clipped_image = clip_image(raw_image)
    im = mod_crop(clipped_image, C.R)

    slice_area = crop_slice(im, C.PATCH_SIZE // 2, C.R)

    im_LR = get_lr(im)         # Prepare the cheap-upscaling images (optional: JPEG compression)

    im_blank_LR = get_lr(im) / im.max()  # Prepare the cheap-upscaling images
    im_LR = im_blank_LR[slice_area]
    im_GX, im_GY, im_GZ = np.gradient(im_LR)  # Calculate the gradient images

    quantization, patchNumber = quantization_border(im_LR, im_GX, im_GY, im_GZ, patchNumber, G_WEIGHT, quantization, instance)  # get the strength and coherence of each patch
    if patchNumber > instance / 2:
        break

# uniform quantization of patches, get the optimized strength and coherence boundaries
quantization = quantization[0:patchNumber, :]
quantization = np.sort(quantization, axis=0)

for i in range(C.Q_STRENGTH - 1):
    stre[i] = quantization[floor((i+1) * patchNumber / C.Q_STRENGTH), 0]
    cohe[i] = quantization[floor((i+1) * patchNumber / C.Q_COHERENCE), 1]

# stre[0] = 0.00103732
# stre[1] = 0.00316019
# cohe[0] = 0.27704942
# cohe[1] = 0.49028277

print(stre, cohe)

Q = cp.array(Q)
V = cp.array(V)

start = time.time()

for file_idx, file in enumerate(file_list):
    file_name = file.split('\\')[-1].split('.')[0]
    filestart = time.time()

    if file_idx >= 100:
        break

    if file in finished_files:
        continue

    print('\rProcessing ' + str(file_idx + 1) + '/' + str(len(file_list)) + ' image (' + file_name + ')')

    raw_image = np.array(nib.load(file).get_fdata(), dtype=np.float32)
    clipped_image = clip_image(raw_image)
    im = mod_crop(clipped_image, C.R)
    slice_area = crop_slice(im, C.PATCH_HALF, C.R)

    im_blank_LR = get_lr(im) / im.max()  # Prepare the cheap-upscaling images
    im_LR = im_blank_LR[slice_area]
    im_HR = im[slice_area] / im.max()

    Q, V, count = train_qv(im_LR, im_HR, G_WEIGHT, stre, cohe, Q, V, count)  # get Q, V of each patch
    
    print(' ' * 30, 'last', '%.1f' % ((time.time() - filestart) / 60), 'min', end='', flush=True)

    finished_files.append(file)
    # ask_save_qv(Q, V, finished_files)

Q = Q.get()
V = V.get()

# save_qv(Q, V, finished_files, count)
compute_h(Q, V)

with open("./arrays/Qfactor_str" + str(C.R), "wb") as sp:
    pickle.dump(stre, sp)

with open("./arrays/Qfactor_coh" + str(C.R), "wb") as cp:
    pickle.dump(cohe, cp)