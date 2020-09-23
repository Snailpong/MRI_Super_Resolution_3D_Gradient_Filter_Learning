import time

import cupy as cp
import numpy as np

import filter_constant as C

from crop_black import *
from filter_func import *
from get_lr import *
from hashtable import *
from matrix_compute import *
from util import *

C.argument_parse()
determine_geometric_func()

C.R = 4
C.USE_PIXEL_TYPE = False

Q, V, finished_files = load_files()

stre = np.zeros((C.Q_STRENGTH - 1))  # Strength boundary
cohe = np.zeros((C.Q_COHERENCE - 1)) # Coherence boundary

trainPath = './train/'

file_list = make_dataset(trainPath)
C.TRAIN_FILE_MAX = min(C.TRAIN_FILE_MAX, len(file_list))

# Preprocessing normalized Gaussian matrix W for hashkey calculation
G_WEIGHT = get_normalized_gaussian()

instance = 5000000                          # use 20000000 patches to get the Strength and coherence boundary
patchNumber = 0                              # patch number has been used
quantization = np.zeros((instance, 2))        # quantization boundary
for file_idx, image in enumerate(file_list):
    print('\r', end='')
    print('' * 60, end='')
    print('\r Quantization: Processing '+ image.split('\\')[-1] + str(instance) + ' patches (' + str(100*patchNumber/instance) + '%)')

    raw_image = nib.load(image).dataobj
    crop_image = mod_crop(raw_image, C.R)
    clipped_image = clip_image(crop_image)
    slice_area = crop_slice(clipped_image, C.PATCH_SIZE // 2, C.R)

    im_LR = get_lr(clipped_image)         # Prepare the cheap-upscaling images (optional: JPEG compression)

    im_blank_LR = get_lr(clipped_image) / clipped_image.max()  # Prepare the cheap-upscaling images
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
for i in range(C.Q_COHERENCE - 1):
    cohe[i] = quantization[floor((i+1) * patchNumber / C.Q_COHERENCE), 1]

# stre[0] = 0.00075061
# stre[1] = 0.00297238
# cohe[0] = 0.42785409
# cohe[1] = 0.61220482

# stre[0] = 0.0010374
# stre[1] = 0.0031624
# cohe[0] = 0.44441757
# cohe[1] = 0.62840797

    im_HR, im_LR = get_train_image(file)

    quantization, patchNumber = quantization_border(im_LR, patchNumber, quantization)  # get the strength and coherence of each patch
    if patchNumber > C.TOTAL_SAMPLE_BORDER / 2:
        break

Q = cp.array(Q)
V = cp.array(V)

# stre[0] = 0.00075061
# stre[1] = 0.00297238
# cohe[0] = 0.42785409
# cohe[1] = 0.61220482

print(stre, cohe)

for file_idx, file in enumerate(file_list):
    file_name = file.split('\\')[-1].split('.')[0]
    file_timer = time.time()

    if file in finished_files:
        continue

    print('\rProcessing ' + str(file_idx + 1) + '/' + str(len(file_list)) + ' image (' + file_name + ')')

    raw_image = nib.load(file).dataobj
    crop_image = mod_crop(raw_image, C.R)
    clipped_image = clip_image(crop_image)
    slice_area = crop_slice(clipped_image, C.PATCH_HALF, C.R)

    im_blank_LR = get_lr(clipped_image) / clipped_image.max()  # Prepare the cheap-upscaling images
    im_LR = im_blank_LR[slice_area]
    im_GX, im_GY, im_GZ = np.gradient(im_LR)  # Calculate the gradient images
    im_HR = clipped_image[slice_area] / clipped_image.max()

    Q, V, mark = train_qv(im_LR, im_HR, im_GX, im_GY, im_GZ, G_WEIGHT, stre, cohe, Q, V, mark)  # get Q, V of each patch
    
    print(' ' * 30, 'last', '%.1f' % ((time.time() - filestart) / 60), 'min', end='', flush=True)

    finished_files.append(file)

    if file_idx + 1 == 25:
        break

    # ask_save_qv(Q, V, finished_files)

Q = Q.get()
V = V.get()

save_qv(Q, V, finished_files)
save_boundary(stre, cohe)
compute_h(Q, V)
