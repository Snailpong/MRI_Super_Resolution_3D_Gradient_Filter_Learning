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

Q, V, finished_files = load_files()

stre = np.zeros((C.Q_STRENGTH - 1))  # Strength boundary
cohe = np.zeros((C.Q_COHERENCE - 1)) # Coherence boundary

trainPath = './train/'

file_list = make_dataset(trainPath)
C.TRAIN_FILE_MAX = min(C.TRAIN_FILE_MAX, len(file_list))

patchNumber = 0                              # patch number has been used
quantization = np.zeros((C.TOTAL_SAMPLE_BORDER, 2))        # quantization boundary
for file_idx, file in enumerate(file_list):
    print('\r Quantization: Processing ' + file.split('\\')[-1] + str(C.TOTAL_SAMPLE_BORDER) + ' patches (' + str(100 * patchNumber / C.TOTAL_SAMPLE_BORDER) + '%)')

    im_HR, im_LR = get_train_image(file)

    quantization, patchNumber = quantization_border(im_LR, patchNumber, quantization)  # get the strength and coherence of each patch
    if patchNumber > C.TOTAL_SAMPLE_BORDER / 2:
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

print(stre, cohe)

for file_idx, file in enumerate(file_list):
    file_name = file.split('\\')[-1].split('.')[0]
    file_timer = time.time()

    if file in finished_files:
        continue

    print('\rProcessing ' + str(file_idx + 1) + '/' + str(len(file_list)) + ' image (' + file_name + ')')

    im_HR, im_LR = get_train_image(file)
    Q, V, mark = train_qv(im_LR, im_HR, stre, cohe, Q, V)  # get Q, V of each patch

    if file_idx + 1 == 100:
        break
    
    print(' ' * 23, 'last', '%.1f' % ((time.time() - file_timer) / 60), 'min', end='', flush=True)

    finished_files.append(file)
    ask_save_qv(Q, V, finished_files)

save_qv(Q, V, finished_files)
save_boundary(stre, cohe)
compute_h(Q, V)
