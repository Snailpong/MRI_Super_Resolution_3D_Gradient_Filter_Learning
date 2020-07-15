import numpy as np
import os
import glob
import nibabel as nib
from PIL import Image
import sys
import matplotlib.pyplot as plt
import pathlib
import cv2
from scipy.ndimage import zoom

sys.path.append('../')

inputDir = 'C:/Users/User/Desktop/RAISR3D/NISR-master/train'
outputDir = 'C:/Users/User/Desktop/RAISR3D/NISR-master/train_low'

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

files = glob.glob(inputDir + "/*.nii.gz")

for filepath in files:
    print(filepath)
    file = nib.load(filepath).get_fdata()
    print('original Data shape is ' + str(file.shape) + ' .')
    im = file
    print (file.shape)
   
    downscale = zoom(file, (0.5, 0.5, 0.5))
    
    print (downscale.shape)

    fig = plt.figure()

    fig.add_subplot(3, 2, 1)
    plt.imshow(file[:, :, 130])
    plt.title('original 130')
    fig.add_subplot(3, 2, 2)
    plt.imshow(downscale[:, :, 65])
    plt.title('downsample 130')

    fig.add_subplot(3, 2, 3)
    plt.imshow(file[:, 130, :])
    plt.title('original 130')
    fig.add_subplot(3, 2, 4)
    plt.imshow(downscale[:, 65, :])
    plt.title('downsample 130')

    fig.add_subplot(3, 2, 5)
    plt.imshow(file[130, :, :])
    plt.title('original 130')
    fig.add_subplot(3, 2, 6)
    plt.imshow(downscale[65, :, :])
    plt.title('downsample 130')

    plt.show()

# def downsample_large_volume(img_path_list, input_voxel_size, output_voxel_size):

#     scale = input_voxel_size / output_voxel_size
#     resampled_zs = []

#     #Resample z slices
#     for img_path in img_path_list:
#         z_slice_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         z_slice_resized = cv2.resize(z_slice_arr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
#         resampled_zs.append(z_slice_resized) # Or save to disk to save RAM and use np.memmap for xz scaling


#     temp_arr = np.dstack(resampled_zs)  # We seem to be in yxz space now
#     final_scaled_slices = []

#     # Resample xz plane at each y
#     for y in range(temp_arr.shape[0]):
#         xz_pane = temp_arr[y, :, :]
#         scaled_xz = cv2.resize(xz_pane, (0, 0), fx=scale, fy=1, interpolation=cv2.INTER_AREA)
#         final_scaled_slices.append(scaled_xz)

#     final_array = np.dstack(final_scaled_slices)


#     img = sitk.GetImageFromArray(np.swapaxes(np.swapaxes(final_array, 0, 1), 1, 2))
#     sitk.WriteImage(img, 'scaled_by_pixel.nrrd')

# downsample_large_volume('C:/Users/User/Desktop/RAISR3D/Jalali-RAISR/r3d/T1w_acpc_dc_restore_brain_194847.nii.gz',2,1)


# data=np.array([[[1,3],[3,5]],[[8,5],[6,3]],[[5,1],[7,3]]])
# print (data)

# trilinear = RegularGridInterpolator((data[0],data[1],data[2]),data)
