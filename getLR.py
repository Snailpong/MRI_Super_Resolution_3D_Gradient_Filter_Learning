import numpy as np
import os
import glob
import nibabel as nib
from PIL import Image
import sys
import matplotlib.pyplot as plt
import pathlib

sys.path.append('../')

def modcrop(im,modulo):
    shape = im.shape
    size0 = shape[0] - shape[0] % modulo
    size1 = shape[1] - shape[1] % modulo
    size2 = shape[2] - shape[2] % modulo
    if len(im.shape) == 2:
        out = im[0:size0, 0:size1]
    else:
        out = im[0:size0, 0:size1, 0:size2]
    return out

inputDir = '.\\train'
outputDir = '.\\train_low'
#inputDir = pathlib.Path('./train')
#outputDir = pathlib.Path('./train_low')

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#files = inputDir.glob("/*.nii.gz")
#for filepath in list(files):
files = glob.glob(inputDir + "/*.nii.gz")
for filepath in files:
    print(filepath)
    file = nib.load(filepath).get_fdata()
    print('original Data shape is ' + str(file.shape) + ' .')
    im = file
    #im=modcrop(file,2)

    fig = plt.figure()

    imgfft = np.fft.fftn(im)
    imgfft_zero = np.zeros((imgfft.shape[0], imgfft.shape[1], imgfft.shape[2]))

    fig.add_subplot(2, 4, 1)
    #plt.imshow(im[:, 130, :])
    #plt.title('original 130 slice')

    #fig.add_subplot(2, 4, 2)
    #plt.imshow(20*np.log(abs(imgfft[:, :, 0]))) 
    #plt.title('after FFT')

    x_area=y_area=z_area=30
    #x_area = imgfft.shape[0] // 4
    #y_area = imgfft.shape[1] // 4
    #z_area = imgfft.shape[2] // 4

    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2

    imgfft_shift = np.fft.fftshift(imgfft)
    imgfft_shift2 = imgfft_shift.copy()

    fig.add_subplot(2, 4, 3)
    plt.imshow(20*np.log(abs(imgfft_shift[:, :, imgfft.shape[2] // 2]))) 
    #plt.title('after FFT shift')

    imgfft_shift[x_center-x_area : x_center+x_area, y_center-y_area : y_center+y_area, z_center-z_area : z_center+z_area] = 0
    imgfft_shift2 = imgfft_shift2 - imgfft_shift

    fig.add_subplot(2, 4, 6)
    #plt.imshow(20*np.log(abs(imgfft_shift2[:, :, imgfft.shape[2] // 2]))) 
    #plt.title('after FFT shift center 40x40')
    #print (imgfft_shift2.shape)

    imgifft3 = np.fft.ifftn(imgfft_shift2)
    #print (imgifft3.shape)
    img_out3 = abs(imgifft3)
    #img_out3 = np.flip(img_out3, axis=0)
    #fig.add_subplot(2, 4, 7)
    #plt.imshow(img_out3[:, 130, :])
    #plt.title('lr 130 (FFT shift)')

    # save lr nii file
    lr_nii = nib.Nifti1Image(img_out3, affine=np.eye(4))

    filename = filepath.split('\\')[-1]
    new_filename = 'LR_' + filename
    new_pathfile = os.path.join(outputDir, new_filename)
    lr_nii.header.get_xyzt_units()
    lr_nii.to_filename(new_pathfile)
    nib.save(lr_nii, new_pathfile)
    print('LR File ' + new_filename + ' is saved in ' + outputDir + ' .')

#plt.show()