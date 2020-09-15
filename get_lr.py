import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

from crop_black import *
from filter_func import *
import filter_constant as C


def get_train_image(file):
    raw_image = nib.load(file).dataobj
    crop_image = mod_crop(raw_image)
    clipped_image = clip_image(crop_image)
    slice_area = crop_slice(clipped_image)

    im_blank_LR = get_lr_image(clipped_image) / clipped_image.max()  # Prepare the cheap-upscaling images
    im_LR = im_blank_LR[slice_area]
    im_HR = clipped_image[slice_area] / clipped_image.max()

    return im_HR, im_LR


def get_lr_image(hr):
    if C.LR_TYPE == 'interpolation':
        lr = get_lr_interpolation(hr)   # Using Image domain
    else:
        lr = get_lr_kspace2(hr)          # Using Frequency domain
    return lr
    

def get_lr_interpolation(im):
    downscaled_lr = zoom(im, 1.0 / C.R, order=2, prefilter=False)
    lr = np.clip(zoom(downscaled_lr, C.R, order=2, prefilter=False), 0, im.max())
    lr[np.where(im == 0)] = 0
    return lr


def get_lr_kspace(hr):
    imgfft = np.fft.fftn(hr)
    imgfft_zero = np.zeros((imgfft.shape[0], imgfft.shape[1], imgfft.shape[2]))

    x_area = y_area = z_area = 50

    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2

    imgfft_shift = np.fft.fftshift(imgfft)
    imgfft_shift2 = imgfft_shift.copy()

    imgfft_shift[x_center-x_area : x_center+x_area, y_center-y_area : y_center+y_area, z_center-z_area : z_center+z_area] = 0
    imgfft_shift2 = imgfft_shift2 - imgfft_shift

    imgifft3 = np.fft.ifftn(imgfft_shift2)
    lr = abs(imgifft3)
    return lr


def get_lr_kspace2(hr):
    imgfft = np.fft.fftn(hr)
    imgfft_zero = np.zeros((imgfft.shape[0], imgfft.shape[1], imgfft.shape[2]))

    ratio=6
    x_area = int(hr.shape[0]/ratio)
    y_area = int(hr.shape[1]/ratio)
    z_area = int(hr.shape[2]/ratio)

    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2

    imgfft_shift = np.fft.fftshift(imgfft)
    imgfft_shift2 = imgfft_shift.copy()

    imgfft_shift[x_center-x_area : x_center+x_area, y_center-y_area : y_center+y_area, z_center-z_area : z_center+z_area] = 0
    imgfft_shift2 = imgfft_shift2 - imgfft_shift

    imgifft3 = np.fft.ifftn(imgfft_shift2)
    lr = abs(imgifft3)
    return lr