import numpy as np
from scipy.ndimage import zoom


def get_lr_interpolation(hr):
    downscaled_lr = zoom(hr, 0.5, order=2)
    lr = zoom(downscaled_lr, 2, order=1)
    return lr

def get_lr_kspace(hr):
    imgfft = np.fft.fftn(hr)
    imgfft_zero = np.zeros((imgfft.shape[0], imgfft.shape[1], imgfft.shape[2]))

    x_area = y_area = z_area = 60

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