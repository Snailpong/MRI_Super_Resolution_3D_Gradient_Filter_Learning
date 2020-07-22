import cv2
import numpy as np
import os
import pickle
import sys
import math

from scipy.ndimage.filters import convolve
from scipy.ndimage import zoom
from numba import jit, njit, cuda, prange, vectorize

from filterVariable import *

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

def dog_sharpener(input, sigma=0.85, alpha=1.414, r=15, ksize=(3,3,3)):
    G1 = gaussian_3d_blur(input, ksize, sigma)
    Ga1 = gaussian_3d_blur(input, ksize, sigma*alpha)
    D1 = add_weight(G1, 1+r, Ga1, -r, 0)

    G2 = gaussian_3d_blur(Ga1, ksize, sigma)
    Ga2 = gaussian_3d_blur(Ga1, ksize, sigma*alpha)
    D2 = add_weight(G2, 1+r, Ga2, -r, 0)

    G3 = gaussian_3d_blur(Ga2, ksize, sigma)
    Ga3 = gaussian_3d_blur(Ga2, ksize, sigma * alpha)
    D3 = add_weight(G3, 1+r, Ga3, -r, 0)


    B1 = blend_image(input, D3)
    B1 = blend_image(input, B1)
    B2 = blend_image(B1, D2)
    B2 = blend_image(input, B2)
    B3 = blend_image(B2, D1)
    B3 = blend_image(input, B3)

    output = B3
    clip_zero(output)

    return output

@njit(parallel=True)
def clip(im, low, high):
    H, W, D = im.shape
    for i in prange(H):
        for j in prange(W):
            for k in prange(D):
                if im[i, j, k] < low:
                    im[i, j, k] = low
                elif im[i, j, k] > high:
                    im[i, j, k] = high

@njit(parallel=True)
def clip_zero(im):
    H, W, D = im.shape
    for i in prange(H):
        for j in prange(W):
            for k in prange(D):
                if im[i, j, k] < 0:
                    im[i, j, k] = 0


@njit
def ct_descriptor(im):
    H, W, D = im.shape
    windowSize = 3
    Census = np.zeros((H, W, D))
    CT = np.zeros((H, W, D, windowSize, windowSize, windowSize))
    C = np.int((windowSize - 1) / 2)
    for i in range(C, H - C):
        for j in range(C, W - C):
            for k in range(C, D - C):
                cen = 0
                for a in range(-C, C + 1):
                    for b in range(-C, C + 1):
                        for c in range(-C, C + 1):
                            if not (a == 0 and b == 0 and c == 0):
                                if im[i + a, j + b, k + c] < im[i, j, k]:
                                    cen += 1
                                    CT[i, j, k, a + C, b + C, c + C] = 1
                Census[i, j, k] = cen
    Census = Census / 26
    return Census, CT

@njit
def blend_weight(LR, HR, ctLR, ctHR, threshold = 10):
    windowSize = 3
    H, W, D = ctLR.shape[:3]
    blended = np.zeros((H, W, D), dtype=np.float64)

    C = np.int((windowSize - 1) / 2)
    for i in range(C, H - C):
        for j in range(C, W - C):
            for k in range(C, D - C):
                dist = 0
                for a in range(-C, C + 1):
                    for b in range(-C, C + 1):
                        for c in range(-C, C + 1):
                            if not (a == 0 and b == 0 and c == 0):
                                if ctLR[i, j, k, a + C, b + C, c + C] != ctHR[i, j, k, a + C, b + C, c + C]:
                                    dist += 1
                if dist > threshold:
                    blended[i, j, k] = LR[i, j, k]
                else:
                    blended[i, j, k] = HR[i, j, k]
    return blended



    blended = census * HR + (1 - census) * LR
    return blended

@njit
def blend_image2(LR, HR):
    census, ct = ct_descriptor(LR)
    blended = census * HR + (1 - census) * LR
    return blended

@njit
def blend_image(LR, HR, threshold = 10):
    censusLR, ctLR = ct_descriptor(LR)
    censusHR, ctHR = ct_descriptor(HR)
    blended = blend_weight(LR, HR, ctLR, ctHR, threshold)
    return blended


def gaussian_3d_blur(input, ksize=(3,3,3), sigma=0.85):
    filter = gaussian_3d(ksize, sigma)
    output = convolve(input, filter)
    return output

@njit(parallel=True)
def ata_add(A, B):
    for i in prange(A.shape[1]):
        for j in prange(A.shape[1]):
            B[i, j] += A[0, i] * A[0, j]

def ata_add_cuda_all(A, B):
    threadsperblock = (32, 32)
    blockspergrid_x = int(math.ceil(A.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(A.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    A_dary = cuda.to_device(A)
    B_dary = cuda.device_array(B.shape, B.dtype)

    ata_add_cuda[blockspergrid, threadsperblock](A_dary, B_dary)
    B_dary.copy_to_host(B)

@cuda.jit
def ata_add_cuda(A, B):
    i, j = cuda.grid(2)
    if i < A.shape[1] and j < A.shape[1]:
        B[i, j] += A[0, i] * A[0, j]


def add_weight(i1, w1, i2, w2, bias):
    return np.dot(i1, w1) + np.dot(i2, w2) + bias


def dog_filter(sigma=0.85, alpha=1.414, rho=15):
    dog_sigma = np.dot(Gaussian3d(sigma = sigma), (1 + rho))
    dog_alpha = np.dot(Gaussian3d(sigma = sigma * alpha), rho)
    return dog_sigma - dog_alpha


def gaussian_3d(shape=(3,3,3), sigma=0.85):
    m,n,o = [(ss-1.)/2. for ss in shape]
    z, y, x = np.ogrid[-m:m+1,-n:n+1, -o:o+1]
    h = np.exp( -(x*x + y*y + z*z) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
