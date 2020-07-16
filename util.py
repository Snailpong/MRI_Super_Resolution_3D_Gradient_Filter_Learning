import cv2
import numpy as np
import os
import pickle
import sys
#from cgls import cgls
#from filterplot import filterplot
#from gaussian2d import gaussian2d
#from gettrainargs import gettrainargs
#from hashkey import hashkey
#from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform
from scipy.ndimage.filters import convolve
from numba import jit

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

    return output

@jit(nopython=True)
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
    return Census

@jit(nopython=True)
def blend_image(LR, HR):
    census = ct_descriptor(LR)
    blended = census * HR + (1 - census) * LR
    return blended


def gaussian_3d_blur(input, ksize=(3,3,3), sigma=0.85):
    filter = gaussian_3d(ksize, sigma)
    output = convolve(input, filter)
    return output


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