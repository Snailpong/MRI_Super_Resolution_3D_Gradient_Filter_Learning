import numpy as np
import math

from scipy.ndimage.filters import convolve
from scipy.ndimage import zoom
from numba import jit, njit, cuda, prange, vectorize, float32
from scipy.sparse.linalg import cg

import filter_constant as C


def dog_sharpener(im, sigma=0.85, alpha=1.414, r=15, k_size=(3,3,3)):
    G1 = gaussian_3d_blur(im, k_size, sigma)
    Ga1 = gaussian_3d_blur(im, k_size, sigma*alpha)
    D1 = add_weight(G1, 1+r, Ga1, -r, 0)

    G2 = gaussian_3d_blur(Ga1, k_size, sigma)
    Ga2 = gaussian_3d_blur(Ga1, k_size, sigma*alpha)
    D2 = add_weight(G2, 1+r, Ga2, -r, 0)

    G3 = gaussian_3d_blur(Ga2, k_size, sigma)
    Ga3 = gaussian_3d_blur(Ga2, k_size, sigma * alpha)
    D3 = add_weight(G3, 1+r, Ga3, -r, 0)

    B1 = blend_image(im, D3)
    B1 = blend_image(im, B1)
    B2 = blend_image(B1, D2)
    B2 = blend_image(im, B2)
    B3 = blend_image(B2, D1)
    B3 = blend_image(im, B3)

    output = np.clip(B3, 0, 1)

    return output


def clip_image(im):
    clip_value = np.sort(im.ravel())[int(np.prod(im.shape) * 0.999)]
    im = np.clip(im, 0, clip_value)
    return im


@njit(parallel=True)
def ct_descriptor(im):
    H, W, D = im.shape
    windowSize = 3
    Census = np.zeros((H, W, D))
    CT = np.zeros((H, W, D, windowSize, windowSize, windowSize))
    C = np.int((windowSize - 1) / 2)
    for i in prange(C, H - C):
        for j in prange(C, W - C):
            for k in prange(C, D - C):
                cen = 0
                for a in prange(-C, C + 1):
                    for b in prange(-C, C + 1):
                        for c in prange(-C, C + 1):
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


@njit
def blend_image(LR, HR, threshold = 10):
    censusLR, ctLR = ct_descriptor(LR)
    censusHR, ctHR = ct_descriptor(HR)
    blended = blend_weight(LR, HR, ctLR, ctHR, threshold)
    return blended


@njit
def blend_image2(LR, SR, threshold = 10):
    H, W, D = LR.shape
    blended = SR.copy()
    print(blended.shape)
    window_size = 3
    a = int((window_size - 1) / 2)

    for i in range(a, H - a):
        for j in range(a, W - a):
            for k in range(a, D - a):
                cur = np.sort(SR[i-a: i+a+1, j-a: j+a+1, k-a: k+a+1].ravel())
                # cur = cur[2:27-2]

                if cur[0] > SR[i, j, k] or cur[-1] < SR[i, j, k]:
                    blended[i, j, k] = LR[i, j, k]
    # blended = blend_weight(LR, HR, ctLR, ctHR, threshold)
    return blended


@njit
def blend_image3(LR, SR, threshold = 3):
    H, W, D = LR.shape
    blended = SR.copy()
    windowSize = 3
    C = np.int((windowSize - 1) / 2)

    for i in range(a, H - a):
        for j in range(a, W - a):
            for k in range(a, D - a):
                std_sr = np.std(LR[i-a: i+a+1, j-a: j+a+1, k-a: k+a+1].ravel())

                if abs(LR[i, j, k] - SR[i, j, k]) > std_sr * threshold:
                    blended[i, j, k] = LR[i, j, k]
    # blended = blend_weight(LR, HR, ctLR, ctHR, threshold)
    return blended


def gaussian_3d_blur(im, k_size=(3, 3, 3), sigma=0.85):
    g_filter = gaussian_3d(k_size, sigma)
    output = convolve(im, g_filter)
    return output



def gaussian_3d(shape=(3, 3, 3), sigma=0.85):
    m,n,o = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-m:m+1,-n:n+1, -o:o+1]
    h = np.exp(-(x*x + y*y + z*z) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_normalized_gaussian():
    weight = gaussian_3d((C.GRADIENT_SIZE, C.GRADIENT_SIZE, C.GRADIENT_SIZE))
    weight = np.diag(weight.ravel())
    weight = np.array(weight, dtype=np.float32)
    return weight


def clipped_hr(hr):
    clip_max = np.sort(hr.ravel())[int(np.prod(hr.shape) * 0.999)]
    hr = np.clip(hr, 0, clip_max)
    return hr


def normalization_hr(hr):
    hr = clipped_hr(hr)
    return hr / hr.max()
