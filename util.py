import cv2
import numpy as np
import os
import pickle
import sys
from cgls import cgls
from filterplot import filterplot
from gaussian2d import gaussian2d
from gettrainargs import gettrainargs
from hashkey import hashkey
from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import transform

def dog_sharpener(input, sigma=0.85, alpha=1.414, r=15, ksize=(3,3)):

    G1 = cv2.GaussianBlur(input, ksize, sigma)
    Ga1 = cv2.GaussianBlur(input, ksize, sigma*alpha)
    D1 = cv2.addWeighted(G1, 1+r, Ga1, -r, 0)

    G2 = cv2.GaussianBlur(Ga1, ksize, sigma)
    Ga2 = cv2.GaussianBlur(Ga1, ksize, sigma*alpha)
    D2 = cv2.addWeighted(G2, 1+r, Ga2, -r, 0)

    G3 = cv2.GaussianBlur(Ga2, ksize, sigma)
    Ga3 = cv2.GaussianBlur(Ga2, ksize, sigma * alpha)
    D3 = cv2.addWeighted(G3, 1+r, Ga3, -r, 0)

    B1 = Blending1(input, D3)
    B1 = Blending1(input, B1)
    B2 = Blending1(B1, D2)
    B2 = Blending1(input, B2)
    B3 = Blending1(B2, D1)
    B3 = Blending1(input, B3)

    output = B3

    return output

def blending_image(LR, HR):
    H,W = LR.shape
    H1,W1 = HR.shape
    assert H1==H and W1==W
    Census,CT = CT_descriptor(LR)
    blending1 = Census*HR + (1 - Census)*LR
    return blending1


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