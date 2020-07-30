import numpy
import os
import pickle

from filter_constant import *

def save_qv(Q, V, finished_files):
    np.save("./arrays/Q", Q)
    np.save("./arrays/V", V)
    with open('./arrays/finished_files.pkl', 'wb') as f:
        pickle.dump(finished_files, f)

def init_buckets():
    patchS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]
    xS = [[[] for i in range(PIXEL_TYPE)] for j in range(Q_TOTAL)]
    return patchS, xS

def load_files():
    # Construct an empty matrix Q, V uses the corresponding LR and HR
    if os.path.isfile('./arrays/Q.npy') and os.path.isfile('./arrays/V.npy'):
        print('Importing exist arrays...', end=' ', flush=True)
        Q = np.load("./arrays/Q.npy")
        V = np.load("./arrays/V.npy")
        with open('./arrays/finished_files.pkl', 'rb') as f:
            finished_files = pickle.load(f)
        print('Done', flush=True)
        
    else:
        Q = np.zeros((Q_TOTAL, PIXEL_TYPE, FILTER_VOL, FILTER_VOL))
        V = np.zeros((Q_TOTAL, PIXEL_TYPE, FILTER_VOL, 1))
        finished_files = []