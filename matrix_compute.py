import numpy as np
import math
import time

from numba import jit, njit, cuda, prange, vectorize

from scipy.sparse.linalg import cg

from filter_constant import *


@njit(parallel=True)
def ata_add(A, B):
    for i in prange(A.shape[1]):
        for j in prange(A.shape[1]):
            B[i, j] += A[0, i] * A[0, j]

def ata_add_cuda_all(A, B):
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(A.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(A.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    A_dary = cuda.to_device(A)
    AT_dary = cuda.to_device(A.T)
    B_dary = cuda.device_array(B.shape, B.dtype)

    #fast_matmul[blockspergrid, threadsperblock](AT_dary, A_dary, B_dary)
    ata_add_cuda2[blockspergrid, threadsperblock](A_dary, B_dary)
    B_dary.copy_to_host(B)

@cuda.jit
def ata_add_cuda2(A, B):
    i, j = cuda.grid(2)
    if i < A.shape[1] and j < A.shape[1]:
        for k in range(A.shape[0]):
            B[i, j] += A[k, i] * A[k, j]

@cuda.jit
def ata_add_cuda(A, B):
    i, j = cuda.grid(2)
    if i < A.shape[1] and j < A.shape[1]:
        B[i, j] += A[0, i] * A[0, j]

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

def compute_h(Q, V):
    h = np.zeros((Q_TOTAL, PIXEL_TYPE, FILTER_VOL))

    print("\nComputing H...   ")
    start = time.time()
    for j in range(Q_TOTAL):
        for t in range(PIXEL_TYPE):
            print(j * PIXEL_TYPE + t, "/", Q_TOTAL * PIXEL_TYPE, end='\r', flush=True)
            h[j, t] = cg(Q[j, t], V[j, t])[0]

    np.save('./arrays/lowR4', h)
    print('Computing H is off in {} minutes'.format((time.time() - start) // 60))

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]