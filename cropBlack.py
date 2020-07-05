import numpy as np
import glob


def cropBlack(array):
    print('original Data shape is ' + str(array.shape) + ' .')
    array = array.copy()
    array = array.round(out=array).astype(np.uint8)

    idx=[]
    x_use=[0, array.shape[0]]
    for i in range (array.shape[0]):    
        if (np.sum(array[i,:,:])) == 0:
            idx.append(i)
    for i in range (len(idx)-1):
        if (idx[i+1]-idx[i]) != 1:
            x_use[0] = idx[i]+1
            x_use[1] = idx[i+1]

    idx2=[]
    y_use=[0, array.shape[1]]
    for i in range (array.shape[1]):    
        if (np.sum(array[:,i,:])) == 0:
            idx2.append(i)
    for i in range (len(idx2)-1):
        if (idx2[i+1]-idx2[i]) != 1:
            y_use[0] = idx2[i]+1
            y_use[1] = idx2[i+1]
  
    idx3=[]
    z_use=[0, array.shape[2]]
    for i in range (array.shape[2]):    
        if (np.sum(array[:,:,i])) == 0:
            idx3.append(i)
    for i in range (len(idx3)-1):
        if (idx3[i+1]-idx3[i]) != 1:
            z_use[0] = idx3[i]+1
            z_use[1] = idx3[i+1]

    return x_use, y_use, z_use
