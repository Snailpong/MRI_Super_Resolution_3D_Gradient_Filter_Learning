import numpy as np

def crop_black(array):
    array = array.copy()

    if str(type(array)) == '<class \'cupy.core.core.ndarray\'>':
        array = array.get()

    #print ('zero ratio:', np.count_nonzero(array==0)*100 / ((array.shape[0])*(array.shape[1])*(array.shape[2])))

    x_use=[]
    y_use=[]
    z_use=[]

    for i in range (array.shape[0]):
        if np.all(array[i,:,:] == 0) == False:
            x_use.append(i)
            break
    for i in reversed(range(array.shape[0])):
        if np.all(array[i,:,:] == 0) == False:
            x_use.append(i + 1)
            break

    for i in range (array.shape[1]):
        if np.all(array[:,i,:] == 0) == False:
            y_use.append(i)
            break
    for i in reversed(range(array.shape[1])):
        if np.all(array[:,i,:] == 0) == False:
            y_use.append(i + 1)
            break
    
    for i in range (array.shape[2]):
        if np.all(array[:,:,i] == 0) == False:
            z_use.append(i)
            break
    for i in reversed(range(array.shape[2])):
        if np.all(array[:,:,i] == 0) == False:
            z_use.append(i + 1)
            break

    return x_use, y_use, z_use