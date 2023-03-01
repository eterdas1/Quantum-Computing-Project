
import numpy as np



def Sparse_to_matrix(a):
    "Converts matrices in our sparse form back into a normal numpy array"

    num_rows = a.shape[0]
    num_cols = a.shape[1]
    b = np.zeros((3,4))

    for j in range(num_cols):
        row = int(a[1,j])
        col = int(a[2,j])
        
        b[row ,col] = a[0,j]

    return b
