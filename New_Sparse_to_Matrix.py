
import numpy as np



def Sparse_to_matrix(a):
    "Converts matrices in our sparse form back into a normal numpy array"
    
    vals = np.array(a[0])
    rows = np.array(a[1])
    cols = np.array(a[2])

    
    
    
    b = np.zeros(a[3])
   

    for j in range(len(cols)):
        
        row = rows[j]
        col = cols[j]
        
        b[row ,col] = vals[j]

    return b
