import numpy as np

from New_Matrix_to_Sparse import Matrix_to_Sparse
from New_Sparse_to_Matrix import Sparse_to_matrix

a = np.zeros((3,2))
a[0,0] = 1
a[1,1] = 2
a[0,1] = 1


b = Matrix_to_Sparse(a)



def Sparse_Tensor_Product(a,b):

    "Does the tensor product between 2 matrices in our sparce form"

    dim = [a[3][0]*b[3][0], a[3][1]*b[3][1]]
    
    
    a_vals = np.array((a[0]))
    a_rows = np.array((a[1]))
    a_cols = np.array((a[2]))

    b_vals = np.array((b[0]))
    b_rows = np.array((b[1]))
    b_cols = np.array((b[2]))

    new_vals = []
    new_rows = []
    new_cols = []



    for i in range(len(a_vals)):
        new_vals.append((a_vals[i]*b_vals))
        new_rows.append((a_rows[i]* b[3][0] + b_rows))
        new_cols.append((a_cols[i]* b[3][1] + b_cols))

    #flat_vals = []
    #flat_rows = []
    #flat_cols = []
    #for i in range(len(new_vals)):
     #           for j in range(len(new_vals[0])):
      #              flat_vals.append(new_vals[i][j])
       #             flat_rows.append(new_rows[i][j])
        #            flat_cols.append(new_cols[i][j])

        
    return np.array((new_vals, new_rows, new_cols, dim), dtype=object)



tp = Sparse_Tensor_Product(b, b)

print(Sparse_to_matrix(tp))