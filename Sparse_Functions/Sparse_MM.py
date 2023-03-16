import numpy as np
from New_Matrix_to_Sparse import Matrix_to_Sparse
from New_Sparse_to_Matrix import Sparse_to_matrix

a = np.zeros((3,3))
a[0,0] = 1
a[1,1] = 2
a[2,0] = 3


print(a)


b = Matrix_to_Sparse(a)

def Sparse_Matrix_Multiply(a,b):

    if a[3][0] != b[3][1]:
        raise ValueError("Matrices are incorrect shape")

    dim = [np.min((a[3][0], b[3][0])),np.min((a[3][1], b[3][1]))]
    
    
    a_vals = np.array((a[0]))
    a_rows = np.array((a[1]))
    a_cols = np.array((a[2]))
    
    b_vals = np.array((b[0]))
    b_rows = np.array((b[1]))
    b_cols = np.array((b[2]))

    new_vals = []
    new_rows = []
    new_cols = []

    for i in range(dim[0]):
        current_row_indices = np.where(a_rows == i)
        current_row_values = a_vals[current_row_indices]
        if current_row_indices[0].shape[0] != 0:
            Js = a_cols[current_row_indices] #These are the j values on the row we're multiplying by
           
            for j in range(dim[1]):
                current_col_indices = np.where(b_cols == j)
                current_column_values = b_vals[current_col_indices]

                if current_col_indices[0].shape[0] != 0:

                    Is = b_rows[current_col_indices] #These are the i values on the column we're multiplying by
                
                    final_indices = np.equal(Js,Is)

                
                    if len(Is) > len(Js):

                        final = Is[final_indices]
                    
                    else:    
                        final = Js[final_indices]

                    if final.shape[0] != 0:
                       
                        

                        value = sum(a_vals[np.where(a_cols == final)] * b_vals[np.where(b_rows == final)])
                        new_vals.append(value)
                        new_rows.append(i)
                        new_cols.append(j)

            
    return np.array((new_vals, new_rows, new_cols, dim), dtype=object)       
          

            

print(Sparse_to_matrix( Sparse_Matrix_Multiply(b, b)))               





