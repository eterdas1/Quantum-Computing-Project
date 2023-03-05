import numpy as np
from New_Matrix_to_Sparse import Matrix_to_Sparse
from New_Sparse_to_Matrix import Sparse_to_matrix

a = np.zeros((3,3))
a[0,0] = 1
a[1,1] = 2
a[0,1] = 1

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
        for j in range(dim[1]):
            "Need to fix when there are zeros on either the row or column"
            row_ind = np.where(a_rows == i)[0]       
            col_ind = np.where(b_cols == j)[0]
            row_vals = a_vals[row_ind] # values on the row we're currently multiplying
            col_vals = b_vals[col_ind] # values on the column we're currently multiplying

            

            j_vals = a_cols[row_ind]  # the j values of the row 
            
            i_vals = b_rows[col_ind]  # i values of the corresponding column 
            

            lis_ij = []   # Stores the i/j values that have a non zero in the corresponding row or column

            for k in i_vals:
                l = np.where(j_vals == k)[0]
                if l.size > 0:
                    
                    lis_ij.append(j_vals[l][0])

            lis_row_vals = []   # Final values of the row and columns we're multiplying together
            lis_col_vals = []    
            #print(lis_ij)
            for g in range(len(lis_ij)):
                #print(lis_ij)
                k = np.where(j_vals == lis_ij[g])
                
                
                t = np.where(i_vals == lis_ij[g])
                if k != 0:
                    
                    lis_row_vals.append( row_vals[k])
                
                if t != 0:
                    lis_col_vals.append( col_vals[t])    

            

            

            lis_row_vals = np.array(lis_row_vals)
            lis_col_vals = np.array(lis_col_vals)


           
            new_vals.append(np.sum(lis_row_vals*lis_col_vals))
            new_rows.append(i)
            new_cols.append(j)

    return np.array((new_vals, new_rows, new_cols, dim), dtype=object)




print(Sparse_to_matrix( Sparse_Matrix_Multiply(b, b)))
