import numpy as np






def Matrix_to_Sparse(a):
    "Converts a matrix into the sparse form of 3 arrays, first array containing the value, the second contains the i, the third contains the j"

    num_rows = a.shape[0]
    num_columns = a.shape[1]

    values = []
    columns = []
    rows = []
    
    
    for i in range(num_rows):
        for j in range(num_columns):
            if a[i,j] != 0:
                values.append(a[i,j])
                columns.append(j)
                rows.append(i)
    return np.array((np.array(values), np.array(rows), np.array(columns), np.array(a.shape)), dtype=object)





            
