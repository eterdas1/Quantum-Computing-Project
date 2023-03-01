import numpy as np

a = np.zeros((3,4))
a[0,0] = 1

a[2,1] = 2
a[2,3] = 2

print(a)



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
    return np.array((values, rows, columns, a.shape), dtype=object)



sp = Matrix_to_Sparse(a)


print(sp)

            