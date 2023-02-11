import numpy as np
"""I think this will only work for square matrices but I don't think we need to deal with non-square ones anyway"""


def Sparse_Converter(mat):
    #number of rows in input matrix
    n = len(mat)
    #create empty matrix to store lists in
    sparse = np.empty(n, dtype = object)
    
    #i is columns, j is rows
    #the below code checks the entries in a column and stores the non-zero entries.
    #the row number of the entry is stored with the entry value.
    #outcome of this is an array of lists containing the non-zero rows and values.
   
    for i in range(n):
        #list to hold non-zero entries in a column
        col = []
        for j in range(n):
            if mat[i][j] != 0:
                col.append((j, mat[i][j]))

    return(sparse)
    
