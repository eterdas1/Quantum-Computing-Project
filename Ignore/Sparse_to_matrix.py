def Sparse_Deconverter(x, cols):
    # Have to input columns manually as we haven't created an object yet

    mat = np.zeros((len(x), cols))

    for i in range(len(x)):
        for j in range(len(x[i])):
            mat[i, x[i][j][0]] = x[i][j][1]

    return mat
