import numpy as np
a = np.ones((2,2)) *2
b = np.ones((1,6))

def tensor_product(x,y):


    rows = [] #list where each element is a row in the final array



    for i in range(x.shape[0]):
        row_value = []
        for j in range(x.shape[1]):


            row_value.append(x[i,j] * y)

            if j == x.shape[1] - 1:
                rows.append(np.hstack(row_value)) #combines the row values into a single array and then appends it to the row list

    tp = np.vstack(rows) #vertical stacks all the rows

    return tp


print(tensor_product(a,b))


