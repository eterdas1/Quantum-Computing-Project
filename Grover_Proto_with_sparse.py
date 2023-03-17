"""I have edited this compared to the github version by adding the diffuser function that makes the diffuser from scratch  """

import numpy as np
import time

def Sparse_X_Multiply(x,a,o):
    if o ==1:
        "For order a,X"
        return np.array((np.multiply(x[0],a[0]), x[1], x[2], x[3]), dtype=object)


    elif o ==2:
        "For 2 matrices shape of X"
        return np.array((np.multiply(x[0], np.flipud(a[0])), x[1], np.flipud(x[2]), x[3]), dtype=object)    

    else:
        "For order X,a"
        return np.array((np.multiply(x[0], np.flipud(a[0])), x[1], x[2], x[3]), dtype=object)    

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
           
            current_col_indices = np.where(b_cols == i)
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
                    new_cols.append(i)

            
    return np.array((new_vals, new_rows, new_cols, dim), dtype=object)    



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

        
    return np.array((np.concatenate(new_vals), np.concatenate(new_rows), np.concatenate(new_cols), np.array(dim)), dtype=object)



    
    

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


           
            if np.sum(lis_row_vals*lis_col_vals) != 0:
                new_vals.append(np.sum(lis_row_vals*lis_col_vals))
                new_rows.append(i)
                new_cols.append(j)

    return np.array((np.array(new_vals), np.array(new_rows), np.array(new_cols), np.array(dim)), dtype=object)

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

def Hadmard_product(A, n): # Tensor products between Hadmards
    result = np.array([[1]])
    for i in range(n):
        result = np.kron(result, A)
    return result

t1 = time.time()

A = (1 / np.sqrt(2)) *np.array([[1, 1], [1, -1]]) # Defines the Hadamard
num_qubits = 3
num_iterations = int(np.pi/4 * np.sqrt(2**num_qubits)) # Calculates the number of iterations
target_in = 2  # Index grover is searching for


state = np.zeros((2**num_qubits)) # Creates the inital state
state[0] = 1


result = Hadmard_product(A, num_qubits) # Does Tensor product between 3 Hadmards
state = np.matmul(state,result) # Applies Hadmard to all qubits in our state



def diffuser(n):
        #make base gates
   

    MCZ = np.eye(2**n)
    MCZ[-1][-1] = -1
    X = np.array([0,1,1,0]).reshape(2,2)
    H = 1/(2**0.5) * np.array([1,1,1,-1]).reshape(2,2)
    t3 = time.time()
    #Turns matrices into sparse format
    MCZ = Matrix_to_Sparse(MCZ)
    X = Matrix_to_Sparse(X)
    
    
    #make tensor products of X gates together
    
    xProduct = X
    for i in range(n-1):
        xProduct = Sparse_Tensor_Product(xProduct,X)
   
    
    minus_xProduct = np.copy(xProduct)
    minus_xProduct[0] = minus_xProduct[0] *-1
    
    
   
    
    t = Sparse_X_Multiply(minus_xProduct, MCZ, 0)
    
    #make U naught
    U_0 = Sparse_X_Multiply(t, xProduct, 2)
    
    U_0 = Sparse_to_matrix(U_0)
    t4 = time.time()

    
    #make tensor products of hadamards together
    
    HProduct =  H
    
    for i in range(n-1):
        HProduct =  np.kron(HProduct,H)
        
    minus_HProduct = HProduct * -1
    
    
    
    U_s = np.matmul(np.matmul(minus_HProduct, U_0), HProduct)
    
  
    

    print(t4 - t3)
   
    return(U_s)


def oracle(state, target): # Oracle function flips the sign of the element we're looking for
    state[target] *= -1
    return state


diff = np.full((2**num_qubits, 2**num_qubits), 2/(2**num_qubits)) - np.eye(2**num_qubits) # Creates the diffuser matrix
diff1 = diffuser(num_qubits)


for i in range(num_iterations):
    for index in range(2**num_qubits): # Number of outputs
        if index == target_in: # Checks the index is equal to what we're searching for and then runs the oracle on that index
            state = oracle(state, index)
    state = np.dot(diff1, state) # Runs the diffuser on the state


print(state*state) # Prints the probability of each state

t2 = time.time()

print(t2-t1)

