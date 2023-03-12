"""
'Generalised cNOT'

This code is for acting a cNOT gate on qubits that are not adjacent to eachother.

Function takes control and target qubit indices (0,1,2,..,N-1) and N the number of qubits.
It returns a 2^N square matrix.

The cNOT is composed of the sum of 2 matrices, each made of tensor products of N 2x2 matrices.
The matrix in each term corresponding to the control qubit in each term is the outer product of |0> and |1> in the
1st and 2nd term respectively. 
The matrix corresponding to the target is the X gate, this applies to te 2nd term only.
Every other matrix is the 2x2 idenitiy. 



"""
import numpy as np


def cNOT(control, target, N):
    
    #make list of identity matrices
    initList = []
    for i in range(N):
        initList.append(np.eye(2))

    #the sum making up cNOT has 2 terms
    term1 = initList
    term2 = term1.copy()
    
    #insert outer product of |0> into 1st term
    term1[control] = np.array([1,0,0,0]).reshape(2,2)
    
    #insert outer product of |1> and X matrix into 2nd term
    term2[control], term2[target] = np.array([0,0,0,1]).reshape(2,2), np.array([0,1,1,0]).reshape(2,2)
    
    #take tensor products of the terms' components seperately to make 2 matrices
    mat1 = term1[0]
    mat2 = term2[0]
    for j in range(N-1):

        mat1 = np.array(np.kron(mat1, term1[j+1]))
        mat2 = np.array(np.kron(mat2, term2[j+1]))
        
    matFinal = mat1 + mat2
    return(matFinal)
    

cNOT(0,1,2)
