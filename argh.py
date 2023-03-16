"""I have edited this compared to the github version by adding the diffuser function that makes the diffuser from scratch  """

import numpy as np
import time



def Hadmard_product(A, n): # Tensor products between Hadmards
    result = np.array([[1]])
    for i in range(n):
        result = np.kron(result, A)
    return result

t1 = time.time()

A = (1 / np.sqrt(2)) *np.array([[1, 1], [1, -1]]) # Defines the Hadamard
num_qubits = 12
num_iterations = int(np.pi/4 * np.sqrt(2**num_qubits)) # Calculates the number of iterations
target_in = 4  # Index grover is searching for


state = np.zeros((2**num_qubits)) # Creates the inital state
state[0] = 1


result = Hadmard_product(A, num_qubits) # Does Tensor product between 3 Hadmards
state = np.matmul(state,result) # Applies Hadmard to all qubits in our state



def diffuser(n):
        #make base gates
    MCZ = np.eye(2**n)
    MCZ[-1][-1] = -1
    X = np.array([0,1,1,0]).reshape(2,2)
    H = np.array([1,1,1,-1]).reshape(2,2)
    
    #make tensor products of X gates together
    
    xProduct = X
    for i in range(n-1):
        xProduct = np.kron(xProduct,X)
    
    #make U naught
    U_0 = np.matmul(np.matmul(-1 * xProduct, MCZ), xProduct)
 

    #make tensor products of hadamards together
    
    HProduct = 1/np.sqrt(2) * H
    for i in range(n-1):
        HProduct = 1/(2**0.5) * np.kron(HProduct,H)
        
    
    
    U_s = np.matmul(np.matmul(-1 * HProduct, U_0), HProduct)
    return(U_s)


def oracle(state, target): # Oracle function flips the sign of the element we're looking for
    state = np.copy(state)
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

print(t2 - t1)