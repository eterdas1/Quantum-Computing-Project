import numpy as np
from Tensor_Product.py import tensor_product
from Gate_Class.py import Phase_Gate
from Gate_Class.py import cnot_Gate
def Hadmard_product(A, n): # Tensor products between Hadmard gates
    result = np.array([[1]])
    for i in range(n):
        result = np.kron(result, A)
    return result

A = (1 / np.sqrt(2)) *np.array([[1, 1], [1, -1]]) # Defines the Hadamard
num_qubits = 4
num_iterations = int(np.pi/4 * np.sqrt(2**num_qubits)) # Calculates the number of iterations
target_in = 1  # Index grover is searching for

state = np.zeros((2**num_qubits)) # Creates the inital state
state[0] = 1


result = Hadmard_product(A, num_qubits) # Does Tensor product between 3 Hadmards
state = np.matmul(state,result) # Applies Hadmard to all qubits in our state



def oracle(state, target): # Oracle function flips the sign of the element we're looking for
    state = np.copy(state)
    state[target] *= -1
    return state


diff = np.full((2**num_qubits, 2**num_qubits), 2/(2**num_qubits)) - np.eye(2**num_qubits) # Creates the diffuser matrix
print(diff)

def diffuser(state):
    state = tensor_product(A,state) #Inital Hadamard gate

    state = tensor_product(Phase_Gate(state,np.pi),state) #Z-gate

    state = tensor_product(A,state) #Controlled-Z gate (CNOT gate with H either side)
    #state = tensor_product(cnot_Gate(),state)
    state = tensor_product(A,state)

    state = tensor_product(A,state) #Final Hadamard

    return state

for i in range(num_iterations):
    for index in range(2**num_qubits): # Number of outputs
        if index == target_in: # Checks the index is equal to what we're searching for and then runs the oracle on that index
            state = oracle(state, index)
    state = np.dot(diff, state) # Runs the diffuser on the state


print(state*state) # Prints the probability of each state
