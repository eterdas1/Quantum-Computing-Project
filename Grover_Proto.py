import numpy as np


def Hadmard_product(A, n): # Tensor products between Hadmards
    result = np.array([[1]])
    for i in range(n):
        result = np.kron(result, A)
    return result

A = (1 / np.sqrt(2)) *np.array([[1, 1], [1, -1]]) # Defines the Hadamard
num_qubits = 3
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


for i in range(num_iterations):
    for index in range(2**num_qubits): # Number of outputs
        if index == target_in: # Checks the index is equal to what we're searching for and then runs the oracle on that index
            state = oracle(state, index)
    state = np.dot(diff, state) # Runs the diffuser on the state


print(state*state) # Prints the probability of each state
