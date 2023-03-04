import numpy as np

# State needs to be tensor product before we can use matrix product
def tensor_product(vectors):
    result = np.array([1])
    for v in vectors:
        result = np.kron(result, v)
    return result
   
#convert a given number of state, with kth target. Make N vectors accordingle
def write_vectors(n,k):
    vectors =np.array([np.array([0,1]) for i in range(n)])
    vectors[k] = np.array([1,0])
    return vectors
   
   
 #This will make oracle matrix, given n state and kth target
def make_oracle(n, k,state):
    I = np.identity(2**n)
    v = np.zeros(2**n)
    v[np.argmax(state)] = 1
    return -2 * np.outer(v, v) + I
   
   
   
def grover(N, target):

    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    
    V = write_vectors(N,target)
    state = tensor_product(V)
    M = np.argmax(state)
    
    H_tensor = tensor_product([H]*N)   
    oracle_tensor=make_oracle(N, target,state)
    
    state = np.dot(H_tensor, state)
    
    i = 0
    while np.max(np.abs(state*state))<0.95:  
        state_min_1 =state
        state = np.dot(oracle_tensor,state)
        state = state - 2 * np.dot(state, state_min_1) / np.dot(state_min_1, state_min_1) * state_min_1 
        i += 1
    result = np.sqrt(np.sum(np.square(state)))
    return i,M,np.argmax(np.abs(state)),max(state*state),result
   
   
   grover(7,5)
