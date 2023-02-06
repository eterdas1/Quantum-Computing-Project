
import numpy as np
def Hadamard_Gate(qubitCoeffs):
    #Function takes in an array containing the coefficients associated with a qubit

    
    #Make numpy array of the form of the Hadamard gate
    H = (1/np.sqrt(2)) * np.array([1,1,1,-1]).reshape(2,2)

    #multiply the input array with the Hadamard gate using matmul function
    prod = np.matmul(H, qubitCoeffs)

    return prod
    
def Phase_Gate(qubitCoeffs):
    phase = np.array([[1,0],[0, 0+1j]])

    return np.matmul(phase,qubitCoeffs)



