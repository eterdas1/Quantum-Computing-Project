# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:24:58 2023

@author: logan
"""
import numpy as np
def Hadamard_Gate(qubitCoeffs):
    #Function takes in an array containing the coefficients associated with a qubit

    
    #Make numpy array of the form of the Hadamard gate
    H = (1/np.sqrt(2)) * np.array([1,1,1,-1]).reshape(2,2)

    #multiply the input array with the Hadamard gate using matmul function
    prod = np.matmul(H, qubitCoeffs)

    return prod
    



