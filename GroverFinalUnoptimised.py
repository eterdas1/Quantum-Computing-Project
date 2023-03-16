"""
The code below contains three classes.

The 1st one holds the gates to be used in the circuit and our implementations
of their operations such as tensor product, this class takes in the desired number of qubits.

The 2nd class contains the code necessary to perform Grover's Algorithm though it uses dense matrices so is not
as fast as it could be. It runs for a maximum of 13 qubits, beyond this the operations require too much memory.
This class takes in the desired number of qubits, the marked target state, and the class from which the gates
will be taken from 

The 3rd class is a test class that executes the rest of the code.

In this version the action of the oracle is hard coded by multiplying the element corresponding to the target 
in the state vector by -1.
The diffuser is made using a combination of H, X and multi-controlled Z
gates. However the MC^n Z gate had to be hard coded as it is too complex to make using elementary gates.
"""

import numpy as np

"""Class that holds gate matrices and their operations"""
class Gate_Class(object):
    
    #Gate class takes in number of qubits in the circuit
    def __init__(self, N):
        self.N = N

    
    #The gates can perform the tensor product, our implementation is defined below
    def tensor_product(self, x,y):

        rows = [] #list where each element is a row in the final array
    
        for i in range(x.shape[0]):
            row_value = []
            for j in range(x.shape[1]):
    
                row_value.append(x[i,j] * y)
    
                if j == x.shape[1] - 1:
                    rows.append(np.hstack(row_value)) #combines the row values into a single array and then appends it to the row list
    
        tp = np.vstack(rows) #vertical stacks all the rows
    
        return tp
    
    
    #the methods below make the forms of the gates and return them as numpy arrays
        
    #Hadamard Gate
    def Hdmd(self): 
        
        H = 1/np.sqrt(2) * np.array([1, 1, 1, -1]).reshape(2,2)
        return H
    
    #Pauli X Gate
    def X(self):
        X = np.array([0,1,1,0]).reshape(2,2)
        return X
    
    #Multi-Controlled Z Gate
    def MCZ(self):
        MCZ = np.eye(2**self.N)
        MCZ[-1][-1] = -1
        
        return MCZ
    
    #Identity Matrix
    def I(self):
        I = np.eye(self.N)
        return I


"""


grover starts here



"""
class Grovers_Circuit(object):

    #init function takes in number of qubits, target state, and the class that gates will be pulled from.
    def __init__(self, N, target, gateSource):
        
        assert gateSource.N == N, "N given to gate class does not match Grover N"
        assert target <= 2**N - 1,"Target state outwith range"
        self.N = N
        self.target = target
        self.gateSource = gateSource
        
        #Calculates the number of iterations necessary
        self.num_iterations = int(np.pi/4 * np.sqrt(2**N)) 

        #initialise state as every qubit being 0
        self.state = np.zeros((2**self.N))
        self.state[0] = 1


    #method to put state in initial superposition using hadamards
    def superposition(self):

        #take tensor product of N hadamard gates
        H = self.gateSource.Hdmd()
        HProduct = H
        for i in range(self.N-1):
            HProduct = self.gateSource.tensor_product(HProduct, H)

        #Apply the matrix to the state
        self.state = np.matmul(self.state, HProduct)
        
        
    #method that makes the diffuser
    def diffuser(self):

        #pull necessary gates from gate class
        H = self.gateSource.Hdmd()
        X = self.gateSource.X()
        MCZ = self.gateSource.MCZ()
        
        #make tensor products of X gates together
        xProduct = X
        for i in range(self.N-1):
            xProduct = self.gateSource.tensor_product(xProduct, X)
        
        #make U_0
        U_0 = np.matmul(np.matmul(-1 * xProduct, MCZ), xProduct)
        
        #make tensor products of hadamards together
        HProduct = H
        for i in range(self.N-1):
            HProduct = self.gateSource.tensor_product(HProduct, H)
            
        #make the diffuser, U_s
        U_s = np.matmul(np.matmul(-1 * HProduct, U_0), HProduct)
        return U_s

    #Oracle function flips the sign of the element we are looking for
    def oracle(self): 
        self.state[self.target] *= -1
        

    #method that runs Grover's Algorithm
    def circuit(self):
    
        #put initial state into superposition
        self.superposition()
        
        #create diffuser using diffuser method
        diff = self.diffuser()
        
        #perform grover's algorithm for the required number of iterations. (Apply oracle then diffuser)
        for i in range(self.num_iterations):
            
            self.oracle()
            
            self.state = np.matmul(diff, self.state) 
        
        #calculate probabilities of each state
        probabilities = self.state*self.state
        print(probabilities) 
        print("Most probable state is: ", np.argmax(probabilities), "with probability: ", np.max(probabilities))


"""Class to run the above code"""
class Grover_Test(object):
    def run(self, num_qubits, target, gateSource):

        g = Grovers_Circuit(num_qubits, target, gateSource)
        g.circuit()


n = 9
test = Grover_Test().run(n, 4, Gate_Class(n))




