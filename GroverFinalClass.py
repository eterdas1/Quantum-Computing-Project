"""
The code below contains four classes.

The 1st one holds all the methods used for sparse matrices including the converter from dense matrices to Sparse

The 2nd one holds the gates to be used in the circuit and our implementations
of their operations such as tensor product, this class takes in the desired number of qubits.

The 3rd class contains the code necessary to perform Grover's Algorithm though it uses dense matrices so is not
as fast as it could be. It runs for a maximum of 13 qubits, beyond this the operations require too much memory.
This class takes in the desired number of qubits, the marked target state, and the class from which the gates
will be taken from 

The 4th class is a test class that executes the rest of the code.

In this version the action of the oracle is hard coded by multiplying the element corresponding to the target 
in the state vector by -1.
The diffuser is made using a combination of H, X and multi-controlled Z
gates. However the MC^n Z gate had to be hard coded as it is too complex to make using elementary gates.
"""

import numpy as np
import time
import matplotlib.pyplot as plt


class Sparse_Class:
    "Class that holds all the sparse methods"


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
            for j in range(dim[1]):
                
                row_ind = np.where(a_rows == i)[0]       
                col_ind = np.where(b_cols == j)[0]
                row_vals = a_vals[row_ind] # values on the row we're currently multiplying
                col_vals = b_vals[col_ind] # values on the column we're currently multiplying

                if row_vals.shape[0] != 0 and col_vals.shape[0] != 0:

                    
            

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

    

            
        return np.array((np.concatenate(new_vals), np.concatenate(new_rows), np.concatenate(new_cols), np.array(dim)), dtype=object)
                    
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
    def __init__(self, N, target, gateSource, sparseSource):
        
        assert gateSource.N == N, "N given to gate class does not match Grover N"
        assert target <= 2**N - 1,"Target state outwith range"
        self.N = N
        self.target = target
        self.gateSource = gateSource
        self.sparseSource = sparseSource
        
        #Calculates the number of iterations necessary
        self.num_iterations = int(np.pi/4 * np.sqrt(2**N)) 

        #initialise state as every qubit being 0
        self.state = np.zeros((2**self.N))
        self.state[0] = 1


    #method to put state in initial superposition using hadamards
    def superposition(self):

        #take tensor product of N hadamard gates
        H = self.gateSource.Hdmd()
        self.HProduct = H
        for i in range(self.N-1):
            self.HProduct = self.gateSource.tensor_product(self.HProduct, H)

        #Apply the matrix to the state
        self.state = np.matmul(self.state, self.HProduct)
        
        
    #method that makes the diffuser
    def diffuser(self):

        #pull necessary gates from gate class
        X = self.gateSource.X()
        MCZ = self.gateSource.MCZ()
        
        #make tensor products of X gates together
        xProduct = X
        for i in range(self.N-1):
            xProduct = self.gateSource.tensor_product(xProduct, X)
        
        #make U_0
        U_0 = np.matmul(np.matmul(-1 * xProduct, MCZ), xProduct)
        
        #make the diffuser, U_s
        U_s = np.matmul(np.matmul(-1 * self.HProduct, U_0), self.HProduct)
        return U_s

    #method that makes the diffuser with sparse methods
    def sparse_diffuser(self):

        #pull necessary gates from gate class
        X = self.gateSource.X()
        MCZ = self.gateSource.MCZ()

        #Turn Sparse gates into Sparse form
        X = self.sparseSource.Matrix_to_Sparse(X)
        MCZ = self.sparseSource.Matrix_to_Sparse(MCZ)
        
        #make tensor products of X gates together
        xProduct = X
        for i in range(self.N-1):
            xProduct = self.sparseSource.Sparse_Tensor_Product(xProduct, X)
        
        #make U_0
        minus_xProduct = np.copy(xProduct)
        minus_xProduct[0] = minus_xProduct[0] *-1
        U_0 = self.sparseSource.Sparse_X_Multiply(self.sparseSource.Sparse_X_Multiply(minus_xProduct, MCZ, 0), xProduct, 2)
        
        #Convert U_0 back into dense form
        U_0 = self.sparseSource.Sparse_to_matrix(U_0)
                    
        #make the diffuser, U_s
        U_s = np.matmul(np.matmul(-1 * self.HProduct, U_0), self.HProduct)
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
        plt.bar(np.arange(2**self.N), probabilities)
        plt.ylabel('Probability of Measuring the State')
        plt.xlabel('The State')
        plt.show()

    #method that runs Grover's Algorithm with sparse
    def sparse_circuit(self):
    
        #put initial state into superposition
        self.superposition()
        
        #create diffuser using diffuser method
        diff = self.sparse_diffuser()
        
        #perform grover's algorithm for the required number of iterations. (Apply oracle then diffuser)
        for i in range(self.num_iterations):
            
            self.oracle()
            
            self.state = np.matmul(diff, self.state) 
        
        #calculate probabilities of each state
        probabilities = self.state*self.state
        print(probabilities) 
        print("Most probable state is: ", np.argmax(probabilities), "with probability: ", np.max(probabilities))
        plt.bar(np.arange(2**self.N), probabilities)
        plt.ylabel('Probability of Measuring the State')
        plt.xlabel('The State')
        plt.show()

    #method that runs Grover's Algorithm with sparse and graphs the vector
    def graph_circuit(self):

        #lists that store the values for plotting
       
        y_vals = []
        
    
        #put initial state into superposition
        self.superposition()

        #y_vals.append(self.state[self.target])
        #x_vals.append((1 - (self.state[self.target])**2)**0.5)

        
        #create diffuser using diffuser method
        diff = self.sparse_diffuser()
        
        #perform grover's algorithm for the required number of iterations. (Apply oracle then diffuser)
        for i in range(self.num_iterations):
            print(self.state)
            
            self.oracle()

            
            
            
            self.state = np.matmul(diff, self.state) 

            y_vals.append((self.state[self.target])**2)
           
            
        
        #calculate probabilities of each state
        probabilities = self.state*self.state
        print(probabilities) 
        print("Most probable state is: ", np.argmax(probabilities), "with probability: ", np.max(probabilities))
        plt.scatter( np.arange(self.num_iterations),y_vals)
        plt.ylabel('Probability of Measuring the Target State')
        plt.xlabel('Number of Iterations')
        plt.show()
          
    

"""Class to run the above code"""
class Grover_Test(object):
    def run(self, num_qubits, target, gateSource, sparseSource):

        g = Grovers_Circuit(num_qubits, target, gateSource, sparseSource)
        t1 = time.time()
        g.circuit()
        t2 = time.time()
        

        print('Time taken without Sparse: ',t2 - t1)


    def run_sparse(self, num_qubits, target, gateSource, sparseSource):

        g = Grovers_Circuit(num_qubits, target, gateSource, sparseSource)
        t1 = time.time()
        g.sparse_circuit()
        t2 = time.time()
        

        print('Time taken with Sparse: ',t2 - t1)    


    def run_graph(self, num_qubits, target, gateSource, sparseSource):

        g = Grovers_Circuit(num_qubits, target, gateSource, sparseSource)
        t1 = time.time()
        g.graph_circuit()
        t2 = time.time()
        

        print('Time taken with Sparse: ',t2 - t1)      


n = 10
test = Grover_Test().run(n, 4, Gate_Class(n), Sparse_Class)
#test = Grover_Test().run_graph(n, 4, Gate_Class(n), Sparse_Class)

