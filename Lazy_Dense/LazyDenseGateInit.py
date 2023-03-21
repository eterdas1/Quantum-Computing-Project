from abc import ABC, abstractmethod #used to implement the functionality of Interfaces
import itertools #useful for manipulating/creating iterables

#defines a MatrixElement object i.e. what goes in each entry of a matrix
class MatrixElement:
    def __init__(self, row: int, col: int, val: complex) -> None:
        self.row = row
        self.col = col
        self.val = val

#Interface defining Square Matrices
class ISquareMatrix(ABC):

    #abstract attributes cannot be declared in constructors
    #so the following decorators are used.
    #dimension implemented as an abstract property since
    #objects derived from ISquareMatrix don't necessarily have
    #the same dimension.
    @property
    @abstractmethod
    def dim(self):
        pass

    #dunder which implements square bracket indexing
    @abstractmethod
    def __getitem__(self, index):
        pass

    #when implemented, should make any object of this type
    #an iterable of MatrixElement objects
    @abstractmethod
    def __iter__(self) -> MatrixElement:
        pass
    
    #overloading dunder to allow for useful print statements
    def __str__(self):
        matrix_str = ""
        for i in self:
            matrix_str += "Matrix Element [{}, {}] has value {}\n".format(i.row, i.col, i.val)
        return matrix_str
    
    #following are yet to be implemented
    #instead of commenting out I could have labeled as abstract 
    #and used pass however this would require me to implement
    #the methods with 'pass' in all subclasses

    #matmul operation
    # def multiply(self, other):
    #     pass

    #overloads '@' symbol for matrix multiplication
    # def __matmul__(self, other):
    #     return self.multiply(other)

    # def tensorProduct():
    #     pass

#Interface defining Quantum gates
class IGate(ISquareMatrix):

    #abstract methods and properties are inherited so
    #code not repeated

    #implements super's abstract method for indexing
    def __getitem__(self, indices):
        row, col = indices

        #checks if indices are valid
        if not (0 <= row < self.dim and 0 <= col < self.dim): 
            raise IndexError("Matrix index out of range")
    
        # retrieve the element at the specified row and column
        element = next(itertools.islice(self, row * self.dim + col, None))
        return element.val

#Interface defining Single Qubit Quantum gates
class ISingleQubitGate(IGate):
    #by definition these have dimension=2
    @property
    def dim(self):
        return 2

#Interface defining Two Qubit Quantum gates
class TwoQubitGate(IGate):
    #by definition these have dimension=4
    @property
    def dim(self):
        return 4
    
#Interface defining Quantum gates with an arbitrary number of Qubits
class INQubitGate(IGate):
    def __init__(self, NumQbits):
        self.NumQbits = NumQbits

#Class defining Hadamard Gate Object
class Hadamard(ISingleQubitGate):

    #generates its MatrixElements as per the gate's definition
    def __iter__(self):
        yield MatrixElement(0,0,1)
        yield MatrixElement(0,1,1)
        yield MatrixElement(1,0,1)
        yield MatrixElement(1,1,-1)

#Class defining Pauli X Gate Object
class X(ISingleQubitGate):  

    #generates its MatrixElements as per the gate's definition 
    def __iter__(self):
        yield MatrixElement(0,0,0)
        yield MatrixElement(0,1,1)
        yield MatrixElement(1,0,0)
        yield MatrixElement(1,1,1)

#Class defining Multo-Controlled Pauli Z Gate Object
class MCZ(INQubitGate):
    @property
    def dim(self):
        return 2**self.NumQbits

    #generates its MatrixElements as per the gate's definition
    def __iter__(self):
        for r in range(self.dim):
            for c in range(self.dim):
                if r == c:
                    if r == self.dim - 1:
                        yield MatrixElement(r,c,-1)
                    else: 
                        yield MatrixElement(r,c,1)
                else:
                    yield MatrixElement(r,c,0)

#Class defining Indentity Gate Object
class I(INQubitGate):
    @property
    def dim(self):
        return self.NumQbits

    #generates its MatrixElements as per the gate's definition
    def __iter__(self):
        for r in range(self.dim):
                for c in range(self.dim):
                    if r == c:
                        yield MatrixElement(r,c,1)
                    else:
                        yield MatrixElement(r,c,0)

#code to show functionality of code
#Unit tests could have been used instead
def test(gate_class):
    if gate_class == Hadamard:
        gate = gate_class()
        print(gate[0,0])
        # print(gate[-1,0])
        print(gate)
    else:
        gate = gate_class(2)
        print(gate[0,0])
        # print(gate[-1,0])
        print(gate)

test(Hadamard)
test(MCZ)
test(I)


