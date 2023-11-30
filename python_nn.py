import numpy as np

#activation function-sigmoid
def nonlinear(x,deriv=False):
    if deriv is True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

# training data
input_data = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])

# output data
output_data= np.array([[0,0,1,1]]).T

np.random.seed(1)

# initializing weight
synapse = 2*np.random.random((3,1)) - 1

for i in range(100000):
    # input layer
    layer0 = input_data

    # output layer
    outputLayer = nonlinear(np.dot(layer0, synapse))

    #difference of output to expected output
    l1_error = output_data - outputLayer

    #delta
    l1_delta = l1_error * nonlinear(outputLayer,True)

    # update the weight
    synapse += np.dot(layer0.T, l1_delta)

print(outputLayer, " training results")





























# Notes:

# np.dot() <- finds the product of two arrays
#   Example: a = [[1, 0], [0, 1]]
#            b = [[4, 1], [2, 2]]
#            np.dot(a, b)
#            array([[4, 1],
#                   [2, 2]])

# .T from numpy (also .transpose())
# view of the transponsed array: returns an array iwth axes transposed
# In linear algebra, the transpose of a matrix is an operator which
# flips a matrix over its diagonal; that is, it switches the row and
# column indices of the matrix A by producing another matrix,
# often denoted by Aᵀ.

# matrix = np.array([[1,2],[3,4],[5,6]])
# transposed = np.transpose(matrix)
# print(transposed, "transposed matrix")

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def propagate(self, input):
        print("placeholder")

    def back_propoagate(self, output_error, learning_rate):
        print("placeholder")