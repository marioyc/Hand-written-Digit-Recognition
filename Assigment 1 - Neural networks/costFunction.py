from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor

    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)

    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
    for i in range(0,m):
        yv[ y[i] ][i] = 1

    # calculate the cost of the neural network (feedforward)
    a = X

    for i in range(0,num_layers - 1):
        a = insert(a,0,values=1,axis=1)
        z = a.dot(transpose(Theta[i]))
        a = sigmoid(z)

    yv = transpose(yv)
    J = (-yv * log(a) - (1 - yv) * log(1 - a)) / m;
    J = sum(J)

    # regularization term
    regularization = 0
    for i in range(0,num_layers - 1):
        regularization += sum(Theta[i][:,1:]**2)
    regularization *= lambd / (2 * m)

    J += regularization

    return J
