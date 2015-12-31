from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def backwards(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the gradient fo the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor

    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)

    # You need to return the following variables correctly
    Theta_grad = [zeros(w.shape) for w in Theta]

    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
    for i in range(0,m):
        yv[ y[i] ][i] = 1

    # ================================ TODO ================================
    # In this point implement the backpropagation algorithm

    print "layers", layers
    print "Theta", [x.shape for x in Theta]
    #print Theta

    for t in range(0,m):
        print "t = ",t
        a = [X[t]]
        z = []

        for i in range(0,num_layers - 1):
            #a = insert(a,0,values=1,axis=1)
            #a.insert(0,1)
            b = copy(a[-1])
            b = insert(b,0,values=1,axis=0)
            z.append(b.dot(transpose(Theta[i])))
            a.append(sigmoid(z[-1]))
        #print "shape a", [x.shape for x in a]
        #print "shape z", [x.shape for x in z]
        delta3 = a[2] - yv[:,t]
        #print Theta[1].shape, delta3.shape, z[0].shape
        delta2 = transpose(Theta[1][:,1:]).dot(delta3) * sigmoidGradient(z[0])
        #print "shape delta", delta2.shape, delta3.shape
        #print delta3.shape,a[2].shape,delta2.shape,a[1].shape
        #print transpose(a[2]).shape
        a[0] = insert(a[0],1,values=1,axis=0)
        a[1] = insert(a[1],1,values=1,axis=0)

        nabla2 = delta3[:,None].dot(transpose(a[1][:,None]))
        #nabla2 = insert(nabla2,0,values=0,axis=1)
        nabla1 = delta2[:,None].dot(transpose(a[0][:,None]))
        #nabla1 = insert(nabla1,0,values=0,axis=1)
        print Theta_grad[1].shape, nabla2.shape, Theta_grad[0].shape, nabla1.shape

        Theta_grad[1] += nabla2 / m
        Theta_grad[0] += nabla1 / m

    # Unroll Params
    print Theta_grad[1]
    print Theta_grad[0]
    Theta_grad = unroll_params(Theta_grad)

    return Theta_grad
