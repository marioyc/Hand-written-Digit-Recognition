from numpy import *
from sigmoid import sigmoid

def predict(Theta, X):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels

    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    # You need to return the following variables correctly
    p = zeros((1,m))

    for t in range(0,m):
        a = X[t]
        print a

        for i in range(0,num_layers - 1):
            a = insert(a,0,values=1,axis=0)
            print Theta[i]
            z = a.dot(transpose(Theta[i]))
            a = sigmoid(z)

        p[0][t] = argmax(a)

    return p
