from numpy import *

def sigmoid(z):
    # SIGMOID returns sigmoid function evaluated at z
    # g = zeros(shape(z))
    # Instructions: Compute sigmoid function evaluated at each value of z.
    g = 1.0 / (1 + exp(-z))
    return g
