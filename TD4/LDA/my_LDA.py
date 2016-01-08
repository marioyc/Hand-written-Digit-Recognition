import numpy as np
import scipy as sp
import scipy.linalg as linalg

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """

    classLabels = np.unique(Y) # different class labels on the dataset
    classNum = len(classLabels)
    datanum, dim = X.shape # dimensions of the dataset
    totalMean = np.mean(X,0) # total mean of the data

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudocode on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.


    m = np.zeros((classNum, dim))
    total = [0] * classNum

    for i in range(0,datanum):
        #print Y[i], classNum
        y = int(Y[i]) - 1
        total[y] += 1
        m[y] += X[i]

    for i in range(0,classNum):
        m[i] /= total[i]

    Sw = np.zeros((dim,dim))

    for i in range(0,datanum):
        y = int(Y[i]) - 1
        #print y, X[i] - m[y]
        e = X[i] - m[y]
        e = e[:,None]
        Sw += e.dot(np.transpose(e))#np.transpose(e).dot(e)
        #print e

    #print X[i].shape
    #print X[i],m[0]
    #print Sw

    Sb = np.zeros((dim,dim))

    for i in range(0,classNum):
        e = m[i] - totalMean
        e = e[:,None]
        Sb += total[i] * e.dot(np.transpose(e))#np.transpose(e).dot(e)

    #print Sb

    eigval, eigvec = linalg.eig(np.linalg.inv(Sw).dot(Sb))
    #print eigval
    #print eigvec

    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]

    #print eigval

    #W = np.concatenate(eigvec[:classNum - 1])
    #print eigvec[0]
    W = np.zeros((dim, classNum - 1))

    for i in range(0,classNum - 1):
        W[:,i] = eigvec[i]
    #print W
    #print classNum

    X_lda = X.dot(W)
    projected_centroid = m.dot(W)

    # =============================================================

    return W, projected_centroid, X_lda
