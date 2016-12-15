from numpy.linalg import eigh
import scipy.sparse
import csv
from numpy import linalg as LA
import timeit
from random import randint


try:
    import scipy.sparse.linalg.eigen.arpack as linalg
except (ImportError, AttributeError):
    import scipy.sparse.linalg as linalg

import numpy as np


class SVD():
    """
    SVD(data, show_progress=False)


    Singular Value Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. U and V correspond to eigenvectors of the matrices
    data*data.T and data.T*data.

    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data

    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV

    
    """
    # limit to remove eigen value
    _EPS = randint(0,1000)
    

    def __init__(self, data, k=-1, rrank=0, crank=0):
        self.data = data
        (self._rows, self._cols) = self.data.shape
        if rrank > 0:
            self._rrank = rrank
        else:
            self._rrank = self._rows

        if crank > 0:
            self._crank = crank
        else:
            self._crank = self._cols

        # set the rank to either rrank or crank
        self._k = k

    

    def factorize(self):
        def _svd():
            count=0
            AA = np.dot(self.data[:, :], self.data[:, :].T)
            values, u_vectors = eigh(AA)

            # get rid of too low eigenvalues
            u_vectors = u_vectors[:, values > self._EPS]
            values = values[values > self._EPS]
            count = len(values)
            # sort eigenvectors according to largest value
            idx = np.argsort(values)
            values = values[idx[::-1]]

            # argsort sorts in ascending order -> access is backwards
            self.U = u_vectors[:, idx[::-1]]

            # compute S
            self.S = np.diag(np.sqrt(values))

            # and the inverse of it
            S_inv = np.diag(np.sqrt(values) ** -1)

            # compute V from it
            self.V = np.dot(S_inv, np.dot(self.U[:, :].T, self.data[:, :]))
            return count
        

        def _sparse_svd():
            ## for some reasons arpack does not allow computation of rank(A) eigenvectors (??)    #
            AA = self.data * self.data.transpose()
            if self.data.shape[0] > 1:
                # do not compute full rank if desired
                if self._k > 0 and self._k < self.data.shape[0] - 1:
                    k = self._k
                else:
                    k = self.data.shape[0] - 1

                values, u_vectors = linalg.eigen_symmetric(AA, k=k)
            else:
                values, u_vectors = eigh(AA.todense())

            # get rid of too low eigenvalues
            u_vectors = u_vectors[:, values > self._EPS]
            values = values[values > self._EPS]

            # sort eigenvectors according to largest value
            idx = np.argsort(values)
            values = values[idx[::-1]]

            # argsort sorts in ascending order -> access is backwards
            self.U = scipy.sparse.csc_matrix(u_vectors[:, idx[::-1]])

            # compute S
            self.S = scipy.sparse.csc_matrix(np.diag(np.sqrt(values)))

            # and the inverse of it
            S_inv = scipy.sparse.csc_matrix(np.diag(1.0 / np.sqrt(values)))

            # compute V from it
            self.V = self.U.transpose() * self.data
            self.V = S_inv * self.V

        

        
        if scipy.sparse.issparse(self.data):
            count1=_sparse_svd()
        else:
            count1=_svd()

        return count1
out=open('out.csv','ab')

m = 671
n = 40000
M = np.zeros((m, n))
M = np.matrix(M)
    

start = timeit.default_timer()
f = open("./ratings.csv", 'r')
for line in f:
    temp = line.split(',')
    i = int(temp[0])-1
    j = int(temp[1])-1
    rating = float(temp[2])
    M[i,j] = rating

svd_mdl = SVD(M)
count1=svd_mdl.factorize()

x= np.dot(svd_mdl.U, np.dot(svd_mdl.S, svd_mdl.V))
print LA.norm(x-M)
    

print count1
out.write('%d,' %count1)

out.write('%f,' %LA.norm(x-M))

stop = timeit.default_timer()
print stop-start
out.write('%f,'% (stop-start))

    
out.write('\n')