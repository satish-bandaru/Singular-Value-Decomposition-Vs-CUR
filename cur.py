from numpy.linalg import eigh
import numpy as np
import scipy.sparse
from numpy import linalg as LA
import timeit

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
    _EPS = 10**-7

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
            AA = np.dot(self.data[:, :], self.data[:, :].T)
            values, u_vectors = eigh(AA)

            # get rid of too low eigenvalues
            u_vectors = u_vectors[:, values > self._EPS]
            values = values[values > self._EPS]

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
            _sparse_svd()
        else:
            _svd()
        
            

def pinv(A, k=-1, eps=10**-8):    
    # Compute Pseudoinverse of a matrix
    # calculate SVD
    svd_mdl =  SVD(A, k=k)
    svd_mdl.factorize()
    
    S = svd_mdl.S
    Sdiag = S.diagonal()
    Sdiag = np.where(Sdiag >eps, 1.0/Sdiag, 0.0)
    
    for i in range(S.shape[0]):
        S[i,i] = Sdiag[i]
            
    if scipy.sparse.issparse(A):            
        A_p = svd_mdl.V.T * (S *  svd_mdl.U.T)
    else:    
        A_p = np.dot(svd_mdl.V.T, np.core.multiply(np.diag(S)[:,np.newaxis], svd_mdl.U.T))

    return A_p

__all__ = ["CUR"]

class CUR(SVD):
    """      
    CUR(data,  data, k=-1, rrank=0, crank=0)
        
    CUR Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. CUR randomly selects rows and columns from
    data for building U and V, respectively. 
    
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    rrank: int, optional 
        Number of rows to sample from data.
        4 (default)
    crank: int, optional
        Number of columns to sample from data.
        4 (default)
    show_progress: bool, optional
        Print some extra information
        False (default)    
    
    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV        
    
    
    """
    
    def __init__(self, data, k=-1, rrank=0, crank=0):
        SVD.__init__(self, data,k=k,rrank=rrank, crank=rrank)
        
        # select all data samples for computing the error:
        # note that this might take very long, adjust self._rset and self._cset 
        # for faster computations.
        self._rset = range(self._rows)
        self._cset = range(self._cols) 

        
    def sample(self, s, probs):        
        prob_rows = np.cumsum(probs.flatten())            
        temp_ind = np.zeros(s, np.int32)
    
        for i in range(s):            
            v = np.random.rand()
                        
            try:
                tempI = np.where(prob_rows >= v)[0]
                temp_ind[i] = tempI[0]        
            except:
                temp_ind[i] = len(prob_rows)
            
        return np.sort(temp_ind)
        
    def sample_probability(self):
        
        if scipy.sparse.issparse(self.data):
            dsquare = self.data.multiply(self.data)    
        else:
            #dsquare = self.data[:,:]**2
            dsquare = np.square(self.data[:,:])
            

            
        prow = np.array(dsquare.sum(axis=1), np.float64)
        pcol = np.array(dsquare.sum(axis=0), np.float64)
        
        prow /= prow.sum()
        pcol /= pcol.sum()    
        
        return (prow.reshape(-1,1), pcol.reshape(-1,1))
                            
    def computeUCR(self):                
        # the next  lines do NOT work with h5py if CUR is used -> double indices in self.cid or self.rid
        # can occur and are not supported by h5py. When using h5py data, always use CMD which ignores
        # reoccuring row/column selections.
        
        if scipy.sparse.issparse(self.data):
            self._C = self.data[:, self._cid] * scipy.sparse.csc_matrix(np.diag(self._ccnt**(1/2)))        
            self._R = scipy.sparse.csc_matrix(np.diag(self._rcnt**(1/2))) * self.data[self._rid,:]        

            self._U = pinv(self._C, self._k) * self.data[:,:] * pinv(self._R, self._k)
                     
        else:        
            self._C = np.dot(self.data[:, self._cid].reshape((self._rows, len(self._cid))), np.diag(self._ccnt**(1/2)))        
            self._R = np.dot(np.diag(self._rcnt**(1/2)), self.data[self._rid,:].reshape((len(self._rid), self._cols)))

            self._U = np.dot(np.dot(pinv(self._C, self._k), self.data[:,:]),pinv(self._R, self._k))
            
        # set some standard (with respect to SVD) variable names 
        self.U = self._C
        self.S = self._U
        self.V = self._R

        #for item in self.S:
            #print item

            
    def factorize(self):
        """ Factorize s.t. CUR = data
            
            Updated Values
            --------------
            .C : updated values for C.
            .U : updated values for U.
            .R : updated values for R.           
        """          
        [prow, pcol] = self.sample_probability()
        self._rid = self.sample(self._rrank, prow)
        self._cid = self.sample(self._crank, pcol)
        
        self._rcnt = np.ones(len(self._rid))
        self._ccnt = np.ones(len(self._cid))    
                                    
        self.computeUCR()

m = 671
n = 40000
M = np.zeros((m, n))
M = np.matrix(M)
f = open("./ratings.csv", 'r')
for line in f:
    temp = line.split(',')
    i = int(temp[0])-1
    j = int(temp[1])-1
    rating = float(temp[2])
    M[i,j] = rating

#np.random.seed(421997)

out=open('out1.csv','ab')
start = timeit.default_timer()
roww = np.random.choice(range(m))
print roww

cur_mdl=CUR(M,k=2,rrank=roww,crank=np.random.choice(range(n)))
cur_mdl.factorize()
x= np.dot(cur_mdl.U, np.dot(cur_mdl.S, cur_mdl.V))
print LA.norm(x-M)
out.write('%d,' %roww)
out.write('%f,' %LA.norm(x-M))
stop = timeit.default_timer()
print stop-start
out.write('%f,'% (stop-start))

    
out.write('\n')