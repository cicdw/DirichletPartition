import numpy as np
import scipy.linalg as la

class ClusteredGraph(object):
    '''Clustered Graph class.

    Parameters
    ----------
    A : (array-like) square adjacency matrix / similarity matrix
    labs : (array-like, optional) integer-valued cluster assignments
    '''

    def __init__(self, A, labs=None):
        self.A = A
        self.labs = labs
        self.L = np.diag(A.sum(axis=1)) - A # graph laplacian

class DirichletPartition(object):
    '''Dirichlet Paritioning of a Graph

    Parameters
    ----------
    n_clusters : (int) number of clusters to partition into
    init : initialization used; 'random' or a matrix defining cluster indicators
    random_state :
    n_init : (int) the number of initializations to try; will output the best
    max_iter : (int) the maximum number of iterations to allow per initialization
    tol : (float) the tolerance level for the eigenvalue solver
    '''

    def _rearrange(self, L, init, alpha=0.01):
        '''Performs one rearrangement step for a given set of inputs:

        L : (array-like) Graph Laplacian
        init : (array-like) vector of zero-indexed cluster assignments for each data point
        alpha : (float) penalization on diagonal perturbation

        Returns:
        Array-like vector of new zero-indexed cluster assignments, along with the objective value.
        '''

        out = np.zeros((L.shape[0], self.n_clusters)) #for storing the new labels
        for k in range(self.n_clusters):
            S = L + alpha*np.diag(init!=k)
            vals, eigs = la.eigh(S, eigvals=(0,0))
            if eigs[0,0]<0:
                out[:,k] = -1 * eigs[:,0]
            else:
                out[:,k] = eigs[:,0]
        return out.argmax(axis=1)

    def _check_is_connected(self, L):
        '''Checks whether the graph is close to being disconnected via
        the second eigenvalue
        '''
        vals, _ = la.eigh(L, eigvals=(0,1))
        return np.isclose(vals[1], 0)

    def _fit_broadcast(self):
        raise NotImplementedError

    def _init(self, size):
        '''Produces initialization labels.'''
        c_size = size/self.n_clusters                
        labs = [k for k in range(self.n_clusters) for _ in range(int(c_size))]
        for _ in range(size % self.n_clusters):
            labs.append(0)
        if self.init=='random':
            return np.random.choice(labs, size=size, replace=False)
        else:
            return labs

    def fit(self, A, method='rearrange', **kwargs):
        L = np.diag(A.sum(axis=1)) - A # laplacian
        labs = self._init(L.shape[0])
        converged, iters = False, 0
        while (not converged) and (iters<self.max_iter):
            old, labs = labs, self._rearrange(L, labs)
            converged = np.all(old==labs)
            iters += 1
        
        return labs

    def __init__(self, n_clusters=2, init='random', random_state=None,
        n_init=1, max_iter=100, tol=1e-6):

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
