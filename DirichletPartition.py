import numpy as np

def partition(A, alpha='auto', n_clusters=2, init='random', random_state=None, 
    n_init=1, max_iter=100):
    '''For a given adjacency matrix, computes the optimal Dirichlet Parition
    by perturbing the graph laplacian and rearranging at each iteration.
    '''

    pass


class DirichletPartition(object):
    '''Dirichlet Paritioning of a Graph

    Parameters
    ----------
    n_clusters : (int)
    init : initialization used; 'random' or a matrix defining cluster indicators
    random_state :
    n_init : (int) the number of initializations to try; will output the best
    max_iter : (int) the maximum number of iterations to allow per initialization
    tol : (float) the tolerance level for the eigenvalue solver
    '''

    def fit(self):
        raise NotImplementedError

    def __init__(self, n_clusters=2, init='random', random_state=None,
        n_init=1, max_iter=100, tol=1e-6):

        self.n_clusters = n_clusters
        self.random_state = random_state
