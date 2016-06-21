import numpy as np
from DirichletPartition import *

def test_one_step_2block():
    '''Two blocks, good initilization.'''
    C = DirichletPartition()
    A = np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]])
    L = np.diag(A.sum(axis=1)) - A
    res = C._rearrange(L, np.array([0,0,1,1]), alpha=0.01)
    assert np.all(res == np.array([0,0,1,1]))

def test_one_step_3block():
    '''Three blocks, good initilization.'''
    C = DirichletPartition(n_clusters=3)
    A = np.array([[0,1,0,0,0], [1,0,0,0,0], [0,0,0,1,0], [0,0,1,0,0],
        [0,0,0,0,1]])
    L = np.diag(A.sum(axis=1)) - A
    res = C._rearrange(L, np.array([0,0,1,1,2]), alpha=0.01)
    assert np.all(res == np.array([0,0,1,1,2]))

def test_two_step_2block():
    '''Two blocks, bad initilization.'''
    C = DirichletPartition()
    A = np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]])
    L = np.diag(A.sum(axis=1)) - A
    res = C._rearrange(L, np.array([0,1,1,1]), alpha=0.01)
    assert np.all(res == np.array([0,0,1,1]))

def test_two_step_3block():
    '''Three blocks, bad initilization.'''
    C = DirichletPartition(n_clusters=3)
    A = np.array([[0,1,0,0,0], [1,0,0,0,0], [0,0,0,1,0], [0,0,1,0,0],
        [0,0,0,0,1]])
    L = np.diag(A.sum(axis=1)) - A
    res = C._rearrange(L, np.array([0,1,1,1,2]), alpha=0.01)
    assert np.all(res == np.array([0,0,1,1,2]))

