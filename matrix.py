import numpy as np
import scipy.linalg

def eigvals(matrix):
    return np.linalg.eig(matrix)[0]

def eigvec(matrix):
    return np.linalg.eig(matrix)[1]

def eigh(matrix):
    '''
    Leverage Hermitian (real symmetric) matrices and use numpy's
    eigh function instead
    '''
    return np.linalg.eigh(matrix)

def eigvalsh(matrix):
    '''
    Return just the eigenvalues in descending order
    '''
    matrix = matrix.asfptype()
    return scipy.sparse.linalg.eigsh(matrix)[0]

def eigvech(matrix):
    '''
    Return a matrix of eigenvectors
    '''
    matrix = matrix.asfptype()
    return scipy.sparse.linalg.eigsh(matrix)[1]

def log_matrix_average(L):
    '''
    Calculate the log average of a list of matrices
    '''
    mat_sum = np.matrix(scipy.linalg.logm(L[0]))
    for mat in L[1:]:
        mat_sum = np.add(mat_sum, np.matrix(scipy.linalg.logm(mat)))
    mat_sum /= len(L)

    return np.matrix(scipy.linalg.expm(mat_sum))

def matrix_sum(L):
    '''
    Calculate the sum of a list of matrices
    '''
    mat_sum = L[0]
    for mat in L[1:]:
        mat_sum = np.add(mat_sum, mat)
    return mat_sum

def matrix_average(L):
    '''
    Calculate the mean of a list of matrices
    '''
    return 1.0 / len(L) * matrix_sum(L)

def polar_decomp_average(L):
    # extract rotation matrix by polar decomposition
    sum = matrix_sum(L)
    return scipy.linalg.polar(sum)[0]