import numpy as np 

def eigen(matrix):
    '''
    Returns the eigenvalues and associated eigenvectors, in descending order 
    of eigenvector, assuming the matrix is real
    '''

    np_eigval, np_eigvector = np.linalg.eig(matrix) # get the eigenstuff with numpy

    eig = []
    for i in range(len(np_eigval)):
        # append eigenvalue and corresponding eigenvector to list
        eig.append((np_eigval[i], np_eigvector[:,i].tolist()))

    # sort in descending order by eigenvalue
    sorted(eig, key=lambda eigen: eigen[0])
    
    return eig