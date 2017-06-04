import numpy as np 
import csv

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
    return sorted(eig, key=lambda elem: elem[0], reverse=True)

def eigenvalues(matrix):
    '''
    Return only the eigenvalues in descending order
    '''
    return [eigen[0] for eigen in eigen(matrix)]

def read_file(file_name):
    '''
    Read an input csv file and return a list of values
    '''
    input_file = open(file_name, newline='')
    reader = csv.reader(input_file, delimiter=',')
    for row in reader:
        yield row

def load_matrices(file_name):
    '''
    Reads a file of multiple matrices and returns list of numpy matrices
    '''
    rows = [row for row in  read_file(file_name)]
    row_count = 0
    matrices = []
    while rows[row_count]:
        num_row, num_col = int(rows[row_count][0]), int(rows[row_count][1])
        try:
            vals = rows[row_count+1:row_count+num_row+1]
            matrices.append(read_matrix(vals, num_row, num_col))
        except IndexError:
            print("!! Error: incorrect matrix dimensions.")
        row_count += num_row + 1
        if row_count > len(rows) - 1:
            break
    return matrices

def read_matrix(vals, num_row, num_col):
    '''
    Converts a 2D array of values into a numpy matrix
    '''
    matrix =  []
    for i in range(num_row):
        row = vals[i]
        if num_col > len(row):
            raise IndexError
        matrix.append(list(map(float, row[:num_col])))
    return np.matrix(matrix)