import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    m = A.shape[1]
    n = A.shape[0]
    A_transpose = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            A_transpose[j][i] = A[i][j]
    return A_transpose
