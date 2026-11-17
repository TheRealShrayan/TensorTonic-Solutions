import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    matrix=np.array(A)
    result=np.transpose(matrix)
    return result