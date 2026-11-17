import numpy as np

def dot_product(x: list, y: list) -> float:
    array1=np.asarray(x,dtype=float)
    array2=np.asarray(y,dtype=float)
    dot_product=np.dot(array1,array2)
    return float(dot_product)