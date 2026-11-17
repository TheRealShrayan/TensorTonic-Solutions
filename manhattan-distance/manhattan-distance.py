import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    array1=np.asarray(x,dtype=float)
    array2=np.asarray(y,dtype=float)
    distance=np.sum(np.abs(array1-array2))
    return float(distance)
  