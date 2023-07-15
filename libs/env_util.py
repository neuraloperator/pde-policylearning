import matlab.engine
import numpy as np

def to_m(numpy_a):
    if type(numpy_a) == float:
        return matlab.double(numpy_a)
    elif type(numpy_a) == int:
        return matlab.int64(numpy_a)
    else:
        return matlab.double(numpy_a.tolist())


def relative_loss(A, B):
    """
    Compute the relative loss between matrices A and B.
    """
    numerator = np.linalg.norm(A - B)  # Numerator: norm of the difference between A and B
    denominator = np.linalg.norm(A)  # Denominator: norm of matrix A
    
    # Handle division by zero case
    if denominator == 0:
        return 0.0
    
    # Compute relative loss
    loss = numerator / denominator
    
    return loss



def apply_periodic_boundary(arr, mod_length=3, axis=0):
    """
    Apply periodic boundary condition to a NumPy array along the first axis.
    
    Parameters:
    - arr: The input NumPy array.
    - mod_length: The length to apply the periodic boundary condition.
    
    Returns:
    A new array with periodic boundary condition applied along the first axis.
    """
    return np.take(arr, np.arange(arr.shape[0]) % mod_length, axis=axis)
