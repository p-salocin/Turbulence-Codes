# Created Date: Thursday, January 8th 2026, 12:30:03 pm
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Importing necessary libraries

import numpy as np


# Defining function

def to_1d (x, name):
    """Coerce input `x` (a time series in .mat files) to a 1D float numpy array.

    Parameters
    - x: array-like or scalar — input time series (samples ordered in time)
    - name: str — variable name used in the error message when coercion fails

    Returns
    - arr: np.ndarray of shape (N,) and dtype float

    Raises
    - ValueError: if the resulting object cannot be converted to a 1D array
    """

    # Creates a numpy array; this may be object-dtype if nested
    arr = np.array(x)

    # Unwrap object-dtype containers (common when loading from .mat files)
    # If the array holds a single element that itself is an array, extract it.
    # Otherwise flatten and concatenate all elements after coercing them to
    # numpy arrays (and squeezing singleton dims).
    
    while arr.dtype == object:
        if arr.size == 1:
            arr = np.array(arr.item())
        else:
            arr = np.concatenate([np.array(e).squeeze() for e in arr.flatten()])

    # Force numeric dtype and remove any singleton dimensions
    arr = np.asarray(arr, dtype=float).squeeze()

    # If it ended up with a column/row vector like shape (N,1) or (1,N),
    # reshape to a flat 1D array.
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)

    # If result is not 1D, raise an error indicating the shape.
    if arr.ndim != 1:
        raise ValueError(f"'{name}' is not 1D array (shape={arr.shape}).")
    
    return arr