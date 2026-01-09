# Created Date: Friday, January 9th 2026, 11:40:58 am
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Importing necessary libraries
import numpy as np


# Defining function
def fix_quotas(weights, N):
    """
    Allocate N items proportionally to weights using the largest remainder method.
    
    Parameters
    ----------
    weights : array-like
        1D array of non-negative weights (proportions) summing to 1.
    N : int
        The total number of items to allocate.
    Returns
    -------
    q : ndarray
        An integer array of length len(weights) containing the allocation for each
        party. The sum of q will always equal N.
    """
    # Calculate initial quotas by flooring weighted allocation
    q = np.floor(weights * N).astype(int)
    
    # Calculate the difference between target and current allocation
    deficit = N - q.sum()
    
    # If deficit is positive, allocate remaining seats to highest residues
    if deficit > 0:
        residues = (weights * N) - q
        for k in np.argsort(-residues)[:deficit]:
            q[k] += 1
    
    # If deficit is negative, remove seats from lowest residues
    elif deficit < 0:
        residues = (weights * N) - q
        for k in np.argsort(residues)[:(-deficit)]:
            q[k] -= 1
    
    return q


