# Created Date: Monday, January 12th 2026, 12:36:40 pm
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Importing necessary libraries
import numpy as np


# Defining function
def _hvg_edges(values):
    """
    Generate edges for a Horizontal Visibility Graph (HVG) from a time series.

    Args:
        values (array-like): A sequence of numerical values representing a time series.
    
    Returns:
        list: A list of tuples representing edges as (source_index, target_index) pairs.
    
    """
    # Convert input values to a numpy array of floats
    x = np.asarray(values, dtype=float)
    
    # Initialize empty list for edges and stack to track indices
    edges, stack = [], []
   
    # Iterate through each index in the time series
    for j in range(len(x)):
        # Remove indices from stack where current value is greater than stacked value
        while stack and x[j] > x[stack[-1]]:
            i = stack.pop()
            
            # Add edge between popped index and current index
            edges.append((i, j))
        
        # If stack is not empty, add edge between top of stack and current index
        if stack:
            edges.append((stack[-1], j))
        
        # Push current index onto the stack
        stack.append(j)
    
    # Return unique edges if any exist, otherwise return empty list
    return list(set(edges)) if edges else edges