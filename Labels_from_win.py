# Created Date: Monday, January 12th 2026, 12:53:58 pm
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Importing necessary libraries
import numpy as np


# Defining function
def labels_time_from_windows(labels_windows, window_size, N, K=3):
    """
    Generates time-index labels (0..N-1) by voting across overlapping windows.
    Uses K difference arrays with cumulative sum for O(N+M) efficiency.
    
    Args:
        labels_windows: Array of class labels, one per window
        window_size: Size of each window
        N: Total number of time points
        K: Number of classes (default=3)
    
    Returns:
        Array of predicted labels for each time index
    """
    # Initialize difference array (K classes × N+1 time points)
    # Used for efficient range updates via cumsum trick
    M = len(labels_windows)
    diff = np.zeros((K, N+1), dtype=np.int32)
    
    # For each window, increment votes at start and decrement at end
    for i, lab in enumerate(labels_windows):
        s = i                              # Window start index
        e = min(N, i + window_size)        # Window end index (capped at N)
        if s >= N:                         # Skip if window starts beyond N
            break
        diff[lab, s] += 1                  # Start voting for this class
        diff[lab, e] -= 1                  # Stop voting at window end
    
    # Cumulative sum converts difference array to vote counts
    votes = np.cumsum(diff[:, :N], axis=1)  # Shape: (K, N)
    
    # Return class with highest votes at each time index
    return np.argmax(votes, axis=0)         # Shape: (N,)


