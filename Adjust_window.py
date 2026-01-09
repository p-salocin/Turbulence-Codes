# Created Date: Thursday, January 8th 2026, 5:33:07 pm
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Importing necessary libraries
import numpy as np


# Defining function
def sliding_windows(series, w):
    """
    Build overlapping sliding windows from a 1D time series.

    Parameters
    ----------
    series : array-like
        1D time series (x).
    w : int
        Desired window length (number of samples per window).

    Returns
    -------
    X : ndarray
        2D array of shape (n_windows, w) containing the overlapping windows.
        Each row is a window from the series.
    w : int
        The (possibly adjusted) window length actually used.

    """
    n = len(series)

    # If the requested window is too large, adjust it to guarantee at least one window.
    if n <= w:
        # Choose a fallback window size:
        w = max(10, min(n-1, n//10 if n >= 20 else 10))
        print(f"[Warning!] Window size adjusted to {w}")

    # Construct overlapping windows. This creates a new array (copies data).
    # Result shape: (n - w + 1, w)
    X = np.array([series[i:i+w] for i in range(n-w+1)], dtype=series.dtype)

    return X, w