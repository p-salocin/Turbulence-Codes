# Created Date: Friday, January 9th 2026, 11:47:08 am
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Importing necessary libraries
import numpy as np


# Defining function
def window_amplitude_metric(Xwin, how="meanabs"):
    """
    Calculate the amplitude metric of a given window of data.

    Parameters:
    Xwin : numpy.ndarray
        A 2D array where each row represents a window of data.
    how : str, optional
        The method to calculate the amplitude. Options are:
        - "rms": Root Mean Square
        - "p95": 95th percentile of absolute values
        - "meanabs": Mean of absolute values (default)

    Returns:
    numpy.ndarray
        An array of amplitude metrics calculated for each window.
    """
    if how == "rms":
        # Calculate the root mean square of each window
        amp = np.sqrt(np.mean(Xwin**2, axis=1))
    elif how == "p95":
        # Calculate the 95th percentile of absolute values for each window
        amp = np.percentile(np.abs(Xwin), 95, axis=1)
    else:
        # Calculate the mean of absolute values for each window
        amp = np.mean(np.abs(Xwin), axis=1)
    
    # Return the amplitude metrics as a float32 array
    return amp.astype(np.float32)


