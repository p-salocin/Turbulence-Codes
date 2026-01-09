# Created Date: Friday, January 9th 2026, 11:37:21 am
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Importing necessary libraries
import numpy as np


# Defining function
def relabel_by_mean(values, labels):
    """
    Relabel array elements based on mean values of groups, sorted in ascending order.
    Parameters
    ----------
    values : array-like
        Array of numerical values corresponding to each label.
    labels : array-like
        Array of labels identifying which group each value belongs to.
        Must be the same length as values.
    Returns
    -------
    new_labels : ndarray
        Relabeled array with sequential integer labels (0, 1, 2, ...) assigned
        based on ascending order of group means.
    means : dict
        Dictionary mapping original labels to their corresponding mean values.
    mapping : dict
        Dictionary mapping original labels to their new sequential labels.
    """
    # Get unique labels from the input array
    uniq = np.unique(labels)
    
    # Calculate the mean value for each unique label
    means = {lab: values[labels == lab].mean() for lab in uniq}
    
    # Sort labels by their corresponding mean values in ascending order
    order = sorted(uniq, key=lambda k: means[k])
    
    # Create a mapping from old labels to new sequential labels (0, 1, 2, ...)
    mapping = {old: new for new, old in enumerate(order)}
    
    # Apply the mapping to relabel the original labels array
    new_labels = np.vectorize(mapping.get)(labels)
    
    # Return the relabeled array, the means dictionary, and the mapping dictionary
    return new_labels, means, mapping
