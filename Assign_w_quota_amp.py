# Created Date: Friday, January 9th 2026, 11:53:29 am
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Importing necessary libraries
import numpy as np


# Defining function
def expand_pool(base_order, need, current_size, slack):
    """
    Expands the pool of candidates based on the required size and slack.

    Parameters:
    base_order (array): The ordered array of candidates.
    need (int): The number of candidates needed.
    current_size (int): The current size of the pool.
    slack (float): The factor to increase the pool size.

    Returns:
    int: The new size of the pool.
    """
    # Check if the current size of the pool is already sufficient
    if current_size >= need:
        return current_size  # Return the current size if no expansion is needed
    
    # Calculate the extra candidates to add based on the slack factor
    extra = int(np.ceil(need * slack))
    
    # Return the new size of the pool, ensuring it does not exceed the total number of candidates
    return min(need + extra, len(base_order))


def assign_with_quota_amplitude(resp, amp, quotas, slack=0.25):
    """
    Assigns samples to three classes (0, 1, 2) based on quotas and amplitude ordering.
    
    Parameters:
    resp (array): Response matrix of shape (N, 3) with scores for each class.
    amp (array): Amplitude values used to order candidates.
    quotas (array): Target number of samples for each class [q0, q1, q2].
    slack (float): Factor to expand pool size beyond quota (default 0.25).
    
    Returns:
    array: Labels assigned to each sample (-1 if unassigned initially).
    """
    N, K = resp.shape
    assert K == 3, "This routine assumes K = 3 (low, medium, high)"
    q0, q1, q2 = quotas.tolist()
    labels = -np.ones(N, dtype=int)  # Initialize all labels as unassigned

    # Sort indices by amplitude: ascending (low) and descending (high)
    order_low  = np.argsort(amp)
    order_high = np.argsort(-amp)

    # Create initial pools for low and high amplitude candidates
    pool0 = order_low[:max(q0, 0)]
    pool2 = order_high[:max(q2, 0)]

    # Expand pools based on slack factor
    pool0_size = expand_pool(order_low,  q0, len(pool0), slack)
    pool2_size = expand_pool(order_high, q2, len(pool2), slack)
    pool0 = order_low[:pool0_size]
    pool2 = order_high[:pool2_size]

    taken = np.zeros(N, dtype=bool)  # Track which samples have been assigned

    # Assign class 0 (low amplitude) from pool0, selecting by highest resp[:, 0] score
    if q0 > 0 and pool0.size:
        cand = pool0[~taken[pool0]]
        pick = cand[np.argsort(-resp[cand, 0])[:q0]]
        labels[pick] = 0; taken[pick] = True

    # Assign class 2 (high amplitude) from pool2, selecting by highest resp[:, 2] score
    # Dynamically expand pool2 if needed to meet quota
    if q2 > 0 and pool2.size:
        while np.sum(labels == 2) < q2:
            cand = pool2[~taken[pool2]]
            missing = q2 - np.sum(labels == 2)
            if cand.size == 0:
                # Expand pool2 if no candidates available
                pool2_size = min(pool2_size + max(1, int(q2*slack)), N)
                pool2 = order_high[:pool2_size]
                cand = pool2[~taken[pool2]]
                if cand.size == 0:
                    break
            pick = cand[np.argsort(-resp[cand, 2])[:missing]]
            labels[pick] = 2; taken[pick] = True

    # Assign class 1 (medium) from remaining unassigned samples
    remaining = np.where(~taken)[0]
    if remaining.size and q1 > 0:
        sel = remaining[np.argsort(-resp[remaining, 1])[:q1]]
        labels[sel] = 1; taken[sel] = True

    # Assign unassigned samples to the class with highest resp score
    unassigned = np.where(labels < 0)[0]
    if unassigned.size:
        labels[unassigned] = np.argmax(resp[unassigned], axis=1)
    
    return labels
