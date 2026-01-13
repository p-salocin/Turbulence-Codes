# Created Date: Monday, January 12th 2026, 12:43:19 pm
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Defining function
def _louvain_levels(G):
    """
    Compute the hierarchical community structure of a graph using the Louvain method.

    Args:
        G: A NetworkX graph object to analyze for community structure.
    
    Returns:
        list: A list of community partitions at each hierarchical level, where:
            - Index 0 represents the finest level (most detailed communities)
            - Index (L-1) represents the root level (coarsest partition, single community)
            - L is the total number of hierarchical levels detected
    """
    
    return G.community_multilevel(return_levels=True)


