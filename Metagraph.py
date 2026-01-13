# Created Date: Monday, January 12th 2026, 12:46:50 pm
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Importing necessary libraries
import igraph as ig
import numpy as np


# Defining function
def _build_metagraph(G, membership):
    """
    Builds a metagraph from a graph and its community structure.
    
    Args:
        G: An igraph Graph object
        membership: List of community IDs for each node
    
    Returns:
        MG: A new igraph Graph where nodes are communities and edges represent inter-community connections
    """
    # Convert membership list to numpy array of integers for efficient indexing
    cid_of = np.asarray(membership, dtype=int)
    
    # Calculate total number of communities
    C = int(cid_of.max()) + 1

    # Dictionary to store edge weights between communities
    W = {}
    
    # Iterate through all edges in the original graph
    for u, v in G.get_edgelist():
        
        # Get community IDs for both endpoints
        cu, cv = cid_of[u], cid_of[v]
        
        # Skip edges within the same community (intra-community edges)
        if cu == cv:
            continue
        
        # Normalize edge direction: smaller community ID comes first
        a, b = (cu, cv) if cu < cv else (cv, cu)
       
       # Increment weight for this inter-community edge
        W[(a, b)] = W.get((a, b), 0) + 1

    # Create new empty metagraph
    MG = ig.Graph()
    
    # Add vertices (one per community)
    MG.add_vertices(C)
    
    # Calculate community sizes using bincount
    sizes = np.bincount(cid_of, minlength=C).astype(int)
    
    # Assign sizes as vertex attributes
    MG.vs["size"] = sizes.tolist()

    # If there are inter-community edges, add them to the metagraph
    if W:
        e_list = list(W.keys())
        w_list = [int(W[e]) for e in e_list]
        MG.add_edges(e_list)
        MG.es["weight"] = w_list
    else:
        # No inter-community edges case
        MG.es["weight"] = []
    
    return MG

