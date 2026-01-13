# Created Date: Monday, January 12th 2026, 12:59:36 pm
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig

# Importing custom functions from other files
from Metagraph import _build_metagraph
from HVG import _hvg_edges
from louvain import _louvain_levels
from Labels_from_win import labels_time_from_windows


# Defining function
def plot_metagraph_with_umap_clusters(
    series,
    labels_windows=None,      # UMAP labels per window (preferred)
    window_size=None,         # required if labels_windows is used
    labels_time=None,         # optional: labels per point already computed (UMAP 2D or 1D)
    nivel="auto",
    colors=None,              # list of 3 RGBA colors for Cluster 1,2,3
    seed=42,
    max_nodes_plot=300,
    ax=None 
):
    """
    Plots a metagraph with UMAP cluster coloring using hierarchical community detection.
    
    Parameters:
    - series: 1D vector of the time series
    - labels_windows + window_size: derive labels_time (per point) by voting
      If you already have labels_time, pass it and omit labels_windows
    - nivel: "auto" (uses -2 if root is unique, else -1) or int (specific level)
    - colors: 3 colors (Cluster 1,2,3). If None, uses inferno colormap
    - seed: random seed for reproducibility
    - max_nodes_plot: maximum number of communities to display
    
    Returns:
    Dictionary with graph, community detection results, and cluster assignments
    """
    # Convert series to float numpy array
    x = np.asarray(series, dtype=float)
    N = len(x)

    # 1) Derive per-point labels from window labels or validate provided labels
    if labels_time is None:
        if labels_windows is None or window_size is None:
            raise ValueError("Provide labels_windows + window_size or labels_time.")
        K = int(np.max(labels_windows)) + 1
        if K != 3:
            raise ValueError("Expected 3 clusters (0,1,2).")
        
        # Convert window-based labels to point-based labels via voting
        labels_time = labels_time_from_windows(np.asarray(labels_windows, dtype=int), int(window_size), N, K=3)
    else:
        labels_time = np.asarray(labels_time, dtype=int)
        if labels_time.shape[0] != N:
            raise ValueError("labels_time must have length N=len(series).")

    # 2) Build Horizontal Visibility Graph and run multi-level Louvain community detection
    edges = _hvg_edges(x)  # Extract HVG edges from time series
    G = ig.Graph(n=N, edges=edges, directed=False)  # Create graph with N nodes and HVG edges
    levels = _louvain_levels(G)  # Run hierarchical Louvain algorithm

    # 3) Select the appropriate hierarchical level
    if nivel == "auto":
        
        # Use -2 if root has single community, else -1
        idx = -2 if len(levels[-1]) == 1 and len(levels) > 1 else -1
    elif isinstance(nivel, int):
        idx = nivel if nivel >= 0 else len(levels) + nivel
        if idx < 0 or idx >= len(levels):
            raise ValueError(f"nivel out of range: {nivel} (have {len(levels)} levels)")
    else:
        raise ValueError("nivel must be 'auto' or an int")

    # Get community membership from selected level
    membership = np.array(levels[idx].membership, dtype=int)
    C = membership.max() + 1  # Total number of communities

    # 3) Assign each community a UMAP cluster label by majority vote
    comm_cluster = np.zeros(C, dtype=int)
    for c in range(C):
        nodes = np.where(membership == c)[0]  # Nodes belonging to community c
        if nodes.size == 0:
            comm_cluster[c] = 0
        else:
            # Count votes for each cluster {0,1,2} among nodes in community
            votes = np.bincount(labels_time[nodes], minlength=3)
            comm_cluster[c] = int(np.argmax(votes))  # Assign most common cluster

    # 4) Build metagraph (one node per community) and style it
    MG = _build_metagraph(G, membership)

    # Set vertex colors based on cluster assignment (3 clusters)
    if colors is None:
        cmap = plt.cm.get_cmap('inferno')
        colors = [cmap(0.15), cmap(0.50), cmap(0.85)]
    vcolors = [colors[comm_cluster[c]] for c in range(C)]

    # # Scale vertex sizes proportionally to community size
    # sizes = np.array(MG.vs["size"], dtype=float)
    # smin, smax = float(sizes.min()), float(sizes.max())
    # if smax > smin:
    #     vsz = 10.0 + 28.0 * (sizes - smin) / (smax - smin)  # Normalize to [10, 38]
    # else:
    #     vsz = np.full(MG.vcount(), 16.0, dtype=float)
    # vsz = list(map(float, vsz))

    # --- Scale vertex sizes (visually robust) ---
    sizes = np.array(MG.vs["size"], dtype=float)

    if sizes.max() > 0:
        sizes_norm = sizes / sizes.max()          # [0, 1]
        vsz = 2.0 + 18.0 * sizes_norm              # ⬅️ smaller & clearer
    else:
        vsz = np.full(MG.vcount(), 10.0, dtype=float)

    vsz = list(map(float, vsz))


    # Scale edge widths proportionally to edge weight
    if "weight" in MG.es.attributes() and len(MG.es["weight"]) > 0:
        w = np.array(MG.es["weight"], dtype=float)
        wmax = float(w.max()) if w.size else 1.0
        edgew = list(1.0 + 4.0 * (w / wmax)) if wmax > 0 else [1.0] * MG.ecount()
    else:
        edgew = [1.0] * MG.ecount()

    # If too many communities, sample the largest ones
    VG = MG
    if MG.vcount() > max_nodes_plot:
        order = np.argsort(-sizes)[:max_nodes_plot]  # Get indices of largest communities
        VG = MG.subgraph(order.tolist())  # Create subgraph with largest communities
        
        # Map old vertex indices to new indices (for reference, though unused here)
        sub_map = {old: new for new, old in enumerate(order.tolist())}
        
        # Re-sample colors and sizes for the subgraph
        vcolors = [colors[comm_cluster[i]] for i in order.tolist()]
        vsz = [vsz[i] for i in order.tolist()]

    # Compute layout using Fruchterman-Reingold with stable seed
    rng = np.random.default_rng(seed)
    init = rng.random((VG.vcount(), 2)).tolist()  # Random initial positions
    layout = VG.layout_fruchterman_reingold(niter=500, seed=init)

    # Plot the metagraph
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ig.plot(
        VG,
        layout=layout,
        vertex_size=vsz,
        vertex_color=vcolors if VG is MG else vcolors,  # Use filtered colors
        edge_width=edgew if VG is MG else None,         # Edge widths (auto if subgraph)
        edge_color="darkgray",
        target=ax,
        bbox=(1500, 1500),
        margin=0
    )

    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")

    return {
        "G": G,                          # Original HVG
        "levels": levels,                # All hierarchical levels from Louvain
        "nivel_usado": idx,              # Index of selected level
        "membership": membership,        # Community assignment per node
        "MG": MG,                        # Metagraph
        "community_cluster": comm_cluster,   # 0/1/2 per community (consistent with UMAP)
        "labels_time": labels_time           # 0/1/2 per point
    }
