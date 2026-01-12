# Created Date: Wednesday, January 7th 2026, 3:48:06 pm
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Importing necessary libraries
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig
import umap
from sklearn.mixture import GaussianMixture

# Importing custom functions from other files
from one_d_array import to_1d
from Adjust_window import sliding_windows
from Fix_quotas import fix_quotas
from Assign_w_quota_amp import assign_with_quota_amplitude
from Windows_metric import window_amplitude_metric
from Relabel_mean import relabel_by_mean


# Set up variables and parameters
mat_path  = r'Data\Bet286.mat'
var_name  = 'Bet286'
window_size = 80
SEED = 42
K = 3
umap_params = dict(n_neighbors=100, 
                   min_dist=0.1, 
                   n_components=2, 
                   metric='euclidean', 
                   random_state=42)
AMPLITUDE_METRIC = "meanabs"   # "meanabs" | "rms" | "p95"


# ============================================================
# 1) LOAD AND PREPROCESS SERIES
# ============================================================
# Load the .mat file
mat = scipy.io.loadmat(mat_path)

# Verify the variable exists in the file, raise error if not found
if var_name not in mat:
    raise KeyError(f"'{var_name}' is not in {mat_path}. Variables: {list(mat.keys())}")

# Convert the variable to a 1D array using custom function
series = to_1d(mat[var_name], var_name)

# Remove NaN/Inf values and convert to float32 for memory efficiency
series = series[np.isfinite(series)].astype(np.float32)

# Get the length of the cleaned series
N = len(series)

# Print confirmation message with series size
print(f"Series '{var_name}' loaded. N={N}")


# ============================================================
# 2) GMM 1D (SORTED SERIES)
# ============================================================
# Sort the series and get the indices that would sort the array
sort_idx = np.argsort(series)
series_sorted = series[sort_idx]

# Reshape sorted series into a 2D array (required format for sklearn)
X1d = series_sorted.reshape(-1, 1)

# Fit a Gaussian Mixture Model to the 1D sorted data
gmm1d = GaussianMixture(n_components=K,
                        covariance_type='full',
                        random_state=SEED
)
gmm1d.fit(X1d)

# Predict cluster labels for each point in the sorted series
labs_sorted_raw = gmm1d.predict(X1d)

# Relabel clusters by their mean values
labs_sorted, means1d, _ = relabel_by_mean(series_sorted, labs_sorted_raw)

# Map the relabeled clusters back to original time order
# Create an array to hold labels in time-series order
labs_time_1d = np.empty_like(labs_sorted)

# Use inverse of sort_idx to restore original time positions
labs_time_1d[sort_idx] = labs_sorted

# Calculate the weight (proportion) of each cluster
# unique contains the cluster IDs, counts contains how many points in each cluster
unique, counts = np.unique(labs_sorted, return_counts=True)
weights_1d = np.zeros(K)
for k, c in zip(unique, counts):
    # Store the proportion of points in each cluster
    weights_1d[k] = c / float(N)


# ============================================================
# 3) UMAP + GMM 2D WITH QUOTAS
# ============================================================
# Create sliding windows from the time series
Xwin, window_size = sliding_windows(series, window_size)
M = Xwin.shape[0]  # M is the number of windows

# Apply UMAP dimensionality reduction to project windows into 2D space
reducer = umap.UMAP(**umap_params)
U = reducer.fit_transform(Xwin.astype(np.float32))

# Fit a Gaussian Mixture Model to the 2D UMAP representation
# Initialize with weights from the 1D GMM for consistency
gmm2d = GaussianMixture(n_components=K,
                        covariance_type='full',
                        random_state=SEED,
                        weights_init=weights_1d
)
gmm2d.fit(U)

# Get the posterior probabilities (soft assignments) for each window to each cluster
resp = gmm2d.predict_proba(U)

# Calculate target quotas (number of windows per cluster) based on 1D weights
quotas = fix_quotas(weights_1d, M)

# Compute amplitude metric for each window (meanabs, rms, or p95)
amp = window_amplitude_metric(Xwin, how=AMPLITUDE_METRIC)

# Assign windows to clusters while respecting quota constraints and amplitude priorities
labels_2d = assign_with_quota_amplitude(resp, amp, quotas, slack=0.25)


# ============================================================
# 4) RECONSTRUCT TIME INDICES FROM WINDOWS
# ============================================================
# Initialize a list of K empty sets to store time indices for each cluster
cluster_idx_2d = [set() for _ in range(K)]

# Iterate through each window and its assigned cluster label
for i, lab in enumerate(labels_2d):
    # Calculate the start and end time indices for this window
    s, e = i, i + window_size
    
    # Skip if the window starts beyond the series length
    if s >= N:
        break
    
    # Clamp start and end indices to valid range [0, N)
    s = max(0, s)
    e = min(N, e)
    
    # Add all time indices in this window to the corresponding cluster set
    cluster_idx_2d[int(lab)].update(range(s, e))

# Convert each cluster's set of time indices to a sorted numpy array
cluster_time_indices_2d = [np.fromiter(sorted(idx), dtype=int) for idx in cluster_idx_2d]

# Save clustering results to a compressed numpy file for later visualization
np.savez("Data/plot_series_data.npz", 
        series=series,
        cluster_time_indices_2d=np.array(cluster_time_indices_2d, dtype=object),
        labels_2d=labels_2d,
        K=K
)

# Count the number of windows assigned to each cluster
M_k = np.bincount(labels_2d, minlength=K).astype(int)

# ============================================================
# 6) REPORT
# =============================================================
print("[GMM 1D] target weights:", np.round(weights_1d, 4))
print("[UMAP 2D] quotas:", quotas.tolist())
print("[UMAP 2D] cluster sizes (M):", M_k.tolist())
print("[UMAP 2D] proportions:", np.round(M_k / M, 4))
