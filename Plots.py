import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data = np.load("Data/plot_series_data.npz", allow_pickle=True)

series = data["series"]
cluster_time_indices_2d = data["cluster_time_indices_2d"]
labels_2d = data["labels_2d"]
K = int(data["K"])

# # ============================================================
# 5) FIGURE (b) ONLY — SERIES + INSET PIE (CODE2 STYLE)
# ============================================================
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

cmap = plt.cm.get_cmap('inferno')
cluster_colors = [cmap(0.05), cmap(0.50), cmap(0.95)]

fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)

# --- time series scatter ---
for k in range(K):
    idx = cluster_time_indices_2d[k]
    if idx.size:
        ax.scatter(
            idx,
            series[idx],
            s=6,
            color=cluster_colors[k],
            alpha=0.95
        )

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\varepsilon(t)$')
ax.text(
    0.01, 0.97, '(b)',
    transform=ax.transAxes,
    ha='left', va='top',
    fontsize=18, fontweight='bold'
)

# --- inset pie ---
M_k = np.bincount(labels_2d, minlength=K).astype(int)
labels_pie = [f'Cluster {i+1}' for i in range(K)]

ax_in = inset_axes(
    ax,
    width="36%", height="36%",
    loc='upper right',
    bbox_to_anchor=(0, 0, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0.6
)

ax_in.pie(
    M_k,
    labels=labels_pie,
    startangle=90,
    counterclock=False,
    colors=cluster_colors
)
ax_in.set_aspect('equal', 'box')

plt.show()
