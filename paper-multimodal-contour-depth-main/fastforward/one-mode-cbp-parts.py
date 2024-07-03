""" Generates the different parts of the unimodal contour boxplot.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from src.data.han_seg_ensembles import get_han_ensemble
from src.depth import inclusion_depth
from src.visualization import get_bp_depth_elements
from src.clustering.cdclust import compute_red, kmeans_cluster_eid

from skimage.filters import gaussian

SHOW_FIGS = False

# Load data 
DATA_DIR = Path("data/han_ensembles/")
PATIENT_ID = "HCAI-036"
OAR, SLICE_NUM = [("BrainStem", 31), ("Parotid_R", 41)][1]

img_vol, gt_vol, masks_vol = get_han_ensemble(DATA_DIR, 
                                              patient_id=PATIENT_ID,
                                              structure_name=OAR,
                                              slice_num=None)
img = img_vol[SLICE_NUM]
gt = gt_vol[SLICE_NUM]
masks = [m[SLICE_NUM] for m in masks_vol]
smooth_masks = [gaussian(m) for m in masks]

print(img.min(), img.max())

# General visualization functions
def get_fig_ax():
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img, cmap="gray", vmin=-350, vmax=450)
    return fig, ax

def style_fig(fig, ax):
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])

def save_fig(fig, ax, fn):
    fig.savefig(f"fastforward/assets/{fn}.png", dpi=500)
    print(f"Saved {fn}")

###################
# SINGLE MODE CBP #
###################
full_dephts = inclusion_depth.compute_depths(masks, modified=True, fast=True)
depth_elements = get_bp_depth_elements(masks, full_dephts, outlier_type="percent", epsilon_out=0.1)

#print(depth_elements)

smooth_median = depth_elements[0]["representatives"]["masks"][0]
smooth_median = gaussian(smooth_median)

smooth_band_50 = depth_elements[0]["bands"]["masks"][1]
smooth_band_100 = depth_elements[0]["bands"]["masks"][0]
smooth_band_50 = gaussian(smooth_band_50)
smooth_band_100 = gaussian(smooth_band_100)

smooth_outliers = depth_elements[0]["outliers"]["masks"]
smooth_outliers = [gaussian(sm) for sm in smooth_outliers]

# Spaghetti plot
fig, ax = get_fig_ax()
for sm in smooth_masks:
    ax.contour(sm, levels=[0.5,], colors=["#66c2a5", ], linewidths=[1,], alpha=0.3)
style_fig(fig, ax)
if SHOW_FIGS:
    plt.show()
else:
    save_fig(fig, ax, "spaghetti")

# CBP (mean)
fig, ax = get_fig_ax()
ax.contour(smooth_median, levels=[0.5,], colors=["yellow", ], linewidths=[3,], alpha=1)
style_fig(fig, ax)
if SHOW_FIGS:
    plt.show()
else:
    save_fig(fig, ax, "1mcbp-mean")

# CBP (add confidence intervals)
fig, ax = get_fig_ax()
ax.contourf(smooth_band_50, levels=[0.5, 1.5], colors=["purple", ], alpha=0.5 * 0.3)
ax.contourf(smooth_band_100, levels=[0.5, 1.5], colors=["plum", ], alpha=1.0 * 0.3)
ax.contour(smooth_median, levels=[0.5,], colors=["yellow", ], linewidths=[3,], alpha=1)
style_fig(fig, ax)
if SHOW_FIGS:
    plt.show()
else:
    save_fig(fig, ax, "1mcbp-mean-bands")

# CBP (add outliers)
fig, ax = get_fig_ax()
for smo in smooth_outliers:
    ax.contour(smo, levels=[0.5,], colors=["red", ], linewidths=[1,], linestyles=["dashed",], alpha=0.5)
ax.contourf(smooth_band_50, levels=[0.5, 1.5], colors=["purple", ], alpha=0.5 * 0.3)
ax.contourf(smooth_band_100, levels=[0.5, 1.5], colors=["plum", ], alpha=1.0 * 0.3)
ax.contour(smooth_median, levels=[0.5,], colors=["yellow", ], linewidths=[3,], alpha=1)
style_fig(fig, ax)
if SHOW_FIGS:
    plt.show()
else:
    save_fig(fig, ax, "1mcbp-mean-bands-outs")

##################
# MULTI MODE CBP #
##################
N_COMPONENTS = 2
SEED_CDCLUST = 1
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3', "red"]
pred_labs = kmeans_cluster_eid(masks, num_clusters=N_COMPONENTS, metric="depth", num_attempts=5, max_num_iterations=10, seed=SEED_CDCLUST)
red_i, red_w, red_b, competing_clusters = compute_red(masks, pred_labs, n_components=N_COMPONENTS) 
depth_elements = get_bp_depth_elements(masks, red_i, pred_labs, outlier_type="percent", epsilon_out=0.1)

print(list(depth_elements.keys()))

#print(depth_elements)

smooth_medians = [depth_elements[i]["representatives"]["masks"][0] for i in range(N_COMPONENTS)]
smooth_medians = [gaussian(sme) for sme in smooth_medians]

smooth_bands_50 = [depth_elements[i]["bands"]["masks"][1] for i in range(N_COMPONENTS)]
smooth_bands_100 = [depth_elements[i]["bands"]["masks"][0] for i in range(N_COMPONENTS)]
smooth_bands_50 = [gaussian(smb) for smb in smooth_bands_50]
smooth_bands_100 = [gaussian(smb) for smb in smooth_bands_100]

smooth_outliers = [depth_elements[i]["outliers"]["masks"] for i in range(N_COMPONENTS)]
smooth_outliers = [[gaussian(smo) for smo in smos] for smos in smooth_outliers]

print(len(smooth_medians))

#############
# All modes #
#############

# CBP (mean)
fig, ax = get_fig_ax()
for i, sme in enumerate(smooth_medians):
    ax.contour(sme, levels=[0.5,], colors=[colors[i], ], linewidths=[3,], alpha=1)
style_fig(fig, ax)
if SHOW_FIGS:
    plt.show()
else:
    save_fig(fig, ax, f"{N_COMPONENTS}mcbp-mean")

# CBP (add confidence intervals)
fig, ax = get_fig_ax()
# for i, smb50 in enumerate(smooth_bands_50):
#     ax.contourf(smooth_band_50, levels=[0.5, 1.5], colors=[colors[i], ], alpha=0.5 * 0.3)
for i, smb100 in enumerate(smooth_bands_100):
    ax.contourf(smb100, levels=[0.5, 1.5], colors=[colors[i], ], alpha=1.0 * 0.3)
for i, sme in enumerate(smooth_medians):
    ax.contour(sme, levels=[0.5,], colors=[colors[i], ], linewidths=[3,], alpha=1)
style_fig(fig, ax)
if SHOW_FIGS:
    plt.show()
else:
    save_fig(fig, ax, f"{N_COMPONENTS}mcbp-mean-bands")

# CBP (add outliers)
fig, ax = get_fig_ax()
for i, smos in enumerate(smooth_outliers):
    for smo in smos:
        ax.contour(smo, levels=[0.5,], colors=[colors[i], ], linewidths=[1,], linestyles=["dashed",], alpha=0.5)
for i, smb100 in enumerate(smooth_bands_100):
    ax.contourf(smb100, levels=[0.5, 1.5], colors=[colors[i], ], alpha=1.0 * 0.3)
for i, sme in enumerate(smooth_medians):
    ax.contour(sme, levels=[0.5,], colors=[colors[i], ], linewidths=[3,], alpha=1)
style_fig(fig, ax)
if SHOW_FIGS:
    plt.show()
else:
    save_fig(fig, ax, f"{N_COMPONENTS}mcbp-mean-bands-outs")

##################
# Separate modes #
##################

for mi in range(N_COMPONENTS):

    # CBP (mean)
    fig, ax = get_fig_ax()
    ax.contour(smooth_medians[mi], levels=[0.5,], colors=[colors[mi], ], linewidths=[3,], alpha=1)
    style_fig(fig, ax)
    if SHOW_FIGS:
        plt.show()
    else:
        save_fig(fig, ax, f"{N_COMPONENTS}mcbp-m{mi}-mean")

    # CBP (add confidence intervals)
    fig, ax = get_fig_ax()
    # for i, smb50 in enumerate(smooth_bands_50):
    #     ax.contourf(smooth_band_50, levels=[0.5, 1.5], colors=[colors[i], ], alpha=0.5 * 0.3)
    ax.contourf(smooth_bands_100[mi], levels=[0.5, 1.5], colors=[colors[mi], ], alpha=1.0 * 0.3)
    ax.contour(smooth_medians[mi], levels=[0.5,], colors=[colors[mi], ], linewidths=[3,], alpha=1)
    style_fig(fig, ax)
    if SHOW_FIGS:
        plt.show()
    else:
        save_fig(fig, ax, f"{N_COMPONENTS}mcbp-m{mi}-mean-bands")

    # CBP (add outliers)
    fig, ax = get_fig_ax()
    for smo in smooth_outliers[mi]:
        ax.contour(smo, levels=[0.5,], colors=[colors[mi], ], linewidths=[1,], linestyles=["dashed",], alpha=0.5)
    ax.contourf(smooth_bands_100[mi], levels=[0.5, 1.5], colors=[colors[mi], ], alpha=1.0 * 0.3)
    ax.contour(smooth_medians[mi], levels=[0.5,], colors=[colors[mi], ], linewidths=[3,], alpha=1)
    style_fig(fig, ax)
    if SHOW_FIGS:
        plt.show()
    else:
        save_fig(fig, ax, f"{N_COMPONENTS}mcbp-m{mi}-mean-bands-outs")
