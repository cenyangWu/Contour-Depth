

# Load dataset
# Signed distance function
# Principal component analysis (matrix)
# Agglomerative hierarchical clustering
# Contour variability plots
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from skimage.measure import regionprops

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")
from src.data.synthetic_data import three_rings
from src.visualization import spaghetti_plot, colors
from src.utils import get_sdfs, get_masks_matrix
from src.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering, get_cvp_pca_medians, get_per_cluster_mean, transform_from_pca_to_sdf, get_cvp_bands_parts

colors_cvp = [colors[2], colors[1], colors[0]] 

def plot_dendrogram(model, **kwargs):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot_cvp(masks, sdf_mat, pca_mat, transform_mat, num_components, ax=None):    

    # - CVP analysis
    pred_labs  = get_cvp_clustering(pca_mat, num_components=num_components)

    clusters_ids, clusters_counts = np.unique(pred_labs, return_counts=True)
    ccount_argsort = np.argsort(clusters_counts)[::-1]
    new_pred_labs = np.ones_like(pred_labs)*-1 
    for i, old_cid in enumerate(ccount_argsort):
        new_pred_labs[pred_labs == old_cid] = i
    pred_labs = new_pred_labs

    pca_medians = get_cvp_pca_medians(pca_mat, pred_labs)
    sdf_means = get_per_cluster_mean(sdf_mat, pred_labs)
    #raw_medians = transform_from_pca_to_sdf(np.array(pca_medians)*0.1, np.array(sdf_means), transform_mat)
    raw_medians = transform_from_pca_to_sdf(np.array(pca_medians)*0.0, np.array(sdf_means), transform_mat)  # using the mean
    medians = []
    for i in range(num_components):        
        median = raw_medians[i].reshape(masks[0].shape)
        # for _ in range(5):
        #     median = gaussian(median, sigma=10)
        medians.append(median)

    band_imgs = []
    bands_sf = []
    bands_parts = get_cvp_bands_parts(sdf_mat, pred_labs, std_mult=1)
    for band_num in range(num_components):

        parts = bands_parts[band_num]
        top = parts[0].reshape(masks[0].shape)
        bottom = parts[1].reshape(masks[0].shape)

        temp_band = np.zeros(top.shape)
        temp_band[top<=0] = 2
        temp_band[bottom<=0] = 0
        temp_band[np.logical_and(top>0, bottom>0)] = 1
        
        # for _ in range(5):
        #             temp_band = gaussian(temp_band, sigma=10)
        bands_sf.append(temp_band)  

    if ax is None:
        fig, ax = plt.subplots(layout="tight")
    for cluster_id in range(num_components):
        ax.contour(medians[cluster_id], levels=[0, ], colors=[colors_cvp[cluster_id], ], linewidths=[1,])
        ax.contourf(bands_sf[cluster_id], levels=[0.5, 1.5], colors=[colors_cvp[cluster_id], ], alpha=(100/100) * 0.3)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for sp in ['top', 'right', 'bottom', 'left']:
        ax.spines[sp].set_visible(False)

    #add_legend(pred_labs, (40, 100), colors_cvp, ax)    
        

if __name__ == "__main__":

    N = 100
    SEED_DATA = 0
    ROWS = COLS = 512
    K = 3

    masks, labs = three_rings(N, ROWS, COLS, return_labels=True, seed=0)        
    labs = np.array(labs)

    area_img = (np.array(masks).sum(0) > 0).astype(int)
    rp = regionprops(area_img)[0]
    bbox = rp["bbox"]
    masks = [m[bbox[0]-1:bbox[2]+1, bbox[1]-1:bbox[3]+1] for m in masks]

    masks_arr = get_masks_matrix(masks)

    sdfs = get_sdfs(masks)

    sdf_mat, pca_mat, transform_mat = get_cvp_sdf_pca_transform(masks)

    # sdf

    # fig, ax = plt.subplots(layout="tight", figsize=(5,5))
    # ax.imshow(masks[0], cmap="gray")
    # ax.set_axis_off()
    # fig.savefig("outputs/cvp-bin.png", dpi=300)

    # fig, ax = plt.subplots(layout="tight", figsize=(5,5))
    # ax.imshow(sdfs[0], cmap="viridis")
    # ax.set_axis_off()    
    # fig.savefig("outputs/cvp-sdf.png", dpi=300)

    # dendrogram

    # ahc = AgglomerativeClustering(distance_threshold=0, n_clusters=None, metric="euclidean", linkage="average").fit(pca_mat)

    # fig, ax = plt.subplots(layout="tight", figsize=(3,6))
    # plot_dendrogram(ahc, ax=ax)
    # ax.set_axis_off()
    # fig.savefig("outputs/cvp-dendrogram.png", dpi=300)

    # cvp

    fig, ax = plt.subplots(layout="tight", figsize=(5,5))
    plot_cvp(masks, sdf_mat, pca_mat, transform_mat, num_components=K, ax=ax)
    ax.invert_yaxis()
    fig.savefig("outputs/cvp-plot.png", dpi=300)
    #plt.show()