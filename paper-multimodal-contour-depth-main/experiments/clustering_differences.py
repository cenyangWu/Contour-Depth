"""Show cases the difference between outputs by CDclust and AHC (CVP) and KMeans
"""
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.synthetic_data import outlier_cluster
from src.clustering.cdclust import kmeans_cluster_eid, multiscale_kmeans_cluster_inclusion_matrix,multiscale_kmeans_cluster_eid
from src.clustering.inits import initial_clustering
from src.visualization import spaghetti_plot
from src.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering


if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/clustering_differences")
    assert outputs_dir.exists()

    SEED_DATA = 0
    SEED_CLUSTER = 0
    ROWS = COLS = 512
    K = 2

    masks, labs = outlier_cluster(100, ROWS, COLS, True, seed=SEED_DATA)
    labs = np.array(labs)
    labs[labs == 1] = 0  # relabel for evaluation
    labs[labs == 2] = 1  # relabel for evaluation

    ###################
    # Data generation #
    ###################

    sdf_mat, pca_mat, transform_mat = get_cvp_sdf_pca_transform(masks, seed=SEED_CLUSTER)
    pred_labs1 = get_cvp_clustering(pca_mat, num_components=K)
    pred_labs2 = kmeans_cluster_eid(masks, num_clusters=K, num_attempts=20, max_num_iterations=30, seed=28)
    pred_labs3 = initial_clustering(masks, num_components=K, feat_mat=pca_mat, method="kmeans", k_means_n_init=5, k_means_max_iter=10, seed=SEED_CLUSTER)
    # print(f"CVP: {adjusted_rand_score(labs, pred_labs1)}")
    # print(f"CDclust: {adjusted_rand_score(labs, pred_labs2)}")
    # print(f"KMeans: {adjusted_rand_score(labs, pred_labs3)}")

    ############
    # Analysis #
    ############

    fig, axs = plt.subplots(ncols=4, layout="tight")

    spaghetti_plot(masks, 0.5, arr=labs, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[0])
    spaghetti_plot(masks, 0.5, arr=pred_labs1, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[1])
    spaghetti_plot(masks, 0.5, arr=pred_labs2, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[2])
    spaghetti_plot(masks, 0.5, arr=pred_labs3, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[3])

    plt.show()

    # individual plots
    for labs_name, labs in [("reference", labs), ("cdclust", pred_labs2), ("kmeans", pred_labs3), ("ahc", pred_labs1)]:
        fig, ax = plt.subplots(figsize=(5,5), layout="tight")
        spaghetti_plot(masks, 0.5, arr=labs, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, linewidth=3, ax=ax)
        fig.savefig(outputs_dir.joinpath(f"{labs_name}.png"), dpi=300)