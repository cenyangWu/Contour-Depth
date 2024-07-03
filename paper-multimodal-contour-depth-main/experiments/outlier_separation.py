"""Show cases the use of CDclust to remove outliers.
"""

"""Show cases the difference between outputs by CDclust and AHC (CVP) and KMeans
"""
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.synthetic_data import main_shape_with_outliers
from src.clustering.cdclust import kmeans_cluster_inclusion_matrix, multiscale_kmeans_cluster_inclusion_matrix
from src.clustering.inits import initial_clustering
from src.visualization import spaghetti_plot
from src.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering




if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/outlier_separation")
    assert outputs_dir.exists()

    SEED_DATA = 0
    SEED_CLUSTER = 0
    ROWS = COLS = 64
    K = 2

    masks, labs = main_shape_with_outliers(100, ROWS, COLS, return_labels=True, seed=SEED_DATA)
    labs = np.array(labs)
    labs = 1 - labs

    ###################
    # Data generation #
    ###################

 
    sdf_mat, pca_mat, transform_mat = get_cvp_sdf_pca_transform(masks, seed=SEED_CLUSTER)
    pred_labs1 = get_cvp_clustering(pca_mat, num_components=K)
    pred_labs2 = kmeans_cluster_inclusion_matrix(masks, num_clusters=K, depth="id", num_attempts=5, max_num_iterations=10, seed=SEED_CLUSTER)
    pred_labs3 = initial_clustering(masks, num_components=K, feat_mat=pca_mat, method="kmeans", k_means_n_init=5, k_means_max_iter=10, seed=SEED_CLUSTER)
    pred_labs4 = kmeans_cluster_inclusion_matrix(masks, num_clusters=K, depth="id", num_attempts=5, max_num_iterations=10, seed=28)
    print(pred_labs4.shape)
    print(f"CVP: {adjusted_rand_score(labs, pred_labs1)}")
    print(f"CDclust: {adjusted_rand_score(labs, pred_labs2)}")
    print(f"KMeans: {adjusted_rand_score(labs, pred_labs3)}")
    print(f"MultiscaleCD: {adjusted_rand_score(labs, pred_labs4)}")
    ############
    # Analysis #
    ############

    fig, axs = plt.subplots(ncols=5, layout="tight")

    spaghetti_plot(masks, 0.5, arr=labs, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[0])
    spaghetti_plot(masks, 0.5, arr=pred_labs1, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[1])
    spaghetti_plot(masks, 0.5, arr=pred_labs2, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[2])
    spaghetti_plot(masks, 0.5, arr=pred_labs3, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[3])
    print(pred_labs4.shape)
    print(pred_labs4)
    spaghetti_plot(masks, 0.5, arr=pred_labs4, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[4])

    plt.show()

    # individual plots
    for labs_name, labs in [("reference", labs), ("cdclust", pred_labs2), ("kmeans", pred_labs3), ("ahc", pred_labs1)]:
        fig, ax = plt.subplots(figsize=(5,5), layout="tight")
        spaghetti_plot(masks, 0.5, arr=labs, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, linewidth=3, ax=ax)
        fig.savefig(outputs_dir.joinpath(f"{labs_name}.png"), dpi=300)