"""For a clustering results, we show that the final state maximizes 
depth over other configurations.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0, "..")
from src.data.synthetic_data import three_rings, shape_families, magnitude_modes
from src.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix
from src.depth.inclusion_depth import compute_depths
from src.clustering.cdclust import compute_red_within, compute_red_between
from src.visualization import spaghetti_plot, plot_red
from src.clustering.inits import initial_clustering

if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/red_explainer")
    assert outputs_dir.exists()

    seed_data = 0
    seed_init = 0

    num_contours = 30
    # masks, labs = magnitude_modes(num_contours, 512, 512, modes_proportions=(0.5, 0.5), return_labels=True, seed=data_seed)
    masks, labs = three_rings(num_contours, 512, 512, (0.5, 0.3, 0.2), return_labels=True, seed=seed_data)
    #masks, labs = shape_families(num_contours, 512, 512, return_labels=True, seed=data_seed)
    labs_no_part = np.zeros(num_contours, dtype=int)
    labs = np.array(labs)
    num_clusters = np.unique(labs).size
    inclusion_mat = compute_inclusion_matrix(masks)
    epsilon_inclusion_mat = compute_epsilon_inclusion_matrix(masks)

    random_labs = initial_clustering(masks, num_components=num_clusters, feat_mat=None, pre_pca=False, method="random", seed=seed_init)

    rng = np.random.default_rng(seed_data)
    perturbed_labs = labs.copy()
    subset_contour_ids = rng.choice(np.arange(num_contours), 10, replace=False)
    shuffled_contour_clusters = labs[subset_contour_ids].copy()
    rng.shuffle(shuffled_contour_clusters)
    perturbed_labs[subset_contour_ids] = shuffled_contour_clusters


    def get_depth_data(masks, labs, n_components, inclusion_mat, use_modified=False):
        red_w = compute_red_within(masks, labs, n_components=n_components, 
                    depth_notion="id", use_modified=use_modified, use_fast=False, inclusion_mat=inclusion_mat)
        red_b, competing_clusters = compute_red_between(masks, labs, n_components=n_components, competing_clusters=None,
                            depth_notion="id", use_modified=use_modified, use_fast=False, inclusion_mat=inclusion_mat)
        red_i = red_w - red_b
        return red_i, red_w, red_b

    info_strict = []
    info_epsilon = []

    ############
    # Analysis #
    ############

    fig, axs = plt.subplots(nrows=3, ncols=4, layout="tight")

    # p1

    axs[0, 0].set_title("No partitioning")
    spaghetti_plot(masks, 0.5, arr=labs_no_part, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, linewidth=3, ax=axs[0, 0])
    axs[0, 0].set_axis_off()

    unimodal_depths = compute_depths(masks, modified=False, fast=False, inclusion_mat=inclusion_mat)
    info_strict.append((unimodal_depths, unimodal_depths, np.zeros_like(unimodal_depths)))
    axs[1, 0].set_title(f"ID: {unimodal_depths.mean():.4f}")
    plot_red(unimodal_depths, np.zeros_like(unimodal_depths), compute_red=True, labs=labs_no_part, ax=axs[1, 0])

    unimodal_depths = compute_depths(masks, modified=True, fast=False, inclusion_mat=epsilon_inclusion_mat)
    info_epsilon.append((unimodal_depths, unimodal_depths, np.zeros_like(unimodal_depths)))
    axs[2, 0].set_title(f"eID: {unimodal_depths.mean():.4f}")
    plot_red(unimodal_depths, np.zeros_like(unimodal_depths), compute_red=True, labs=labs_no_part, ax=axs[2, 0])

    # p2

    axs[0, 1].set_title("Target labels")
    spaghetti_plot(masks, 0.5, arr=labs, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, linewidth=3, ax=axs[0, 1])
    axs[0, 1].set_axis_off()

    red_i, red_w, red_b = get_depth_data(masks, labs, n_components=num_clusters, inclusion_mat=inclusion_mat, use_modified=False)
    info_strict.append((red_i, red_w, red_b))
    axs[1, 1].set_title(f"ID: {red_i.mean():.4f}")
    plot_red(red_w, red_b, compute_red=True, labs=labs, ax=axs[1, 1])

    red_i, red_w, red_b = get_depth_data(masks, labs, n_components=num_clusters, inclusion_mat=epsilon_inclusion_mat, use_modified=True)
    info_epsilon.append((red_i, red_w, red_b))
    axs[2, 1].set_title(f"eID: {red_i.mean():.4f}")
    plot_red(red_w, red_b, compute_red=True, labs=labs, ax=axs[2, 1])

    # p3

    axs[0,2].set_title("Random labels")
    spaghetti_plot(masks, 0.5, arr=random_labs, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, linewidth=3, ax=axs[0, 2])
    axs[0,2].set_axis_off()

    red_i, red_w, red_b = get_depth_data(masks, random_labs, n_components=num_clusters, inclusion_mat=inclusion_mat, use_modified=False)
    info_strict.append((red_i, red_w, red_b))
    axs[1,2].set_title(f"ID: {red_i.mean():.4f}")
    plot_red(red_w, red_b, compute_red=True, labs=random_labs, ax=axs[1,2])

    red_i, red_w, red_b = get_depth_data(masks, random_labs, n_components=num_clusters, inclusion_mat=epsilon_inclusion_mat, use_modified=True)
    info_epsilon.append((red_i, red_w, red_b))
    axs[2,2].set_title(f"eID: {red_i.mean():.4f}")
    plot_red(red_w, red_b, compute_red=True, labs=random_labs, ax=axs[2,2])

    # p4

    axs[0,3].set_title("Some misplaced labels")
    spaghetti_plot(masks, 0.5, arr=perturbed_labs, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, linewidth=3, ax=axs[0, 3])
    axs[0,3].set_axis_off()

    red_i, red_w, red_b = get_depth_data(masks, perturbed_labs, n_components=num_clusters, inclusion_mat=inclusion_mat, use_modified=False)
    info_strict.append((red_i, red_w, red_b))
    axs[1,3].set_title(f"ID: {red_i.mean():.4f}")
    plot_red(red_w, red_b, compute_red=True, labs=perturbed_labs, ax=axs[1,3])

    red_i, red_w, red_b = get_depth_data(masks, perturbed_labs, n_components=num_clusters, inclusion_mat=epsilon_inclusion_mat, use_modified=True)
    info_epsilon.append((red_i, red_w, red_b))
    axs[2,3].set_title(f"eID: {red_i.mean():.4f}")
    plot_red(red_w, red_b, compute_red=True, labs=perturbed_labs, ax=axs[2,3])

    plt.show()


    #############################
    # Elements for paper figure #
    #############################

    labels = ["no_part", "gt", "random", "misplaced"]

    all_labs = [labs_no_part, labs, random_labs, perturbed_labs]
    for i, v in enumerate(all_labs):
        fig, ax = plt.subplots(layout="tight", figsize=(3, 3))
        spaghetti_plot(masks, 0.5, arr=v, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, linewidth=3, ax=ax)
        ax.set_axis_off()
        fig.savefig(outputs_dir.joinpath(f"{labels[i]}.png"), dpi=300)

    for i, (red_i, red_w, red_b) in enumerate(info_strict):
        fig, ax = plt.subplots(layout="tight", figsize=(3, 3))
        ax.set_title(f"Mean ReD: {red_i.mean():.4f}")
        plot_red(red_w, red_b, compute_red=True, labs=all_labs[i], ax=ax)
        ax.set_xlabel("Contour ID")
        ax.set_ylabel("Depth")
        fig.savefig(outputs_dir.joinpath(f"red-{labels[i]}.png"), dpi=300)