""" Experiment testing whether ReD can be used to determine the optimal
number of clusters.
"""
from pathlib import Path
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.data.synthetic_data import magnitude_modes, three_rings, shape_families
from src.utils import get_masks_matrix, get_sdfs


from src.clustering.cdclust import compute_sil, compute_red, kmeans_cluster_eid
from src.visualization import plot_clustering_eval, spaghetti_plot

from src.depth.utils import compute_epsilon_inclusion_matrix

if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/num_clust_selection")
    assert outputs_dir.exists()

    # files exist? 
    masks_path = outputs_dir.joinpath("masks.pkl")
    clusterings_path = outputs_dir.joinpath("clusterings.pkl")
    sils_path = outputs_dir.joinpath("sils.pkl")
    reds_path = outputs_dir.joinpath("reds.pkl")
    if clusterings_path.exists() and sils_path.exists() and reds_path.exists():
        data_exists = True
    else:
        data_exists = False

    if not data_exists:
        print("Data does not exist, computing it ...")
        seed_data = 0
        seed_clustering = [0,1,2,3,4,5,6,7,8,9]  # Before it was only 1, we added more seeds to have more trials

        num_contours = 100
        # masks, labs = magnitude_modes(num_contours, 521, 512, return_labels=True, seed=data_seed)
        masks, labs = three_rings(num_contours, 512, 512, return_labels=True, seed=seed_data)
        # masks, labs = shape_families(num_contours, 512, 512, return_labels=True, seed=data_seed)
        labs = np.array(labs)
        num_clusters = np.unique(labs).size

        with open(masks_path, "wb") as f:
            pickle.dump(masks, f)

        # precompute matrix

        sdfs = get_sdfs(masks)
        sdfs_mat = get_masks_matrix(sdfs)
        sdfs_mat_red = PCA(n_components=50, random_state=seed_data).fit_transform(sdfs_mat)

        #inclusion_mat = compute_inclusion_matrix(masks)
        inclusion_mat = compute_epsilon_inclusion_matrix(masks)

        ###################
        # Clustering algo #
        ###################

        # Input: data, labels (labels have been assigned by a method or randomly)
        # Input: number of clusters K

        ks = list(range(2, 10))
        clusterings = dict()
        sils = dict()
        reds = dict()

        for k in ks:
            for s in seed_clustering:
                print(k, s)
                #pred_labs = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=seed_clustering).fit_predict(sdfs_mat_red)
                pred_labs = kmeans_cluster_eid(masks, k, metric="depth", num_attempts=5, max_num_iterations=10, seed=s)
                # pred_labs = kmeans_cluster_inclusion_mat(masks, k, threshold=0.99, metric="depth", num_attempts=5, max_num_iterations=10, seed=seed_clustering)
                sil_i, _, _, _ = compute_sil(sdfs_mat_red, pred_labs, n_components=k)
                red_i, _, _, _ = compute_red(masks, pred_labs, n_components=k, competing_clusters=None, depth_notion="id", use_modified=True, use_fast=False, inclusion_mat=inclusion_mat)
                
                if k not in clusterings:
                    clusterings[k] = []
                if k not in sils:
                    sils[k] = []
                if k not in reds:
                    reds[k] = []
                
                clusterings[k].append(pred_labs)
                sils[k].append(sil_i.mean())
                reds[k].append(red_i.mean())
        
        with open(clusterings_path, "wb") as f:
            pickle.dump(clusterings, f)
        with open(sils_path, "wb") as f:
            pickle.dump(sils, f)
        with open(reds_path, "wb") as f:
            pickle.dump(reds, f)
    else:
        print("Data exists, loading it ...")
        with open(masks_path, "rb") as f:
            masks = pickle.load(f)
        with open(clusterings_path, "rb") as f:
            clusterings = pickle.load(f)
        with open(sils_path, "rb") as f:
            sils = pickle.load(f)
        with open(reds_path, "rb") as f:
            reds = pickle.load(f)

    # print(clusterings)
    # print(reds)
    # print(sils)

    #############################
    # Elements for paper figure #
    #############################
            
    # Setup data

    ks = list(clusterings.keys())

    df = pd.DataFrame(reds).T
    df = df.reset_index()
    df = df.melt(id_vars="index")

    # PLOT: Plot individual elements

    plt.clf()
    fig, ax1 = plt.subplots(layout="tight", figsize=(4, 3))

    for s in range(10):
        reds_list = [reds[k][s] for k in ks]
        plot_clustering_eval(ks, reds_list,                          
                            metric_a_id="ReD", metric_a_lab="Average ReD",
                            #metric_b=sils, metric_b_id="Sil", metric_b_lab="Average Sil", 
                            ax=ax1)
    ax1.plot(ks, [reds[k][6] for k in ks], c="blue")

    ax1.set_title("Average ReD per K-clustering (6 in blue)")

    fig.savefig(outputs_dir.joinpath("clust-eval-red-all-samples.png"), dpi=300)

    # PLOT: Plot aggregate
    xmin_id = df.groupby("index").mean().value.argmin()
    xmax_id = df.groupby("index").mean().value.argmax()
    xmin = ks[xmin_id]
    xmax = ks[xmax_id]
    
    plt.clf()
    fig, ax1 = plt.subplots(layout="tight", figsize=(4, 3))
    sns.lineplot(df, x="index", y="value", color="orange", ax=ax1)
    ax1.axvline(x=xmin, c="orange", linestyle="--")
    ax1.axvline(x=xmax, c="orange", linestyle="--")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("$\mu$ReD", usetex=True)
    ax1.set_title("$\mu$ReD per K-clustering", usetex=True)
    #plt.show()
    fig.savefig(outputs_dir.joinpath("clust-eval-red.png"), dpi=300)

    # Plot ensembles

    SAMPLE_ID = 6
    reds_sample = [reds[k][SAMPLE_ID] for k in ks]

    xmin_id = np.argmin(reds_sample)
    xmax_id = np.argmax(reds_sample)
    xmin = ks[xmin_id]
    xmax = ks[xmax_id]

    labels = ["min", "max"]

    all_labs = [clusterings[xmin][SAMPLE_ID], clusterings[xmax][SAMPLE_ID]]
    for i, v in enumerate(all_labs):
        fig, ax = plt.subplots(layout="tight", figsize=(3, 3))
        spaghetti_plot(masks, 0.5, arr=v, is_arr_categorical=True, ax=ax)
        ax.set_axis_off()
        fig.savefig(outputs_dir.joinpath(f"{labels[i]}.png"), dpi=300)


    for k in ks:
        fig, ax = plt.subplots(layout="tight", figsize=(1, 1))
        spaghetti_plot(masks, 0.5, arr=clusterings[k][SAMPLE_ID], is_arr_categorical=True, ax=ax)
        ax.set_axis_off()
        fig.savefig(outputs_dir.joinpath(f"clustering_k{k}.png"), dpi=300)

