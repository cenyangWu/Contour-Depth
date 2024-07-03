"""Generates figure demonstrating clustering analysis on meteorological data.
We attempted to run the Matlab code provided by the authors of the CVP paper, but we could not.
Therefore, this script replicates their analysis in Python based on their paper and matlab code.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize, EuclideanTransform, warp
from skimage.io import imread

import sys
sys.path.insert(0, "..")
from src.data import ecmwf_ensembles as ecmwf
from src.clustering.cdclust import compute_red, kmeans_cluster_eid
from src.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering, get_cvp_pca_medians, get_cvp_bands, get_cvp_bands_parts, transform_from_pca_to_sdf, get_per_cluster_mean
from src.visualization import spaghetti_plot, plot_contour_boxplot, colors

colors_cvp = [colors[2], colors[1], colors[0]] 
colors_cdclust = colors_cvp # [colors[1], colors[0], colors[2]] 

from src.visualization import get_bp_cvp_elements, get_bp_depth_elements

from skimage.filters import gaussian

def add_legend(labels, r0c0, cluster_colors, ax):
    from matplotlib.patches import Rectangle    
    RECT_HEIGHT = 50
    RECT_WIDTH = 500
    bar_x0 = r0c0[1]
    BAR_Y0 = r0c0[0]

    num_contours = labels.size
    clusters_ids, num_contours_cluster = np.unique(labels, return_counts=True)

    for cluster_id in clusters_ids:
        cluster_color = cluster_colors[cluster_id]
        bar_width = RECT_WIDTH*(num_contours_cluster[cluster_id]/num_contours)        
        rect = Rectangle((bar_x0, BAR_Y0), bar_width, RECT_HEIGHT, color=cluster_color, edgecolor=None)
        ax.add_patch(rect)
        bar_x0 += bar_width

    return ax

if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/rd_meteo")
    assert outputs_dir.exists()

    # Path to data
    data_dir = Path("../data/cvp-paper-meteo/")

    seed_init = 1
    seed_ddclust = 1

    ########
    # Data #
    ########

    masks = ecmwf.load_data(data_dir, config_id=0)    
    masks = [m.data[::-1,:] for m in masks]  # masks are masked arrays with a `data` and `mask` entries

    bg_img = imread(data_dir.joinpath("picking_background.png"), as_gray=True)
    flipped_img = bg_img[::-1, :]
    print(flipped_img.shape)
    
    # - CVP analysis
    extent = (70, 1920, 150, 1300-50)
    sdf_mat, pca_mat, transform_mat = get_cvp_sdf_pca_transform(masks, seed=0)
    pred_labs  = get_cvp_clustering(sdf_mat, num_components=3)

    cid, ccount = np.unique(pred_labs, return_counts=True)
    ccount_argsort = np.argsort(ccount)[::-1]
    new_pred_labs = np.ones_like(pred_labs)*-1 
    for i, old_cid in enumerate(ccount_argsort):
        new_pred_labs[pred_labs == old_cid] = i
    pred_labs = new_pred_labs

    pca_medians = get_cvp_pca_medians(pca_mat, pred_labs)
    sdf_means = get_per_cluster_mean(sdf_mat, pred_labs)
    raw_medians = transform_from_pca_to_sdf(np.array(pca_medians)*0.1, np.array(sdf_means), transform_mat)
    medians = []
    for i in range(3):        
        median = raw_medians[i].reshape(masks[0].shape)
        median = resize(median, (extent[3]-extent[2], extent[1]-extent[0]), order=1)
        for _ in range(5):
            median = gaussian(median, sigma=10)
        medians.append(median)

    band_imgs = []
    bands_sf = []
    bands_parts = get_cvp_bands_parts(sdf_mat, pred_labs, std_mult=1)
    for band_num in range(3):

        parts = bands_parts[band_num]
        top = parts[0].reshape(masks[0].shape)
        bottom = parts[1].reshape(masks[0].shape)

        temp_band = np.zeros(top.shape)
        temp_band[top<=0] = 2
        temp_band[bottom<=0] = 0
        temp_band[np.logical_and(top>0, bottom>0)] = 1
        
        temp_band = resize(temp_band, (extent[3]-extent[2], extent[1]-extent[0]), order=1)
        for _ in range(5):
                    temp_band = gaussian(temp_band, sigma=10)
        bands_sf.append(temp_band)    
    
    # - CVP depths
    red_i, red_w, _, _ = compute_red(masks, pred_labs, n_components=3, use_modified=True, use_fast=True)
    depth_cluster_statistics_cvp = get_bp_depth_elements(masks, red_w, pred_labs, outlier_type="percent", epsilon_out=0.1)

    # - depth clustering
    pred_labs_cdclust = kmeans_cluster_eid(masks, num_clusters=3, metric="depth")    
    red_i_cdclust, red_w_cdclust, _, _ = compute_red(masks, pred_labs_cdclust, n_components=3)
    depth_cluster_statistics_cdclust = get_bp_depth_elements(masks, red_w_cdclust, pred_labs_cdclust, outlier_type="percent", epsilon_out=0.1)

    # -- reorder cd clust labels in decreasing order
    cid, ccount = np.unique(pred_labs_cdclust, return_counts=True)
    ccount_argsort = np.argsort(ccount)[::-1]    
    new_pred_labs = np.ones_like(pred_labs_cdclust)*-1 
    new_depth_cluster_statistics_cdclust = {}
    for i, old_cid in enumerate(ccount_argsort):
        new_pred_labs[pred_labs_cdclust == old_cid] = i
        new_depth_cluster_statistics_cdclust[i] = depth_cluster_statistics_cdclust[old_cid]
    pred_labs_cdclust = new_pred_labs
    depth_cluster_statistics_cdclust = new_depth_cluster_statistics_cdclust
    # colors_cdclust = [colors_cdclust[i] for i in ccount_argsort]


    #################
    # Visualization #
    #################

    # start_point = (145, 72)  # OG
    start_point = (1000, 0)
    # target_size = (1292, 1914)  # OG
    target_size = (int(1292/1), int(1914/1))  # OG

    # (1921, 83, 1920, 1080) left botton width and height from fig position in matlab code
    # (1 100 1920 870) left botton width and height from plotpos in matlab code
    img_size = (1080, 1920)
    rs_bg_img = resize(bg_img, img_size)    

    tform_order = 1
    area_within_img = (870, 1920-35)
    start_corner = (-70, -0)

    extent = (70, 1920, 150, 1300-50)

    median_linewidth = 2
    general_linewidth = 1


    # * Spaghetti *
    if True:
        fig, ax = plt.subplots(layout="tight")
        ax.imshow(bg_img, cmap="gray")        

        color_ids = np.random.choice(np.arange(len(colors)), len(masks), replace=True)
        spaghetti_colors = [colors[e] for e in color_ids]

        for i, m in enumerate(masks):
            tmp_mask = resize(m, (extent[3]-extent[2], extent[1]-extent[0]), order=1)
            for _ in range(5):
                tmp_mask = gaussian(tmp_mask, sigma=10)
            ax.contour(tmp_mask, levels=[0.5, ], colors=[spaghetti_colors[i], ], extent=extent, alpha=0.5)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for sp in ['top', 'right', 'bottom', 'left']:
            ax.spines[sp].set_visible(False)

        # plt.show()

        fig.savefig(outputs_dir.joinpath(f"spaghetti-plot.png"), dpi=300)



    # * CVP *
    if True:
        
        fig, ax = plt.subplots(layout="tight")
        ax.imshow(bg_img, cmap="gray")
        for cluster_id in range(3):
            ax.contour(medians[cluster_id], levels=[0, ], colors=[colors_cvp[cluster_id], ], linewidths=[median_linewidth,], extent=extent)
            ax.contourf(bands_sf[cluster_id], levels=[0.5, 1.5], colors=[colors_cvp[cluster_id], ], alpha=(100/100) * 0.3, extent=extent)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for sp in ['top', 'right', 'bottom', 'left']:
            ax.spines[sp].set_visible(False)

        # plt.show()

        add_legend(pred_labs, (40, 100), colors_cvp, ax)

        fig.savefig(outputs_dir.joinpath(f"cvp_boxplot.png"), dpi=300)

    
    # * Depth *    
    if False:
        medians = []
        bands_sf = []
        outliers = []
        for cluster_id in range(3):
            cluster_info = depth_cluster_statistics_cvp[cluster_id]
            median = cluster_info["representatives"]["masks"][0]
            median = resize(median, (extent[3]-extent[2], extent[1]-extent[0]), order=0)
            for _ in range(5):
                median = gaussian(median, sigma=10)
            medians.append(median)

            band = cluster_info["bands"]["masks"][0] # 100% band
            for _ in range(0):
                    band = gaussian(band, sigma=1)
            bands_sf.append(band) 
            
            outs = []
            for out in cluster_info["outliers"]["masks"]:
                new_out = out
                new_out = resize(new_out, (extent[3]-extent[2], extent[1]-extent[0]), order=0)
                for _ in range(5):
                    new_out = gaussian(new_out, sigma=10)
                outs.append(new_out)
            outliers.append(outs)

        fig, ax = plt.subplots(layout="tight")
        ax.imshow(bg_img, cmap="gray")
        for cluster_id in range(3):
            ax.contour(medians[cluster_id], levels=[0.5, ], colors=[colors_cvp[cluster_id], ], linewidths=[3,], extent=extent)
            ax.contourf(bands_sf[cluster_id], levels=[0.5, 1.5], colors=[colors_cvp[cluster_id], ], alpha=(100/100) * 0.3, extent=extent)
            for out in outliers[cluster_id]:
                ax.contour(out, levels=[0.5, ], colors=[colors[cluster_id], ], linestyles=["dashed",], linewidths=[1,], extent=extent)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for sp in ['top', 'right', 'bottom', 'left']:
            ax.spines[sp].set_visible(False)

        # plt.show()

        fig.savefig(outputs_dir.joinpath(f"depth-cvp_boxplot.png"), dpi=300)


    # * Depth clustering *
    if True:
        medians = {}
        bands_sf = {}
        outliers = {}
        clusters_ids = [0, 1, 2, ] #list(range(3))
        for cluster_id in clusters_ids:
            cluster_info = depth_cluster_statistics_cdclust[cluster_id]

            median = cluster_info["representatives"]["masks"][0]
            median = resize(median, (extent[3]-extent[2], extent[1]-extent[0]), order=0)
            for _ in range(5):
                median = gaussian(median, sigma=10)
            medians[cluster_id] = median

            band = cluster_info["bands"]["masks"][0] # 100% band
            band = resize(band, (extent[3]-extent[2], extent[1]-extent[0]), order=1)
            for _ in range(5):
                    band = gaussian(band, sigma=10)
            bands_sf[cluster_id] = band
            
            outs = []
            for out in cluster_info["outliers"]["masks"]:
                new_out = out
                new_out = resize(new_out, (extent[3]-extent[2], extent[1]-extent[0]), order=0)
                for _ in range(5):
                    new_out = gaussian(new_out, sigma=10)
                outs.append(new_out)
            outliers[cluster_id] = outs

        fig, ax = plt.subplots(layout="tight")
        ax.imshow(bg_img, cmap="gray")
        for cluster_id in clusters_ids:
            ax.contour(medians[cluster_id], levels=[0.5, ], colors=[colors_cdclust[cluster_id], ], linewidths=[median_linewidth,], extent=extent)
            ax.contourf(bands_sf[cluster_id], levels=[0.5, 1.5], colors=[colors_cdclust[cluster_id], ], alpha=(100/100) * 0.3, extent=extent)
            # for out in outliers[cluster_id]:
            #     ax.contour(out, levels=[0.5, ], colors=[colors[cluster_id], ], linestyles=["dashed",], linewidths=[1,], extent=extent)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for sp in ['top', 'right', 'bottom', 'left']:
            ax.spines[sp].set_visible(False)

        add_legend(pred_labs_cdclust, (40, 100), colors_cdclust, ax)

        # plt.show()

        fig.savefig(outputs_dir.joinpath(f"depth-cluster_boxplot.png"), dpi=300)


    
    # idx = np.where(pred_labs_cdclust == 2)[0] 
    # cols = 4
    # rows = idx.size // cols + 1
    # fig, axs = plt.subplots(ncols=cols, nrows=rows)
    # axs = axs.flatten()
    # for i, j in enumerate(idx):
    #     axs[i].set_title(j)
    #     axs[i].imshow(masks[j])
    # plt.show()
    





