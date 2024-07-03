# Dataset 1
# Dataset 2
# Dataset 3
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")
from src.data.synthetic_data import outlier_cluster, main_shape_with_outliers, three_rings
from src.visualization import spaghetti_plot

def plot_sd_overview(masks, labs, fname):
        area_img = (np.array(masks).sum(0) > 0).astype(int)
        rp = regionprops(area_img)[0]
        bbox = rp["bbox"]
        new_masks = [m[bbox[0]-1:bbox[2]+1, bbox[1]-1:bbox[3]+1] for m in masks]
        fig, ax = plt.subplots(layout="tight", figsize=(5, 5))
        spaghetti_plot(new_masks, iso_value=0.5, arr=labs, is_arr_categorical=True, smooth=True, ax=ax)
        fig.savefig(f"outputs/{fname}.png", dpi=300)

if __name__ == "__main__":
        N = 100
        SEED_DATA = 0
        ROWS = COLS = 512
        K = 2

        masks, labs = outlier_cluster(N, ROWS, COLS, return_labels=True, seed=1)
        labs = np.array(labs)
        labs[labs == 1] = 0  # relabel for evaluation
        labs[labs == 2] = 1  # relabel for evaluation
        plot_sd_overview(masks, labs, "sd-outlier-cluster")

        masks, labs = main_shape_with_outliers(N, ROWS, COLS, return_labels=True, seed=0)
        labs = np.array(labs)
        labs = 1 - labs
        plot_sd_overview(masks, labs, "sd-main-shape-with-outliers")

        masks, labs = three_rings(N, ROWS, COLS, return_labels=True, seed=0)        
        labs = np.array(labs)
        num_clusters = np.unique(labs).size
        plot_sd_overview(masks, labs, "sd-three-rings")

        

        