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
from src.clustering.cdclust import kmeans_cluster_inclusion_matrix, multiscale_kmeans_cluster_inclusion_matrix,  multiscale_kmeans_cluster_eid, largest_depth
from src.clustering.inits import initial_clustering
from src.visualization import spaghetti_plot
from src.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering
from skimage.measure import find_contours





def find_boundary(arr):
    rows, cols = arr.shape
    boundary = np.zeros_like(arr)
    
    for i in range(rows):
        for j in range(cols):
            if arr[i, j] == 1:
                if (i > 0 and arr[i - 1, j] == 0) or \
                   (i < rows - 1 and arr[i + 1, j] == 0) or \
                   (j > 0 and arr[i, j - 1] == 0) or \
                   (j < cols - 1 and arr[i, j + 1] == 0):
                    boundary[i, j] = 1
    return boundary

def process_3d_array(arr_3d):
    # 获取三维数组的尺寸
    num_arrays, rows, cols = arr_3d.shape
    
    # 创建一个同样尺寸的三维数组用于存储结果
    boundaries_3d = np.zeros_like(arr_3d)
    
    # 遍历每个二维数组
    for k in range(num_arrays):
        boundaries_3d[k] = find_boundary(arr_3d[k])
    
    return boundaries_3d

def extract_contours(masks):
        contours = []
        for mask in masks:
            # find_contours 返回等高线为列表，假设mask是二维的且为二进制图像
            contour = find_contours(mask, level=0.5)  # level 0.5适合二进制图像
            if contour:  # 仅添加非空轮廓
                contours.append(contour[0])  # 只取第一个轮廓，假设每个掩模只有一个主要轮廓
        return contours
def plot_spaghetti(masks, best_id):
        contours = extract_contours(masks)
        fig, ax = plt.subplots()
        for i, contour in enumerate(contours):
            if i == best_id:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')  # 用红色显示best_id的轮廓
            else:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='black')  # 其他用黑色

        ax.invert_yaxis()  # Y轴反转以符合图像坐标
        ax.set_title('Spaghetti Plot of Mask Contours')
        plt.show()

if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/outlier_separation")
    assert outputs_dir.exists()

    SEED_DATA = 0
    SEED_CLUSTER = 65
    ROWS = COLS = 64
    K = 2

    masks, labs = main_shape_with_outliers(100, ROWS, COLS, return_labels=True, seed=SEED_DATA)
    labs = np.array(labs)
    labs = 1 - labs
    # print(masks)
    masksarray = np.array(masks, dtype=np.float32)
    print(masksarray.shape)
    deepest = largest_depth(masks,depth="eid", metric="depth")
    boundaries_3d = process_3d_array(masksarray)
    # 输出结果的一个示例二维数组
    np.set_printoptions(threshold=np.inf)
    # print(boundaries_3d[0])
    # fig, axes = plt.subplots(1, 2, figsize=(ROWS, COLS))
    # axes[0].imshow(deepest, cmap='gray', interpolation='none')
    # axes[0].set_title("masks")

    # # 在第二个子图中绘制第二个数组
    # axes[1].imshow(boundaries_3d[0], cmap='gray', interpolation='none')
    # axes[1].set_title("boundaries")
   
    # plt.show()
    # 定义一个函数来提取所有掩模的轮廓 
    
    plot_spaghetti(masks, 89)

    ###################
    # Data generation #
    ###################

 
    sdf_mat, pca_mat, transform_mat = get_cvp_sdf_pca_transform(masks, seed=SEED_CLUSTER)
    pred_labs1 = get_cvp_clustering(pca_mat, num_components=K)
    pred_labs2 = kmeans_cluster_inclusion_matrix(masks, num_clusters=K, depth="id", num_attempts=10, max_num_iterations=30, seed=SEED_CLUSTER)
    pred_labs3 = initial_clustering(masks, num_components=K, feat_mat=pca_mat, method="kmeans", k_means_n_init=5, k_means_max_iter=10, seed=SEED_CLUSTER)
    pred_labs4 = kmeans_cluster_inclusion_matrix(masks, num_clusters=K, depth="id", num_attempts=10, max_num_iterations=30, seed=SEED_CLUSTER)
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
    # print(pred_labs4)
    spaghetti_plot(masks, 0.5, arr=pred_labs4, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, ax=axs[4])

    plt.show()

    # individual plots
    for labs_name, labs in [("reference", labs), ("cdclust", pred_labs2), ("kmeans", pred_labs3), ("ahc", pred_labs1)]:
        fig, ax = plt.subplots(figsize=(5,5), layout="tight")
        spaghetti_plot(masks, 0.5, arr=labs, is_arr_categorical=True, smooth=True, smooth_its=1, smooth_kernel_size=1, linewidth=3, ax=ax)
        fig.savefig(outputs_dir.joinpath(f"{labs_name}.png"), dpi=300)