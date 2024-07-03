"""
Computes inclusion matrices for simple ensemble of contours (mock-ensemble)
We then use the values in the plots to construct the depiction in the paper.
"""
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix

masks = [imread(f"mock-ensemble/mock-ensemble-m{i+1}.png") for i in range(6)]
masks = [rgb2gray(m[:, :, 0:3]) for m in masks]
masks = [m.astype(int) for m in masks]

strict_inclusion_mat = compute_inclusion_matrix(masks)
np.fill_diagonal(strict_inclusion_mat, 1)
epsilon_inclusion_mat = compute_epsilon_inclusion_matrix(masks)
np.fill_diagonal(epsilon_inclusion_mat, 1)

fig, axs = plt.subplots(ncols=3, figsize=(10, 4), layout="tight")

axs[0].set_title("Ensemble")
for m in masks:
    axs[0].contour(m, colors=["orange"])
axs[0].set_axis_off()

axs[1].set_title("Strict inclusion mat")
axs[1].matshow(strict_inclusion_mat)

axs[2].set_title("Epsilon inclusion mat")
axs[2].matshow(epsilon_inclusion_mat)

# axs[3].set_title("mat_epsilon x inv(mat_strict)")
# axs[3].matshow(unknown_matrix)

plt.show()


##################################
# THRESHOLDING OF EPSILON MATRIX #
##################################

fig, axs = plt.subplots(ncols=3)
axs[0].set_title("Strict inclusion mat")
axs[0].matshow(strict_inclusion_mat)
axs[1].set_title("Epsilon inclusion mat")
axs[1].matshow(epsilon_inclusion_mat)
axs[2].set_title("Epsilon inclusion mat (thresholded)")
axs[2].matshow(epsilon_inclusion_mat > 0.98)
plt.show()


##################################
# QUANTIZATION OF EPSILON MATRIX #
##################################

num_bins = 7
bin_size = 1 / num_bins
quantized_mat = epsilon_inclusion_mat.copy()
for i in range(num_bins):
    quantized_mat[np.logical_and(quantized_mat > i * bin_size, quantized_mat <= (i+1) * bin_size)] = i
# quantized_mat[np.logical_and(quantized_mat > 1 * bin_size, quantized_mat <= 2 * bin_size)] = 1
# quantized_mat[np.logical_and(quantized_mat > 2 * bin_size, quantized_mat <= 3 * bin_size)] = 2
# quantized_mat[np.logical_and(quantized_mat > 3 * bin_size, quantized_mat <= 4 * bin_size)] = 3
# quantized_mat[np.logical_and(quantized_mat > 4 * bin_size, quantized_mat <= 5 * bin_size)] = 4

fig, axs = plt.subplots(ncols=2)
axs[0].set_title("Epsilon inclusion mat")
axs[0].matshow(epsilon_inclusion_mat)
axs[1].set_title("Epsilon inclusion mat (quantized)")
axs[1].matshow(quantized_mat)
plt.show()




# 


def multiscale_kmeans_cluster_inclusion_matrix(masks, num_clusters, depth="ecbd", metric="depth", num_attempts=5, size_window=60,max_num_iterations=10, seed=42):
    assert(depth in ["eid", "id", "cbd", "ecbd"])
    assert(metric in ["depth", "red"])
    masks = np.array(masks, dtype=np.float32)
    print(masks.shape)
    num_masks = masks.shape[0]
    size_row = masks.shape[1]
    size_col = masks.shape[2]

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)  
    
    def check_valid_assignment(assignment, num_clusters):
        for c in range(num_clusters):
            if np.sum(assignment == c) < 3:
                return False
        return True

    best_depth_sum = -np.inf
    best_cluster_assignment = None
    for _ in range(num_attempts):
        print(_)
        cluster_assignment = rng.integers(low=0, high=num_clusters, size=num_masks)
        # 生成随机的分类
        for _ in range(max_num_iterations):
            depth_in_cluster = np.empty((num_clusters, size_row-size_window, size_col-size_window, num_masks), dtype=np.float32)
            for i in range (size_row-size_window):
                for j in range (size_col-size_window):
                    for k in range (num_masks):
                        window = np.zeros((num_masks,size_window, size_window))
                        window[k] = masks[k,i:i+size_window,j:j+size_window]
                    if depth == "eid" or depth == "ecbd":
                        inclusion_matrix = compute_epsilon_inclusion_matrix(window)
                        np.fill_diagonal(inclusion_matrix, 1) # Required for feature parity with the O(N) version of eID.
                    else:
                        inclusion_matrix = compute_inclusion_matrix(window)
                        # print(inclusion_matrix)
                    for c in range(num_clusters):
                        j_in_cluster = cluster_assignment == c
                        
                        N = np.sum(j_in_cluster)#该类的数量
                        if depth == "cbd" or depth == "id" or depth == "eid":
                            N_a = np.sum(inclusion_matrix[:,j_in_cluster], axis=1)#包含矩阵中与该类相关的部分的加和
                            N_b = np.sum(inclusion_matrix.T[:,j_in_cluster], axis=1)

                        if depth == "cbd":
                            # We need to normalize the depth such that it is  not dependent on the number of contours in the cluster.
                            # If the contour is already in the cluster then N_a and N_b range from 0 to N-1
                            # If the contour is *not* in the cluster then N_a and N_b range from 0 to N
                            N_ab_range = N - j_in_cluster
                            depth_in_cluster[c][i][j] = (N_a * N_b) / (N_ab_range * N_ab_range)
                        if depth == "ecbd":
                            i_in_j = inclusion_matrix[:,j_in_cluster]
                            j_in_i = inclusion_matrix.T[:,j_in_cluster]
                            # We need to normalize the depth such that it is  not dependent on the number of contours in the cluster.
                            depth_in_cluster[c,i,j,:] = np.minimum(i_in_j, j_in_i) / N
                        else: # ID / eID
                            depth_in_cluster[c][i][j] = np.minimum(N_a, N_b) / N
                  
                        #else: # eCBD (epsilon Cluster Band Depth)
                        #    i_in_j = inclusion_matrix[:,j_in_cluster]
                        #    j_in_i = inclusion_matrix.T[:,j_in_cluster]
                        #    # We need to normalize the depth such that it is  not dependent on the number of contours in the cluster.
                        #    depth_in_cluster[c] = np.minimum(i_in_j, j_in_i) / N
            print(depth_in_cluster.shape)
            if metric == "depth":
                metric_values = np.empty((num_clusters, num_masks), dtype=np.float32)
                for c in range(num_clusters):
                    metric_values[c] = np.mean(depth_in_cluster[c], axis=(0, 1))
            else: # Relative Depth (ReD)
                red = np.empty((num_clusters, num_masks), dtype=np.float32)
                depth_in_cluster_mean = np.empty((num_clusters, num_masks), dtype=np.float32)
                for c in range(num_clusters):
                    depth_in_cluster_mean[c] = np.mean(depth_in_cluster[c], axis=(0, 1))
                for c in range(num_clusters):
                    # Compute the max value exluding the current cluster.
                    # There is a more efficient, but slightly dirtier, solution.
                    depth_between = np.max(np.roll(depth_in_cluster_mean, -c, axis=0)[1:,:], axis=0)
                    depth_within = depth_in_cluster_mean[c,:]
                    red[c,:] = depth_within - depth_between
                metric_values = red

            old_cluster_assignment = cluster_assignment
            print(metric_values)
            cluster_assignment = np.argmax(metric_values, axis=0)
            print(cluster_assignment)
            print("try")
            if not check_valid_assignment(cluster_assignment, num_clusters) or np.all(cluster_assignment == old_cluster_assignment):
                best_cluster_assignment = cluster_assignment
                print("best")
                break

            depth_sum = np.sum(np.choose(cluster_assignment, metric_values))
            if depth_sum > best_depth_sum:
                best_cluster_assignment = cluster_assignment
                best_depth_sum = depth_sum

    return best_cluster_assignment