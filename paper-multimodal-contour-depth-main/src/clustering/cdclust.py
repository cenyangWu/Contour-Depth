
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from src.depth.inclusion_depth import compute_depths as inclusion_depths
from src.depth.inclusion_depth import inclusion_depth_modified_fast
from src.depth.inclusion_depth import get_precompute_in, get_precompute_out
from src.depth.band_depth import compute_depths as band_depths
from src.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix
from src.utils import get_masks_matrix, get_sdfs


def multiscale_kmeans_cluster_inclusion_matrix(masks, num_clusters, depth="ecbd", metric="depth", num_attempts=5, size_window=60,max_num_iterations=10, seed=42):
    assert(depth in ["eid", "id", "cbd", "ecbd"])
    assert(metric in ["depth", "red"])
    masks = np.array(masks, dtype=np.float32)
    print(masks.shape)
    num_masks = masks.shape[0]
    size_row = masks.shape[1]
    size_col = masks.shape[2]
    np.set_printoptions(threshold=np.inf)
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
    return 0
    for _ in range(num_attempts):
        print(_)
        cluster_assignment = rng.integers(low=0, high=num_clusters, size=num_masks)
        # 生成随机的分类
        for _ in range(max_num_iterations):
            depth_in_cluster = np.empty((num_clusters, size_row-size_window, size_col-size_window, num_masks), dtype=np.float32)
            for i in range (size_row-size_window):
                for j in range (size_col-size_window):
                    window = np.zeros((num_masks,size_window, size_window), dtype=np.float32)
                    for k in range (num_masks):
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


def kmeans_cluster_inclusion_matrix(masks, num_clusters, depth="eid", metric="depth", num_attempts=5, max_num_iterations=10, seed=42):
    assert(depth in ["eid", "id", "cbd"])
    assert(metric in ["depth", "red"])

    masks = np.array(masks, dtype=np.float32)
    num_masks = masks.shape[0]
    if depth == "eid" or depth == "ecbd":
        inclusion_matrix = compute_epsilon_inclusion_matrix(masks)
        np.fill_diagonal(inclusion_matrix, 1) # Required for feature parity with the O(N) version of eID.
    else:
        inclusion_matrix = compute_inclusion_matrix(masks)

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
        cluster_assignment = rng.integers(low=0, high=num_clusters, size=num_masks)
        print(cluster_assignment)
        for _ in range(max_num_iterations):
            depth_in_cluster = np.empty((num_clusters, num_masks), dtype=np.float32)
            for c in range(num_clusters):
                j_in_cluster = cluster_assignment == c
                
                N = np.sum(j_in_cluster)
                if depth == "cbd" or depth == "id" or depth == "eid":
                    N_a = np.sum(inclusion_matrix[:,j_in_cluster], axis=1)
                    N_b = np.sum(inclusion_matrix.T[:,j_in_cluster], axis=1)

                    if depth == "cbd":
                        # We need to normalize the depth such that it is  not dependent on the number of contours in the cluster.
                        # If the contour is already in the cluster then N_a and N_b range from 0 to N-1
                        # If the contour is *not* in the cluster then N_a and N_b range from 0 to N
                        N_ab_range = N - j_in_cluster
                        depth_in_cluster[c] = (N_a * N_b) / (N_ab_range * N_ab_range)
                    else: # ID / eID
                        depth_in_cluster[c] = np.minimum(N_a, N_b) / N
                #else: # eCBD (epsilon Cluster Band Depth)
                #    i_in_j = inclusion_matrix[:,j_in_cluster]
                #    j_in_i = inclusion_matrix.T[:,j_in_cluster]
                #    # We need to normalize the depth such that it is  not dependent on the number of contours in the cluster.
                #    depth_in_cluster[c] = np.minimum(i_in_j, j_in_i) / N
            if metric == "depth":
                metric_values = depth_in_cluster
            else: # Relative Depth (ReD)
                red = np.empty(depth_in_cluster.shape, dtype=np.float32)
                for c in range(num_clusters):
                    # Compute the max value exluding the current cluster.
                    # There is a more efficient, but slightly dirtier, solution.
                    depth_between = np.max(np.roll(depth_in_cluster, -c, axis=0)[1:,:], axis=0)
                    depth_within = depth_in_cluster[c,:]
                    red[c,:] = depth_within - depth_between
                metric_values = red
            print(metric_values)
            old_cluster_assignment = cluster_assignment
            cluster_assignment = np.argmax(metric_values, axis=0)
            print(cluster_assignment)
            if not check_valid_assignment(cluster_assignment, num_clusters) or np.all(cluster_assignment == old_cluster_assignment):
                print("best")
                break

            depth_sum = np.sum(np.choose(cluster_assignment, metric_values))
            if depth_sum > best_depth_sum:
                best_cluster_assignment = cluster_assignment
                best_depth_sum = depth_sum

    return best_cluster_assignment

def multiscale_kmeans_cluster_eid(masks, num_clusters, metric="depth", num_attempts=5, max_num_iterations=10,  size_window=63, seed=42):
    assert(metric in ["depth", "red"])

    masks = np.array(masks, dtype=np.float32)
    num_masks, height, width = masks.shape
    neg_masks = 1 - masks
    areas = np.sum(masks, axis=(1, 2))
    inv_areas = 1 / areas    

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)  
    
    best_depth_sum = -np.inf
    best_cluster_assignment = None
    for _ in range(num_attempts):
        print(_)
        cluster_assignment = rng.integers(low=0, high=num_clusters, size=num_masks)
        print(cluster_assignment)
        for _ in range(max_num_iterations):
            for i in range (height-size_window):
                for j in range (width-size_window):
                    for k in range (num_masks):
                        window = np.zeros((num_masks,size_window, size_window))
                        window[k] = masks[k,i:i+size_window,j:j+size_window]
                    np.set_printoptions(threshold=np.inf)
                    print("masks:",masks)
                    print("window",window)
                    precompute_in = np.empty((num_clusters, size_window, size_window), dtype=np.float32)
                    precompute_out = np.empty((num_clusters, size_window, size_window), dtype=np.float32)
                    neg_masks = 1 - window
                    areas = np.sum(window, axis=(1, 2))
                    for k in range(num_masks):
                        if areas[k] == 0:
                            inv_areas[k] = 0
                        else:
                            inv_areas[k] = 1 / areas[k]
                    for c in range(num_clusters):
                        j_in_cluster = cluster_assignment == c
                        selected_masks = window[j_in_cluster]
                        selected_areas = areas[j_in_cluster]
                        selected_inv_masks = neg_masks[j_in_cluster]
                        safe_selected_areas = np.where(selected_areas == 0, np.finfo(float).eps, selected_areas)
                        precompute_in[c] = np.sum(selected_inv_masks, axis=0)
                        precompute_out[c] = np.sum((selected_masks.T / safe_selected_areas.T).T, axis=0)

                    depth_in_cluster = np.empty((num_clusters, height-size_window, width-size_window, num_masks), dtype=np.float32)
                    empty_cluster = False
                    for c in range(num_clusters):
                        N = np.sum(cluster_assignment == c)
                        if N == 0:
                            empty_cluster = True
                            break
                        IN_in = np.empty(num_masks)
                        IN_out = np.empty(num_masks)
                        for k in range(num_masks):
                            if areas[k] == 0 :
                                IN_in[k] = 0
                                IN_out[k] = 0
                            else:
                                IN_in[k] = N - inv_areas[k] * np.sum(window * precompute_in[c], axis=(1,2))[k]
                                IN_out[k] = N - np.sum(neg_masks * precompute_out[c], axis=(1, 2))[k]
                        depth_in_cluster[c][i][j] = np.minimum(IN_in, IN_out) / N
            if empty_cluster:
                break
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

            print(metric_values)
            old_cluster_assignment = cluster_assignment
            cluster_assignment = np.argmax(metric_values, axis=0)
            print(cluster_assignment)
            if np.all(cluster_assignment == old_cluster_assignment):
                print("best")
                break
            depth_sum = np.sum(np.choose(cluster_assignment, metric_values))
            if depth_sum > best_depth_sum:
                best_cluster_assignment = cluster_assignment
                best_depth_sum = depth_sum

    return best_cluster_assignment

def kmeans_cluster_eid(masks, num_clusters, metric="depth", num_attempts=5, max_num_iterations=10, seed=42):
    assert(metric in ["depth", "red"])

    masks = np.array(masks, dtype=np.float32)
    num_masks, height, width = masks.shape
    neg_masks = 1 - masks
    areas = np.sum(masks, axis=(1, 2))
    inv_areas = 1 / areas    

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)  
    
    best_depth_sum = -np.inf
    best_cluster_assignment = None
    for _ in range(num_attempts):
        cluster_assignment = rng.integers(low=0, high=num_clusters, size=num_masks)
        for _ in range(max_num_iterations):
            precompute_in = np.empty((num_clusters, height, width), dtype=np.float32)
            precompute_out = np.empty((num_clusters, height, width), dtype=np.float32)
            
            for c in range(num_clusters):
                j_in_cluster = cluster_assignment == c
                selected_masks = masks[j_in_cluster]
                selected_areas = areas[j_in_cluster]
                selected_inv_masks = neg_masks[j_in_cluster]

                precompute_in[c] = np.sum(selected_inv_masks, axis=0)
                precompute_out[c] = np.sum((selected_masks.T / selected_areas.T).T, axis=0)

            depth_in_cluster = np.empty((num_clusters, num_masks), dtype=np.float32)
            empty_cluster = False
            for c in range(num_clusters):
                N = np.sum(cluster_assignment == c)
                if N == 0:
                    empty_cluster = True
                    break
                IN_in = N - inv_areas * np.sum(masks * precompute_in[c], axis=(1,2))
                IN_out = N - np.sum(neg_masks * precompute_out[c], axis=(1, 2))
                depth_in_cluster[c] = np.minimum(IN_in, IN_out) / N
            if empty_cluster:
                break

            if metric == "depth":
                metric_values = depth_in_cluster
            else: # Relative Depth (ReD)
                red = np.empty(depth_in_cluster.shape, dtype=np.float32)
                for c in range(num_clusters):
                    # Compute the max value exluding the current cluster.
                    # There is a more efficient, but slightly dirtier, solution.
                    depth_between = np.max(np.roll(depth_in_cluster, -c, axis=0)[1:,:], axis=0)
                    depth_within = depth_in_cluster[c,:]
                    red[c,:] = depth_within - depth_between
                metric_values = red

            old_cluster_assignment = cluster_assignment
            cluster_assignment = np.argmax(metric_values, axis=0)
            if np.all(cluster_assignment == old_cluster_assignment):
                break
            depth_sum = np.sum(np.choose(cluster_assignment, metric_values))
            if depth_sum > best_depth_sum:
                best_cluster_assignment = cluster_assignment
                best_depth_sum = depth_sum
    return best_cluster_assignment

#################################
# Competing cluster computation #
#################################

# A good competing cluster is one that
#  a) is nearby (for sil we want to decrease the within term -> increases compactness)
#  b) one that could potentially profit more from having the contour in terms of depth (for red we want to increase depth)
#  b1) alternatively, if we do not want to compute the depth for all cluster, we could calculate the medians and 
#      assess the inclusion relationship
# between depth is potentially more expensive to compute than between sil. 

def get_competing_clusters(clustering_ids, n_components, sil_contour_cluster_rels, red_contour_cluster_rels, inclusion_mat, depths, method="sil"):
    num_contours = clustering_ids.size
    if method=="sil":
        competing_clusters = np.argmin(sil_contour_cluster_rels, axis=1)  # we pick the most compact cluster after transferring the contour.
    elif method=="red":
        competing_clusters = np.argmax(red_contour_cluster_rels, axis=1)  # we pick the deepest cluster after transferring the contour.
    elif method=="inclusion_rel":
        inclusion_rels = np.empty((num_contours, n_components), dtype=float)
        for contour_id, cluster_id_1 in zip(np.arange(num_contours), clustering_ids):
            inclusion_rels[contour_id, cluster_id_1] = 0
            for cluster_id_2 in np.setdiff1d(np.unique(clustering_ids), cluster_id_1):
                cluster_ids = np.where(clustering_ids == cluster_id_2)[0]
                median_id = np.argmax(depths[cluster_ids])
                median_glob_id = cluster_ids[median_id]
                ls = inclusion_mat[contour_id, median_glob_id]
                rs = inclusion_mat[median_glob_id, contour_id]
                inclusion_rels[contour_id, cluster_id_2] = ls + rs  # TODO: add thresholding?
        competing_clusters = np.argmax(inclusion_rels, axis=1) # we only compare the contour against the medians and pick the one with which it has the strongest inclusion relationship
    return competing_clusters


def get_depth_competing_clusters(clustering_ids, n_components, depths, inclusion_mat):
    num_contours = clustering_ids.size
    inclusion_rels = np.empty((num_contours, n_components), dtype=float)
    for contour_id, cluster_id_1 in zip(np.arange(num_contours), clustering_ids):
        inclusion_rels[contour_id, cluster_id_1] = 0
        for cluster_id_2 in np.setdiff1d(np.unique(clustering_ids), cluster_id_1):
            cluster_ids = np.where(clustering_ids == cluster_id_2)[0]
            median_id = np.argmax(depths[cluster_ids])
            median_glob_id = cluster_ids[median_id]
            ls = inclusion_mat[contour_id, median_glob_id]
            rs = inclusion_mat[median_glob_id, contour_id]
            inclusion_rels[contour_id, cluster_id_2] = ls + rs  # TODO: add thresholding?
    competing_clusters = np.argmax(inclusion_rels, axis=1) # we only compare the contour against the medians and pick the one with which it has the strongest inclusion relationship
    return competing_clusters


###################
# Sil computation #
###################
# Silhouette width: sil(c_i) = (b_i - a_i)/max(a_i, b_i) 
#   with a_i = d(c_i|other members members of c_i's cluster) 
#   and  b_i = min_{other clusters besides the one c_i's in} d(c_i|other members in said cluster)

# first we compute sil_a
def compute_sil_within(contours_mat, clustering, n_components):
    num_contours = contours_mat.shape[0]
    clustering_ids = np.arange(n_components)
    sil_a = np.zeros(num_contours)
    for cluster_id in clustering_ids:
        contour_ids = np.where(clustering == cluster_id)[0]
        dmat = cdist(contours_mat[contour_ids, :], contours_mat[contour_ids, :], metric="sqeuclidean")
        mean_dists = dmat.mean(axis=1)
        sil_a[contour_ids] = mean_dists
    return sil_a


# we need instance to cluster matrix (N, C)

# then we compute sil_b
def compute_sil_between(contours_mat, clustering, n_components):
    num_contours = contours_mat.shape[0]
    clustering_ids = np.arange(n_components)
    contour_cluster_rels = np.empty((num_contours, n_components), dtype=float)
    for cluster_id1 in clustering_ids:
        contour_ids_1 = np.where(clustering == cluster_id1)[0]
        for contour_id in contour_ids_1:
            contour_cluster_rels[contour_id, cluster_id1] = np.inf
            for cluster_id2 in np.setdiff1d(clustering_ids, cluster_id1):
                contour_ids_2 = np.where(clustering == cluster_id2)[0]
                # compute distance from contour_id to all contours in contour_ids_2 (outputs a vector)
                dmat = cdist(contours_mat[contour_id, :].reshape(1, -1), contours_mat[contour_ids_2, :], metric="sqeuclidean")
                contour_cluster_rels[contour_id, cluster_id2] = dmat.mean()
    competing_cluster_ids = np.argmin(contour_cluster_rels, axis=1)
    sil_b = contour_cluster_rels[np.arange(num_contours), competing_cluster_ids]
    return sil_b, contour_cluster_rels

def compute_sil(contours_mat, clustering, n_components):
    sil_a = compute_sil_within(contours_mat, clustering, n_components)
    sil_b, contour_cluster_rels = compute_sil_between(contours_mat, clustering, n_components)
    sil_i = (sil_b - sil_a)/np.maximum(sil_a, sil_b)
    return sil_i, sil_a, sil_b, contour_cluster_rels


#####################
# Depth computation #
#####################
# Relative depth of a point ReD(c_i): D^w(c_i) - D^b(c_i)
#   with D^w(c_i) = ID(c_i|other members of c_i's cluster)
#   and  D^b(c_i) = min_{other clusters besides the one c_i's in} ID(c_i|other members in said cluster)

# first we compute d_w
# for ID, depth is not defined for clusters of size N>1 so we return 0
# for CBD, depth is not defined for clusters of size N>2 so we return 0
def compute_red_within(masks, clustering, n_components, 
                       depth_notion="id", use_modified=True, use_fast=True, 
                       inclusion_mat=None, precompute_ins=None, precompute_outs=None):
    num_contours = len(masks)
    clustering_ids = np.arange(n_components)
    depth_w = np.zeros(num_contours)
    for cluster_id in clustering_ids:
        contour_ids = np.where(clustering == cluster_id)[0]
        mask_subset = [masks[i] for i in contour_ids]
        if (depth_notion == "cbd" and contour_ids.size <= 2) or (depth_notion == "id" and contour_ids.size <= 1):                
            depths = np.zeros_like(contour_ids.size)
        else:
            if use_modified and use_fast:
                assert depth_notion == "id"  # only supported for depth_notion == "id"
                precompute_in = precompute_ins[cluster_id].copy() if precompute_ins is not None and cluster_id in precompute_ins else None
                precompute_out = precompute_outs[cluster_id].copy() if precompute_outs is not None and cluster_id in precompute_outs else None
                depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, precompute_in=precompute_in, precompute_out=precompute_out)
            else:
                inclusion_mat_subset = None
                if inclusion_mat is not None:
                    inclusion_mat_subset = inclusion_mat[np.ix_(contour_ids, contour_ids)]
                    
                if depth_notion == "id":
                    depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
                elif depth_notion == "cbd":
                    depths = band_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
                else:
                    raise ValueError("Unsupported depth notion (only id and cbd supported)")
        depth_w[contour_ids] = depths
    return depth_w

# then we compute d_b
def compute_red_between(masks, clustering, n_components, competing_clusters=None, 
                        depth_notion="id", use_modified=True, use_fast=True, 
                        inclusion_mat=None, precompute_ins=None, precompute_outs=None):
    # if you pass competing clusters, then the same are outputted
    # if you dont pass competing clusters then clusters that maximize depth are outputted
    num_contours = len(masks)
    clustering_ids = np.arange(n_components)
    depth_b_cluster = np.empty((num_contours, n_components), dtype=float)    
    #depth_delta_cluster = np.empty((num_contours, n_components), dtype=float)    
    for cluster_id1 in clustering_ids:
        contour_ids_1 = np.where(clustering == cluster_id1)[0]
        for contour_id in contour_ids_1:
            depth_b_cluster[contour_id, cluster_id1] = -np.inf
            #depth_delta_cluster[contour_id, cluster_id1] = -np.inf
            if competing_clusters is not None:  # we just want the depth of the competing cluster
                competing_cids = [competing_clusters[contour_id], ]
                other_cids = np.setdiff1d(np.setdiff1d(cluster_id1, cluster_id1), competing_clusters[contour_id])
                for ocid in other_cids:
                    depth_b_cluster[contour_id, ocid] = -np.inf
                    #depth_delta_cluster[contour_id, ocid] = -np.inf
            else:
                competing_cids = np.setdiff1d(clustering_ids, np.array([cluster_id1])) # all other clusters
            
            for competing_cid in competing_cids:
                contour_ids_2 = np.where(clustering == competing_cid)[0].tolist()
                contour_ids_2.append(contour_id)
                mask_subset = [masks[i] for i in contour_ids_2]
                if (depth_notion == "cbd" and len(contour_ids_2) <= 2) or (depth_notion == "id" and len(contour_ids_2) <= 1):                
                    depths = np.zeros(len(contour_ids_2))
                else:
                    if use_modified and use_fast:
                        assert depth_notion == "id"  # only supported for depth_notion == "id"
                        precompute_in = precompute_ins[competing_cid].copy() if precompute_ins is not None and competing_cid in precompute_ins else None
                        precompute_out = precompute_outs[competing_cid].copy() if precompute_outs is not None and competing_cid in precompute_outs else None
                        if precompute_in is not None:
                            precompute_in += 1 - masks[contour_id]
                        if precompute_out is not None:
                            precompute_out += masks[contour_id]/masks[contour_id].sum()
                        dval = inclusion_depth_modified_fast(masks[contour_id], mask_subset, precompute_in, precompute_out)
                        depths = [dval,]
                        #depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, precompute_in=precompute_in, precompute_out=precompute_out) 
                    else:
                        inclusion_mat_subset = None
                        if inclusion_mat is not None:
                            inclusion_mat_subset = inclusion_mat[np.ix_(contour_ids_2, contour_ids_2)]
                            
                        if depth_notion == "id":
                            depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
                        elif depth_notion == "cbd":
                            depths = band_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
                        else:
                            raise ValueError("Unsupported depth notion (only id and cbd supported)")
                    
                depth_b_cluster[contour_id, competing_cid] = depths[-1]  # (from ddclust paper): we only want the depth of the last contour we appended
                #depth_delta_cluster[contour_id, competing_cid] = depths.mean()  # we also want to keep track to the effect of adding a contour to a cluster
    
    if competing_clusters is None:
        competing_clusters = np.argmax(depth_b_cluster, axis=1) # the competing cluster is the one with the highest depth
        # competing_clusters = np.argmax(depth_delta_cluster, axis=1) # the competing cluster is the one with the highest depth
        # competing_clusters = np.argmax(depth_b_cluster - depth_delta_cluster, axis=1) # the competing cluster is the one with the highest depth
        # from scipy.stats import rankdata
        # ranks = rankdata(depth_b_cluster, method='min', axis=1) - 1
        # ranks = ranks[:, 1:][:, ::-1]  # remove -np.inf column and flip so it is descending
        # is_tie = []
        # for r in ranks:
        #     if np.where(r == r[0])[0].size > 1:
        #         is_tie.append(True)
        #     else:
        #         is_tie.append(False)
        # print(np.any(is_tie == True))

    depth_b = np.array([depth_b_cluster[i, competing_cid] for i, competing_cid in enumerate(competing_clusters)]) # the competing cluster is the one with the highest depth

    return depth_b, competing_clusters

def compute_red(masks, clustering, n_components, competing_clusters=None, depth_notion="id", use_modified=True, use_fast=True, inclusion_mat=None):
    red_w = compute_red_within(masks, clustering, n_components, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, inclusion_mat=inclusion_mat)
    red_b, competing_clusters = compute_red_between(masks, clustering, n_components, competing_clusters, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, inclusion_mat=inclusion_mat)
    red_i = red_w - red_b
    return red_i, red_w, red_b, competing_clusters


########
# Cost #
########

def compute_cost(sils, reds, weight=0.5):
    return (1 - weight) * sils + weight * reds
