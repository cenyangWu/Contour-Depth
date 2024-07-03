""" Implementation of Inclusion Depth for contour ensembles.
"""
from time import time
import numpy as np
from skimage.measure import find_contours

from src.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix
from src.utils import get_sdfs, get_masks_matrix


def compute_depths(data,                   
                   modified=False,
                   fast=False,
                   inclusion_mat=None,
                   precompute_in=None,
                   precompute_out=None,
                   verbose=False
                   ):
    """Calculate depth of a list of contours using the inclusion depth (ID) method.

    Parameters
    ----------
    data : list
        List of contours to calculate the ID from. A contour is assumed to be
        represented as a binary mask with ones denoting inside and zeros outside
        regions.
    modified : bool, optional
        Whether to use or not the epsilon ID (eID). This reduces the sensitivity of
        the method to outliers but yields more informative depth estimates when curves
        cross over a lot, by default False.
    fast : bool, optional
        Whether to use the fast implementation, by default False.
    inclusion_mat : ndarray, optional
        Square (N x N) numpy array with the inclusion relationships between 
        contours, by default None.
    verbose: bool
        Whether to print status messages to the console.

    Returns
    -------
    ndarray
        Depths of the N contours in data.
    """
    
    num_contours = len(data)

    if not modified and fast:
        raise NotImplementedError("ID not available for modified=False and fast=True")

    # Precomputed masks for modified versions
    if modified and fast:
        if precompute_in is None:
            precompute_in = get_precompute_in(data)
        if precompute_out is None:
            precompute_out = get_precompute_out(data)

    if inclusion_mat is None and not fast:
        if verbose:
            print("Warning: pre-computed inclusion matrix not available, computing it ... ")
        if modified:            
            inclusion_mat = compute_epsilon_inclusion_matrix(data)
        else:
            inclusion_mat = compute_inclusion_matrix(data)

    depths = []
    for i in range(num_contours):
        if modified:
            if fast:
                depth = inclusion_depth_modified_fast(data[i], data, precompute_in=precompute_in, precompute_out=precompute_out)
            else:
                depth = inclusion_depth_modified(i, inclusion_mat)
        else:
            depth = inclusion_depth_strict(i, inclusion_mat)

        depths.append(depth)

    return np.array(depths, dtype=float)


def inclusion_depth_strict(mask_index, inclusion_mat):
    num_masks = inclusion_mat[mask_index].size
    in_count = (inclusion_mat[mask_index, :] > 0).sum()
    out_count = (inclusion_mat[:, mask_index] > 0).sum()

    return np.minimum(in_count/num_masks, out_count/num_masks)


def inclusion_depth_strict_fast(data):
    num_contours = len(data)
    data_sdfs = get_sdfs(data)
    
    R = get_masks_matrix(data_sdfs)
    R_p = np.argsort(R, axis=0)
    R_pp = np.argsort(R_p, axis=0) + 1

    # for p in range(R.shape[1]):
    #     for i in range(1, R.shape[0]):
    #         if R[R_p[i, p], p] <= R[R_p[i-1, p], p]: 
    #             R_pp[R_p[i, p], p] = R_pp[R_p[i-1, p], p]

    n_b = np.min(R_pp, axis=1) - 1
    n_a = num_contours - np.max(R_pp, axis=1)

    depths_fast_sdf = np.array([n_b, n_a])/num_contours
    depths_fast_sdf = np.min(depths_fast_sdf, axis=0)
    
    return depths_fast_sdf.tolist()


def inclusion_depth_modified(mask_index, inclusion_mat):
    in_vals = inclusion_mat[mask_index, :]
    out_vals = inclusion_mat[:, mask_index]

    return np.minimum(np.mean(in_vals), np.mean(out_vals))


def inclusion_depth_modified_fast(in_ci, masks, precompute_in=None, precompute_out=None):
    num_masks = len(masks)
    if precompute_in is None:
        precompute_in = get_precompute_in(masks)
    if precompute_out is None:
        precompute_out = get_precompute_out(masks)

    IN_in = num_masks - ((in_ci / in_ci.sum()) * precompute_in).sum()
    IN_out = num_masks - ((1-in_ci) * precompute_out).sum()

    # We remove from the count in_ci, which we do not consider as it adds to both IN_in and IN_out equally
    return np.minimum((IN_in - 1)/num_masks, (IN_out - 1)/num_masks)


def get_precompute_in(masks):
    precompute_in = np.zeros_like(masks[0])
    for i in range(len(masks)):
        precompute_in += 1 - masks[i]
    return precompute_in

def get_precompute_out(masks):
    precompute_out = np.zeros_like(masks[0])
    for i in range(len(masks)):
        precompute_out += masks[i]/masks[i].sum()
    return precompute_out
