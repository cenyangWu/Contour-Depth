"""General utilities. 
"""

def get_masks_matrix(masks):
    import numpy as np
    mat = np.concatenate([mask.flatten()[np.newaxis] for mask in masks], axis=0)
    return mat

def get_sdfs(masks):
    from scipy.ndimage import distance_transform_edt
    sdfs = [distance_transform_edt(mask) + -1 * distance_transform_edt(1 - mask) for mask in masks]
    return sdfs