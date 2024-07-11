import numpy as np

def compute_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous. 

    Parameters
    ----------
    masks : list
        list of ndarrays corresponding to an ensemble of binary masks.
    """
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    for i in range(num_masks):
        inclusion_mat[i, :] = np.all((masks & masks[i]) == masks[i], axis=(1, 2))
        inclusion_mat[i, i] = 0
    return inclusion_mat


def compute_epsilon_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous. 

    Parameters
    ----------
    masks : list
        list of ndarrays corresponding to an ensemble of binary masks.
    """
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    inv_masks = 1 - masks
    area = np.sum(masks, axis=(1, 2))
    
    for i in range(num_masks):
        if area[i] > 0:  # Check to prevent division by zero
            inclusion_scores = 1 - np.sum(inv_masks & masks[i], axis=(1, 2)) / area[i]
        else:
            inclusion_scores = np.zeros(num_masks)  # If mask area is 0, set all relations to 0
            
        inclusion_mat[i, :] = inclusion_scores
        inclusion_mat[i, i] = 0  # Set diagonal to 0 as a mask cannot include itself

    return inclusion_mat
