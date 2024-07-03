""" Contour variability plots
"""

# Get masks 
# Convert masks to sdfs
# Get PC of sdf matrix (tau=0.999)
# Run AHC with average linking on PC mat (euclidean distance)
# Determine number of clusters using L-method

# Obtain median contour: 
#    artificial, geometric median of PC in cluster and extract 0 contour
#    line width encodes relative cluster size
#    add a parplot on the side which shows the proportions

# Bands: alpha units of standard deviation (alpha=1)
#    visualized as transparent 2D polygons

# Outliers: clusters of cardinality one: depicted as lines

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from src.utils import get_sdfs, get_masks_matrix

import numpy as np
from scipy.spatial.distance import cdist, euclidean

def geometric_median(X, eps=1e-5):
    # https://stackoverflow.com/a/30305181
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def get_cvp_sdf_pca_transform(masks, threshold_explained_var=0.999, seed=None):
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    sdfs = get_sdfs(masks)
    sdf_mat = get_masks_matrix(sdfs)

    pca_embedder = PCA(random_state=seed)
    pca_embedder.fit(sdf_mat)
    exp_var = pca_embedder.explained_variance_ratio_
    cut_off = np.where((np.cumsum(exp_var)<=0.999) == False)[0][0] + 1
    transform_mat = pca_embedder.components_[:cut_off, :]    
    pca_mat = np.matmul(sdf_mat - sdf_mat.mean(axis=0), transform_mat.T)

    return sdf_mat, pca_mat, transform_mat


# 1. Compute SDF of contours
# 2. Compute PCA
# 3. Perform clustering using AHC
# 4. Compute bands and medians


def get_cvp_clustering(pca_mat, num_components):
    
    ahc = AgglomerativeClustering(n_clusters=num_components, 
                                  metric="euclidean", linkage="average").fit(pca_mat)
    labs = ahc.labels_

    return labs


def get_cvp_pca_medians(pca_mat, labs):
    medians = []
    for k in np.unique(labs):
        pca_cluster_mat = pca_mat[np.where(labs == k)]
        pca_median = geometric_median(pca_cluster_mat)
        medians.append(pca_median)
    return medians


def transform_from_pca_to_sdf(pca_vectors, mean_sdf_vector, transform_mat):
    # https://stackoverflow.com/questions/32750915/pca-inverse-transform-manually
    return mean_sdf_vector + np.matmul(pca_vectors, transform_mat)


def get_per_cluster_mean(mat, labs):
    means = []
    for k in np.unique(labs):
        cluster_mat = mat[np.where(labs == k)]
        mean = np.mean(cluster_mat, axis=0)
        means.append(mean)
    return means

def get_cvp_bands(sdf_mat, labs, std_mult=1.5):
    bands = []
    for k in np.unique(labs):
        sdf_cluster_mat = sdf_mat[np.where(labs == k)]
        sdf_mean = np.mean(sdf_cluster_mat, axis=0)
        sdf_var = np.std(sdf_cluster_mat, axis=0)
        band_bounds = np.array([- (sdf_mean - std_mult * sdf_var), 
                         sdf_mean + std_mult * sdf_var])
        band = np.prod((band_bounds > 0).astype(float), axis=0)
        #band = band_bounds.min(axis=0)  # limits are in 0-level        
        bands.append(band)
    return bands


def get_cvp_bands_parts(sdf_mat, labs, std_mult=1.5):
    bands = []
    for k in np.unique(labs):
        sdf_cluster_mat = sdf_mat[np.where(labs == k)]
        sdf_mean = np.mean(sdf_cluster_mat, axis=0)
        sdf_var = np.std(sdf_cluster_mat, axis=0)
        band_bounds = np.array([- (sdf_mean - std_mult * sdf_var), 
                         sdf_mean + std_mult * sdf_var]) 
        #band = band_bounds.min(axis=0)  # limits are in 0-level        
        bands.append(band_bounds)
    return bands

