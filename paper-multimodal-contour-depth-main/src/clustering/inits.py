"""Functions for initializing clustering methods.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from src.utils import get_sdfs, get_masks_matrix


def initial_clustering(masks, num_components, feat_mat=None, pre_pca=False, method="kmeans", k_means_n_init=5, k_means_max_iter=10, seed=None):
    """Generates initial clustering of the input masks with the specified number of components.
    Supports several methods.

    Parameters
    ----------
    masks : list
        List with the binary masks of the ensemble.
    num_components : int
        Number of components (K).
    feat_mat : ndarray, optional
        Feature matrix of shape (N, M) used by the kmeans and ahc initializations to compute distances between ensemble members.
    pre_pca : bool, optional
        Pre computed PCA, by default False
    method : str, optional
        Method to use for initialization, by default "kmeans". "random" produces a random labeling with the provided seed.
    k_means_n_init : int, optional
        Number of initializations for kmeans, by default 5
    k_means_max_iter : int, optional
        Max number of iterations for kmeans, by default 10
    seed : _int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    list
        Labels
    """
    num_masks = len(masks)

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    if feat_mat is None:
        print("[initial_clustering] Warning: feat_mat is None, using sdfs of the masks as feature matrix ...")
        sdfs = get_sdfs(masks)
        feat_mat = get_masks_matrix(sdfs)
    mat = feat_mat

    if pre_pca:
        pca_embedder = PCA(n_components=30)
        mat = pca_embedder.fit_transform(mat)
    
    if method == "random":
        labs = rng.integers(0, num_components, num_masks)
    elif method == "kmeans":
        labs = KMeans(n_clusters=num_components, n_init=k_means_n_init, max_iter=k_means_max_iter).fit_predict(mat)
    elif method == "ahc":
        ahc = AgglomerativeClustering(n_clusters=num_components, metric="euclidean", linkage="average").fit(mat)
        labs = ahc.labels_
    else:
        raise ValueError("Only kmeans and ahc supported for now.")

    return labs