import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from skimage.draw import ellipse, polygon2mask
from skimage.measure import find_contours
from sklearn.decomposition import KernelPCA
from sklearn.cluster import AgglomerativeClustering

# Define necessary functions
def circle_ensemble(num_masks, num_rows, num_cols, center_mean=(0.5, 0.5), center_std=(0, 0),
                    radius_mean=0.25, radius_std=0.25 * 0.1, seed=None):

    rng = np.random.default_rng(seed)
    RADIUS_MEAN = np.minimum(num_rows, num_cols) * radius_mean
    RADIUS_STD = np.minimum(num_rows, num_cols) * radius_std
    radii = rng.normal(RADIUS_MEAN, RADIUS_STD, num_masks)
    centers_rows = rng.normal(num_rows * center_mean[0], num_rows * center_std[0], num_masks)
    centers_cols = rng.normal(num_cols * center_mean[1], num_cols * center_std[1], num_masks)

    masks = []
    for i in range(num_masks):
        mask = np.zeros((num_rows, num_cols))
        rr, cc = ellipse(centers_rows[i], centers_cols[i], radii[i], radii[i], shape=(num_rows, num_cols))
        mask[rr.astype(int), cc.astype(int)] = 1
        masks.append(mask)

    return masks

def get_base_gp(num_masks, domain_points, scale=0.01, sigma=1.0, seed=None):
    rng = np.random.default_rng(seed)
    thetas = domain_points.flatten().reshape(-1, 1)
    num_vertices = thetas.size
    gp_mean = np.zeros(num_vertices)

    gp_cov_sin = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.sin(thetas), np.sin(thetas), "sqeuclidean"))
    gp_sample_sin = rng.multivariate_normal(gp_mean, gp_cov_sin, num_masks)
    gp_cov_cos = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.cos(thetas), np.cos(thetas), "sqeuclidean"))
    gp_sample_cos = rng.multivariate_normal(gp_mean, gp_cov_cos, num_masks)

    return gp_sample_sin + gp_sample_cos

def get_xy_coords(angles, radii):
    num_members = radii.shape[0]
    angles = angles.flatten().reshape(1, -1)
    angles = np.repeat(angles, num_members, axis=0)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y

def rasterize_coords(x_coords, y_coords, num_rows, num_cols):
    masks = []
    for xc, yc in zip(x_coords, y_coords):
        coords_arr = np.vstack((xc, yc)).T
        coords_arr *= num_rows // 2
        coords_arr += num_cols // 2
        mask = polygon2mask((num_rows, num_cols), coords_arr).astype(float)
        masks.append(mask)
    return masks

def main_shape_with_outliers(num_masks, num_rows, num_cols, num_vertices=100,
                             population_radius=0.5, normal_scale=0.0003, normal_freq=0.9,
                             outlier_scale=0.005, outlier_freq=0.04,
                             p_contamination=0.5, return_labels=False, seed=None):

    rng = np.random.default_rng(seed)
    thetas = np.linspace(0, 2 * np.pi, num_vertices)
    population_radius = np.ones_like(thetas) * population_radius

    gp_sample_normal = get_base_gp(num_masks, thetas, scale=normal_scale, sigma=normal_freq, seed=seed)
    gp_sample_outliers = get_base_gp(num_masks, thetas, scale=outlier_scale, sigma=outlier_freq, seed=seed)+0.1

    should_contaminate = rng.random(num_masks) < p_contamination
    should_contaminate = should_contaminate.reshape(-1, 1)
    should_contaminate = np.repeat(should_contaminate, len(thetas), axis=1)

    radii = population_radius + gp_sample_normal * (~should_contaminate) + gp_sample_outliers * should_contaminate

    xs, ys = get_xy_coords(thetas, radii)
    contours = rasterize_coords(xs, ys, num_rows, num_cols)
    labels = should_contaminate[:, 0].astype(int)

    if return_labels:
        return contours, labels
    else:
        return contours

def compute_sdf(binary_mask):
    inside_dist = distance_transform_edt(binary_mask)
    outside_dist = distance_transform_edt(1 - binary_mask)
    sdf = inside_dist - outside_dist
    return sdf

def extract_contours(masks):
    contours = []
    for mask in masks:
        contour = find_contours(mask, level=0.5)
        if contour:
            contours.append(contour[0])  # Take the first contour
        else:
            contours.append(np.array([]))
    return contours

# Generate contours and true labels
num_samples = 50
num_rows = num_cols = 100
size_window = 50
contours_masks, true_labels = main_shape_with_outliers(num_samples, num_rows, num_cols, return_labels=True, seed=66)

window = np.zeros((num_samples, size_window, size_window), dtype=np.float32)
i = 0
j = 0
for k in range(num_samples):
    window[k] = contours_masks[k][i:i+size_window, j:j+size_window]

# Extract contours from window
contours = extract_contours(window)

# Number of points to sample along each contour
N = 50
sampled_contours = []

for contour in contours:
    if contour.size == 0:
        # If contour is empty, use zeros
        sampled_points = np.zeros((N, 2))
    else:
        # Compute cumulative arc length along the contour
        deltas = np.diff(contour, axis=0)
        segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
        total_length = cumulative_lengths[-1]
        if total_length == 0:
            # Contour is a single point, repeat it N times
            sampled_points = np.repeat(contour[0][np.newaxis, :], N, axis=0)
        else:
            # Normalize cumulative lengths to [0,1]
            normalized_lengths = cumulative_lengths / total_length
            # Sample N equally spaced points along the normalized arc length
            sample_points = np.linspace(0, 1, N)
            # Interpolate x and y coordinates
            interp_func_x = interp1d(normalized_lengths, contour[:, 1], kind='linear')
            interp_func_y = interp1d(normalized_lengths, contour[:, 0], kind='linear')
            sampled_x = interp_func_x(sample_points)
            sampled_y = interp_func_y(sample_points)
            sampled_points = np.vstack((sampled_x, sampled_y)).T  # Shape (N, 2)
    sampled_contours.append(sampled_points)

# Now, flatten the sampled points into vectors
flattened_contours = [points.flatten() for points in sampled_contours]
flattened_array = np.array(flattened_contours)

# Perform Kernel PCA
kpca = KernelPCA(n_components=8, kernel='rbf', gamma=1e-1)

contours_kpca = kpca.fit_transform(flattened_array)
print(contours_kpca.shape)
# Clustering using Agglomerative Hierarchical Clustering (AHC)
# Initialize AHC with 2 clusters
ahc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster_labels = ahc.fit_predict(contours_kpca)

# 创建一个包含两个子图的图形
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# 第一个子图：Clusters in Kernel PCA Space
scatter1 = ax[0].scatter(contours_kpca[:, 0], contours_kpca[:, 1], c=cluster_labels, cmap='viridis')
ax[0].set_xlabel('Kernel PCA Component 1')
ax[0].set_ylabel('Kernel PCA Component 2')
ax[0].set_title('Clusters in Kernel PCA Space')
cbar1 = fig.colorbar(scatter1, ax=ax[0])
cbar1.set_label('Cluster Label')

# 第二个子图：True Labels in Kernel PCA Space
scatter2 = ax[1].scatter(contours_kpca[:, 0], contours_kpca[:, 1], c=true_labels, cmap='coolwarm')
ax[1].set_xlabel('Kernel PCA Component 1')
ax[1].set_ylabel('Kernel PCA Component 2')
ax[1].set_title('True Labels in Kernel PCA Space')
cbar2 = fig.colorbar(scatter2, ax=ax[1])
cbar2.set_label('True Label')

# 显示两个子图
plt.tight_layout()
plt.show()

# Plot all contours in one figure, colored by cluster labels
plt.figure(figsize=(8, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, sampled_points in enumerate(sampled_contours):
    cluster = cluster_labels[i]
    plt.plot(sampled_points[:, 0], sampled_points[:, 1], color=colors[cluster % len(colors)], linewidth=1)
plt.gca().invert_yaxis()
plt.title('All Contours Colored by Cluster Labels')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Optionally, display the original contours with cluster labels
fig, axes = plt.subplots(5, 10, figsize=(20, 10))
axes = axes.ravel()
for i in range(num_samples):
    axes[i].imshow(contours_masks[i], cmap='gray')
    axes[i].set_title(f'Cluster {cluster_labels[i]}')
    axes[i].axis('off')
plt.tight_layout()
plt.show()
