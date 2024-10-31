import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from skimage.measure import find_contours
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.synthetic_data import main_shape_with_outliers
from src.clustering.cdclust import kmeans_cluster_inclusion_matrix, multiscale_kmeans_cluster_inclusion_matrix,  multiscale_kmeans_cluster_eid, largest_depth
from src.clustering.inits import initial_clustering
from src.visualization import spaghetti_plot
from src.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering
from skimage.measure import find_contours


# 生成带有异常值的轮廓
num_masks = 3  # 生成多个轮廓
num_rows, num_cols = 100, 100
contours = main_shape_with_outliers(num_masks, num_rows, num_cols, num_vertices=100, 
                             population_radius=0.5,
                             normal_scale=0.003, normal_freq=0.09,
                             outlier_scale=0.009, outlier_freq=0.04,
                             p_contamination=0.1, return_labels=False, seed=26)

plt.figure(figsize=(10, 6))

# 遍历所有轮廓并进行处理
for i in range(num_masks):
    # 提取每个轮廓的点
    contour = contours[i]
    contour_points = find_contours(contour, level=0.5)[0]
    x_points, y_points = contour_points[:, 1], contour_points[:, 0]

    # 参数化
    tck, u = splprep([x_points, y_points], s=0)
    new_t = np.linspace(0, 1, 1000)
    x_interp, y_interp = splev(new_t, tck)

    # 计算原始曲线的导数
    dx_interp, dy_interp = splev(new_t, tck, der=1)

    # 计算局部弧长 (ds = sqrt((dx/du)^2 + (dy/du)^2))
    ds = np.sqrt(dx_interp**2 + dy_interp**2)

    # 使用颜色映射变化率，并为每个轮廓生成单独的散点图
    sc = plt.scatter(x_interp, y_interp, c=new_t, cmap='plasma', s=10, label=f'Contour {i+1}')

# 添加颜色条和图例
plt.colorbar(sc, label='Rate of Change')

# 设置图例和标题
plt.title('Rate of Change of Multiple Parametric Curves')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()