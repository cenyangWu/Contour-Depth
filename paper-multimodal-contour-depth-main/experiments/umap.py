import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from skimage.measure import find_contours
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import sys
import os

# 添加项目目录到路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入所需的模块和函数
from src.data.synthetic_data import main_shape_with_outliers
from src.clustering.cdclust import kmeans_cluster_inclusion_matrix, multiscale_kmeans_cluster_inclusion_matrix, multiscale_kmeans_cluster_eid, largest_depth
from src.clustering.inits import initial_clustering
from src.visualization import spaghetti_plot
from src.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering
import umap  # 修改导入方式

# 生成带有异常值的轮廓
num_masks = 3  # 生成多个轮廓
num_rows, num_cols = 100, 100
contours = main_shape_with_outliers(num_masks, num_rows, num_cols, num_vertices=100,
                                    population_radius=0.5,
                                    normal_scale=0.003, normal_freq=0.09,
                                    outlier_scale=0.009, outlier_freq=0.04,
                                    p_contamination=0.1, return_labels=False, seed=26)

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 定义颜色列表，以便区分不同的轮廓
colors = ['r', 'g', 'b']

# 对每个轮廓进行处理
for i in range(num_masks):
    # 提取每个轮廓的点
    contour = contours[i]
    contour_points = find_contours(contour, level=0.5)[0]
    x_points, y_points = contour_points[:, 1], contour_points[:, 0]

    # 参数化
    tck, u = splprep([x_points, y_points], s=0, per=True)
    new_t = np.linspace(0, 1, 1000)
    x_interp, y_interp = splev(new_t, tck)

    # 计算原始曲线的导数
    dx_interp, dy_interp = splev(new_t, tck, der=1)

    # 构建四维数据
    data_4d = np.column_stack((x_interp, y_interp, dx_interp, dy_interp))

    # 使用UMAP将四维数据降维到一维
    umap_model = umap.UMAP(n_components=1, random_state=42)
    u = umap_model.fit_transform(data_4d).flatten()

    # 在同一张图上绘制new_t与降维后的一维变量u的关系图
    plt.plot(new_t, u, color=colors[i % len(colors)], label=f'Contour {i+1}')

# 设置标签和标题
plt.xlabel('new_t')
plt.ylabel('u')
plt.title('UMAP Reduction of 4D Data for Multiple Contours')
plt.legend()
plt.show()
