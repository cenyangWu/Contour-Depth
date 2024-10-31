from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.synthetic_data import main_shape_with_outliers
from src.clustering.cdclust import largest_depth
from skimage.measure import find_contours





def find_boundary(arr):
    rows, cols = arr.shape
    boundary = np.zeros_like(arr)
    
    for i in range(rows):
        for j in range(cols):
            if arr[i, j] == 1:
                if (i > 0 and arr[i - 1, j] == 0) or \
                   (i < rows - 1 and arr[i + 1, j] == 0) or \
                   (j > 0 and arr[i, j - 1] == 0) or \
                   (j < cols - 1 and arr[i, j + 1] == 0):
                    boundary[i, j] = 1
    return boundary

def process_3d_array(arr_3d):
    # 获取三维数组的尺寸
    num_arrays, rows, cols = arr_3d.shape
    
    # 创建一个同样尺寸的三维数组用于存储结果
    boundaries_3d = np.zeros_like(arr_3d)
    
    # 遍历每个二维数组
    for k in range(num_arrays):
        boundaries_3d[k] = find_boundary(arr_3d[k])
    
    return boundaries_3d

def extract_contours(masks):
        contours = []
        for mask in masks:
            # find_contours 返回等高线为列表，假设mask是二维的且为二进制图像
            contour = find_contours(mask, level=0.5)  # level 0.5适合二进制图像
            if contour:  # 仅添加非空轮廓
                contours.append(contour[0])  # 只取第一个轮廓，假设每个掩模只有一个主要轮廓
        return contours
def plot_spaghetti(masks, best_id):
        contours = extract_contours(masks)
        fig, ax = plt.subplots()
        for i, contour in enumerate(contours):
            if i == best_id:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')  # 用红色显示best_id的轮廓
            else:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='black')  # 其他用黑色

        ax.invert_yaxis()  # Y轴反转以符合图像坐标
        ax.set_title('Spaghetti Plot of Mask Contours')
        plt.show()

if __name__ == "__main__":

    outputs_dir = Path("outputs/outlier_separation")
    assert outputs_dir.exists()

    SEED_DATA = 0
    SEED_CLUSTER = 65
    ROWS = COLS = 64
    K = 2

    masks, labs = main_shape_with_outliers(100, ROWS, COLS, return_labels=True, seed=SEED_DATA)
    labs = np.array(labs)
    labs = 1 - labs
    # print(masks)
    masksarray = np.array(masks, dtype=np.float32)
    print(masksarray.shape)
    deepest, best_id = largest_depth(masks,depth="eid", metric="depth")
    boundaries_3d = process_3d_array(masksarray)
    # 输出结果的一个示例二维数组
    np.set_printoptions(threshold=np.inf)
    plot_spaghetti(masks, best_id)