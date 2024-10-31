import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep

# 计算原始曲线的导数
dx_interp, dy_interp = splev(new_t, tck, der=1)

# 计算局部弧长 (ds = sqrt((dx/du)^2 + (dy/du)^2))
ds = np.sqrt(dx_interp**2 + dy_interp**2)

# 计算累计弧长 (积分)
arc_length = np.cumsum(ds)
arc_length = np.insert(arc_length, 0, 0)  # 在起点插入 0，表示起始弧长为 0
total_arc_length = arc_length[-1]  # 总弧长

# 重新参数化，使得弧长均匀分布
uniform_arc_t = np.linspace(0, total_arc_length, len(new_t))

# 使用新参数进行插值，生成新的曲线
x_arc_interp = np.interp(uniform_arc_t, arc_length, x_interp)
y_arc_interp = np.interp(uniform_arc_t, arc_length, y_interp)

# 可视化原始曲线和弧长参数化后的曲线
plt.figure(figsize=(10, 6))

# 原始曲线
plt.plot(x_interp, y_interp, label='Original Parametric Curve', color='blue')

# 弧长参数化后的曲线
plt.plot(x_arc_interp, y_arc_interp, label='Arc Length Parametrization', color='red', linestyle='--')

# 设置图例和标题
plt.title('Comparison of Original and Arc Length Parametrization')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
