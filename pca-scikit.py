import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 读取数据文件
data = np.loadtxt('PointsNormals绿萝.txt')  # 读取点云数据文件

# 获取点云的坐标和法向量数据
points = data[:, :3]  # 提取前3列数据，作为点的坐标
normals = data[:, 3:]  # 提取后3列数据，作为法向量

# 数据中心化
points_centered = points - np.mean(points, axis=0)  # 对点的坐标进行中心化处理，即每个坐标减去其均值
normals_centered = normals - np.mean(normals, axis=0)  # 对法向量进行中心化处理

# 合并中心化后的点云坐标和法向量数据
data_centered = np.hstack((points_centered, normals_centered))  # 合并中心化后的点坐标和法向量数据

# 使用PCA进行降维,调库的核心部分，其实手搓真没难多少，我个人体感这是我做过最舒服的手搓代码. 当然我们这次使用的是124主元。
pca = PCA(n_components=4)  # 初始化PCA，设定需要4个主成分
pca.fit(data_centered)  # 拟合PCA模型
components = pca.components_  # 获取主成分

# 选择第一个、第二个和第四个主成分用于可视化
selected_indices = [0, 1, 3]  # 选择第1、2和4个主成分
selected_components = components[selected_indices, :]  # 提取对应的主成分

# 投影到选择的主成分
data_projected = np.dot(data_centered, selected_components.T)  # 将数据投影到选择的主成分上

# 绘制二维投影散点图
plt.figure(figsize=(8, 6))  # 创建一个8x6的画布
plt.scatter(data_projected[:, 0], data_projected[:, 1], c='b', marker='o', s=10, alpha=0.5)  # 绘制二维散点图，使用蓝色点
plt.title('2D Projection using 1st and 2nd Principal Components with scikit by team Guoruizhi')  # 设置图形标题
plt.xlabel('Principal Component 1')  # 设置X轴标签
plt.ylabel('Principal Component 2')  # 设置Y轴标签
plt.grid(True)  # 添加网格线
plt.axis('equal')  # 设置比例尺相等
plt.xlim(np.min(data_projected[:, 0]) * 1.2, np.max(data_projected[:, 0]) * 1.2)  # 调整X轴范围，放大比例尺
plt.ylim(np.min(data_projected[:, 1]) * 1.2, np.max(data_projected[:, 1]) * 1.2)  # 调整Y轴范围，放大比例尺
plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示二维投影散点图

# 绘制三维投影散点图
fig = plt.figure(figsize=(10, 8))  # 创建一个10x8的画布
ax = fig.add_subplot(111, projection='3d')  # 在图形对象中添加一个3D子图
ax.scatter(data_projected[:, 0], data_projected[:, 1], data_projected[:, 2], c='g', marker='o', s=10, alpha=0.5)  # 绘制三维散点图，使用绿色点
ax.set_title('3D Projection using 1st, 2nd and 4th Principal Components with scikit by team Guoruizhi')  # 设置图形标题
ax.set_xlabel('Principal Component 1')  # 设置X轴标签
ax.set_ylabel('Principal Component 2')  # 设置Y轴标签
ax.set_zlabel('Principal Component 4')  # 设置Z轴标签
ax.view_init(30, 45)  # 调整视角，设置俯仰角和方位角
ax.set_box_aspect([1, 1, 1])  # 设置比例尺相等
ax.set_xlim(np.min(data_projected[:, 0]) * 1.2, np.max(data_projected[:, 0]) * 1.2)  # 调整X轴范围，放大比例尺
ax.set_ylim(np.min(data_projected[:, 1]) * 1.2, np.max(data_projected[:, 1]) * 1.2)  # 调整Y轴范围，放大比例尺
ax.set_zlim(np.min(data_projected[:, 2]) * 1.2, np.max(data_projected[:, 2]) * 1.2)  # 调整Z轴范围，放大比例尺
plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示三维投影散点图
