import numpy as np  # 导入Numpy库，用于科学计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

# 读取数据文件
data = np.loadtxt('PointsNormals绿萝.txt')  # 通过Numpy的loadtxt函数读取点云数据文件

# 获取点云的坐标和法向量数据
points = data[:, :3]  # 提取前3列数据，作为点的坐标
normals = data[:, 3:]  # 提取后3列数据，作为法向量

# 数据中心化
points_centered = points - np.mean(points, axis=0)  # 对点的坐标进行中心化处理，即每个坐标减去其均值
normals_centered = normals - np.mean(normals, axis=0)  # 对法向量进行中心化处理

# 合并中心化后的点云坐标和法向量数据
data_centered = np.hstack((points_centered, normals_centered))  # 将中心化后的点坐标和法向量数据合并

# 计算协方差矩阵
cov_matrix = np.cov(data_centered, rowvar=False)  # 计算协方差矩阵，用于衡量不同变量之间的线性关系

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # 对协方差矩阵进行特征值分解，得到特征值和特征向量

# 按特征值从大到小排序
sorted_indices = np.argsort(eigenvalues)[::-1]  # 将特征值按从大到小排序，返回排序索引
sorted_eigenvalues = eigenvalues[sorted_indices]  # 根据排序索引，得到排序后的特征值
sorted_eigenvectors = eigenvectors[:, sorted_indices]  # 根据排序索引，得到排序后的特征向量

# 选择第一个、第二个和第四个主成分用于可视化
indices = [0, 1, 3]  # 选择第1、2和4个主成分的索引
top3_eigenvectors = sorted_eigenvectors[:, indices]  # 根据选择的索引提取对应的特征向量

# 投影到选择的主成分
data_projected_3d = np.dot(data_centered, top3_eigenvectors)  # 将中心化后的数据投影到选择的主成分上，得到投影后的数据

# 绘制三维投影散点图
fig = plt.figure(figsize=(10, 8))  # 创建一个图形对象，设置图形大小
ax = fig.add_subplot(111, projection='3d')  # 在图形对象中添加一个3D子图
ax.scatter(data_projected_3d[:, 0], data_projected_3d[:, 1], data_projected_3d[:, 2], c='g', marker='o', s=10, alpha=0.5)  # 绘制3D散点图，使用绿色的点，设置点的大小和透明度
ax.set_title('3D Projection using 1st, 2nd and 4th Principal Components')  # 设置图形标题
ax.set_xlabel('Principal Component 1')  # 设置X轴标签
ax.set_ylabel('Principal Component 2')  # 设置Y轴标签
ax.set_zlabel('Principal Component 4')  # 设置Z轴标签
ax.view_init(elev=20., azim=30)  # 调整视角，设置俯仰角和方位角
ax.set_box_aspect([1,1,1])  # 设置比例尺相等
ax.set_xlim(np.min(data_projected_3d[:, 0]) * 1.2, np.max(data_projected_3d[:, 0]) * 1.2)  # 设置X轴范围，放大比例尺
ax.set_ylim(np.min(data_projected_3d[:, 1]) * 1.2, np.max(data_projected_3d[:, 1]) * 1.2)  # 设置Y轴范围，放大比例尺
ax.set_zlim(np.min(data_projected_3d[:, 2]) * 1.2, np.max(data_projected_3d[:, 2]) * 1.2)  # 设置Z轴范围，放大比例尺
plt.tight_layout()  # 自动调整子图间距，避免重叠
plt.show()  # 显示图形
