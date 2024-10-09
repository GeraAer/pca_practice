import numpy as np  # 导入Numpy库，用于科学计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具 ,后来我发现并没有什么卵用。

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

# 计算协方差矩阵
cov_matrix = np.cov(data_centered, rowvar=False)  # 计算协方差矩阵，用于衡量不同变量之间的线性关系

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # 对协方差矩阵进行特征值分解，得到特征值和特征向量

# 按特征值从大到小排序
sorted_indices = np.argsort(eigenvalues)[::-1]  # 将特征值按从大到小排序，返回排序索引
sorted_eigenvalues = eigenvalues[sorted_indices]  # 根据排序索引，得到排序后的特征值
sorted_eigenvectors = eigenvectors[:, sorted_indices]  # 根据排序索引，得到排序后的特征向量

# 选择前两个和前三个主成分用于可视化
k2 = 2  # 选择前两个主成分
k3 = 3  # 选择前三个主成分
top2_eigenvectors = sorted_eigenvectors[:, :k2]  # 提取前两个特征向量
top3_eigenvectors = sorted_eigenvectors[:, :k3]  # 提取前三个特征向量

# 投影到前两个主成分
data_projected_2d = np.dot(data_centered, top2_eigenvectors)  # 将数据投影到前两个主成分上，得到二维投影
# 投影到前三个主成分
data_projected_3d = np.dot(data_centered, top3_eigenvectors)  # 将数据投影到前三个主成分上，得到三维投影

# 绘制二维投影散点图
plt.figure(figsize=(8, 6))  # 创建一个8x6的画布
plt.scatter(data_projected_2d[:, 0], data_projected_2d[:, 1], c='b', marker='o', s=10, alpha=0.5)  # 绘制二维散点图，使用蓝色点
plt.title('2D Projection by team Guoruizhi')  # 设置图形标题
plt.xlabel('Principal Component 1')  # 设置X轴标签
plt.ylabel('Principal Component 2')  # 设置Y轴标签
plt.grid(True)  # 添加网格线
plt.axis('equal')  # 设置比例尺相等
plt.xlim(np.min(data_projected_2d[:, 0]) * 1.2, np.max(data_projected_2d[:, 0]) * 1.2)  # 调整X轴范围，放大比例尺
plt.ylim(np.min(data_projected_2d[:, 1]) * 1.2, np.max(data_projected_2d[:, 1]) * 1.2)  # 调整Y轴范围，放大比例尺
plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示二维投影散点图

# 绘制三维投影散点图
fig = plt.figure(figsize=(10, 8))  # 创建一个10x8的画布
ax = fig.add_subplot(111, projection='3d')  # 在图形对象中添加一个3D子图
ax.scatter(data_projected_3d[:, 0], data_projected_3d[:, 1], data_projected_3d[:, 2], c='g', marker='o', s=10, alpha=0.5)  # 绘制三维散点图，使用绿色点
ax.set_title('3D Projection by team Guoruizhi')  # 设置图形标题
ax.set_xlabel('Principal Component 1')  # 设置X轴标签
ax.set_ylabel('Principal Component 2')  # 设置Y轴标签
ax.set_zlabel('Principal Component 3')  # 设置Z轴标签
ax.view_init(30, 45)  # 调整视角，设置俯仰角和方位角
ax.set_box_aspect([1,1,1])  # 设置比例尺相等
ax.set_xlim(np.min(data_projected_3d[:, 0]) * 1.2, np.max(data_projected_3d[:, 0]) * 1.2)  # 调整X轴范围，放大比例尺
ax.set_ylim(np.min(data_projected_3d[:, 1]) * 1.2, np.max(data_projected_3d[:, 1]) * 1.2)  # 调整Y轴范围，放大比例尺
ax.set_zlim(np.min(data_projected_3d[:, 2]) * 1.2, np.max(data_projected_3d[:, 2]) * 1.2)  # 调整Z轴范围，放大比例尺
plt.tight_layout()  # 自动调整子图间距
plt.show()   # 大家好，我是三维投影散点图！