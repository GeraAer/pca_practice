import numpy as np  # 导入Numpy库，用于科学计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from mpl_toolkits.mplot3d import Axes3D  # 从mpl_toolkits.mplot3d中导入Axes3D，用于3D绘图

# 生成数据
np.random.seed(42)  # 固定随机种子，确保结果可重复
n_points = 1000  # 点的数量

# 生成在平面上的点
X = np.random.uniform(-10, 10, n_points)  # 生成1000个在-10到10范围内均匀分布的随机数，作为X坐标
Y = np.random.uniform(-10, 10, n_points)  # 生成1000个在-10到10范围内均匀分布的随机数，作为Y坐标
Z = 0.5 * X + 0.2 * Y  # 构建平面Z = 0.5X + 0.2Y

# 添加噪声
noise = np.random.normal(0, 10, n_points)  # 添加标准差为10的正态分布噪声
Z_noise = Z + noise  # 将噪声添加到Z坐标上，使数据更接近真实世界的情况

# 合并生成的点
points = np.vstack((X, Y, Z_noise)).T  # 将X, Y, Z_noise合并成一个1000x3的矩阵，并进行转置，得到1000个点的坐标

# 数据中心化
points_centered = points - np.mean(points, axis=0)  # 对点的坐标进行中心化处理，即每个坐标减去其均值

# 随机SVD算法
def randomized_svd(data, n_components, n_oversamples=10, n_iterations=5):
    n_samples, n_features = data.shape  # 获取数据的样本数和特征数
    P = np.random.randn(n_features, n_components + n_oversamples)  # 生成随机投影矩阵
    Z = np.dot(data, P)  # 将数据投影到低维子空间
    for _ in range(n_iterations):  # 进行多次迭代以提高投影精度
        Z = np.dot(data, np.dot(data.T, Z))
    Q, _ = np.linalg.qr(Z)  # 对投影矩阵进行QR分解，构建正交基
    B = np.dot(Q.T, data)  # 在低维子空间中计算近似SVD
    U_hat, Sigma, VT = np.linalg.svd(B, full_matrices=False)  # 计算近似SVD
    U = np.dot(Q, U_hat)  # 恢复原始数据的近似主成分
    U, Sigma, VT = U[:, :n_components], Sigma[:n_components], VT[:n_components, :]  # 截取前n_components个主成分
    return U, Sigma, VT  # 返回近似SVD的结果

# 使用随机SVD拟合数据
U, Sigma, VT = randomized_svd(points_centered, n_components=2)

# 获取主成分
components = VT

# 投影到前两个主成分
points_projected = np.dot(points_centered, components.T)

# 拟合平面
mean = np.mean(points_centered, axis=0)
normal_vector = np.cross(components[0], components[1])  # 计算前两个主成分的叉积，得到平面的法向量
d = -mean.dot(normal_vector)  # 计算平面方程中的常数项d

# 生成用于绘制平面的网格数据
xx, yy = np.meshgrid(np.linspace(np.min(points_centered[:, 0]), np.max(points_centered[:, 0]), 10),
                     np.linspace(np.min(points_centered[:, 1]), np.max(points_centered[:, 1]), 10))
zz = (-normal_vector[0] * xx - normal_vector[1] * yy - d) * 1. / normal_vector[2]

# 绘制三维点云和拟合平面
fig = plt.figure(figsize=(10, 8))  # 创建一个图形对象，设置图形大小为10x8
ax = fig.add_subplot(111, projection='3d')  # 在图形对象中添加一个3D子图
ax.scatter(points_centered[:, 0], points_centered[:, 1], points_centered[:, 2], c='g', marker='o', s=10, alpha=0.5)  # 绘制中心化后的三维点云，使用绿色点，设置点的大小和透明度
ax.plot_surface(xx, yy, zz, color='g', alpha=0.5)  # 绘制拟合平面，颜色为绿色，透明度为0.5

ax.set_title('3D Point Cloud and Fitted Plane using Randomized SVD by team Guoruizhi')  # 设置图形标题
ax.set_xlabel('X')  # 设置X轴标签
ax.set_ylabel('Y')  # 设置Y轴标签
ax.set_zlabel('Z')  # 设置Z轴标签
ax.view_init(elev=20., azim=30)  # 调整视角，设置俯仰角为20度，方位角为30度
ax.set_box_aspect([1, 1, 1])  # 设置比例尺相等

plt.tight_layout()  # 自动调整子图间距，避免重叠
plt.show()  # 显示图形
