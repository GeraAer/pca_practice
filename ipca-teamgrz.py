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
noise = np.random.normal(0, 5, n_points)  # 增加噪声的幅度 (将标准差从1增加到10)#,老师和助教如果调节scale为10 则会发现噪点变多，拟合性能变差，也就出现了我报告中另一张图的样子
Z_noise = Z + noise  # 将噪声添加到Z坐标上，使数据更接近真实世界的情况

# 合并生成的点
points = np.vstack((X, Y, Z_noise)).T  # 将X, Y, Z_noise合并成一个1000x3的矩阵，并进行转置，得到1000个点的坐标

# 数据中心化
points_centered = points - np.mean(points, axis=0)  # 对点的坐标进行中心化处理，即每个坐标减去其均值

# 迭代PCA算法
def iterative_pca(data, n_components, n_iterations=100, tolerance=1e-9):
    n_samples, n_features = data.shape  # 获取数据的样本数和特征数
    components = np.random.randn(n_components, n_features)  # 随机初始化主成分向量
    fig = plt.figure(figsize=(10, 8))  # 创建一个图形对象，设置图形大小为10x8
    ax = fig.add_subplot(111, projection='3d')  # 在图形对象中添加一个3D子图
    for iteration in range(n_iterations):  # 迭代次数
        old_components = components.copy()  # 备份当前主成分向量
        scores = np.dot(data, components.T)  # 计算数据在当前主成分上的投影
        components = np.dot(scores.T, data)  # 更新主成分向量
        components /= np.linalg.norm(components, axis=1)[:, np.newaxis]  # 对主成分向量进行归一化处理

        # 绘制每次迭代的拟合平面
        mean = np.mean(data, axis=0)  # 计算数据的均值
        normal_vector = np.cross(components[0], components[1])  # 计算前两个主成分的叉积，得到平面的法向量
        d = -mean.dot(normal_vector)  # 计算平面方程中的常数项d
        xx, yy = np.meshgrid(np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 10),  # 生成X轴的网格数据
                             np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 10))  # 生成Y轴的网格数据
        zz = (-normal_vector[0] * xx - normal_vector[1] * yy - d) * 1. / normal_vector[2]  # 计算Z轴的网格数据
        ax.plot_surface(xx, yy, zz, color='r', alpha=0.3)  # 绘制拟合平面，颜色为红色，透明度为0.3

        if np.allclose(components, old_components, atol=tolerance):  # 判断是否收敛
            break  # 如果收敛则退出迭代

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='g', marker='o', s=10, alpha=0.5)  # 绘制中心化后的三维点云，使用绿色点，设置点的大小和透明度

    # 绘制最终拟合平面
    normal_vector = np.cross(components[0], components[1])  # 重新计算最终的法向量
    d = -mean.dot(normal_vector)  # 计算最终的常数项d
    zz = (-normal_vector[0] * xx - normal_vector[1] * yy - d) * 1. / normal_vector[2]  # 计算最终的Z轴网格数据
    ax.plot_surface(xx, yy, zz, color='g', alpha=0.5)  # 绘制最终的拟合平面，颜色为绿色，透明度为0.5

    ax.set_title('3D Point Cloud and Fitted Plane using Iterative PCA by team Guoruizhi')  # 设置图形标题
    ax.set_xlabel('X')  # 设置X轴标签
    ax.set_ylabel('Y')  # 设置Y轴标签
    ax.set_zlabel('Z')  # 设置Z轴标签
    ax.view_init(elev=20., azim=30)  # 调整视角，设置俯仰角为20度，方位角为30度
    ax.set_box_aspect([1, 1, 1])  # 设置比例尺相等

    plt.tight_layout()  # 自动调整子图间距，避免重叠
    plt.show()  # 显示图形
    return components  # 返回主成分向量

# 使用迭代PCA拟合数据
components = iterative_pca(points_centered, n_components=2)  # 调用迭代PCA函数，计算主成分
