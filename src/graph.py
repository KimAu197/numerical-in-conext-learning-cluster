import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics.pairwise import euclidean_distances
import torch
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
import json
import os

def generate_points(N, k, seed=None):
    """
    根据输入的点数 N 生成k cluster的点组, 并返回生成的点及其类型。

    :param N: 生成的点数
    :param k: cluster的数量
    :param seed: 随机种子（可选）
    :return: points, point_type - 生成的点和点的类型
    """
    # 随机选择生成点的类型
    if seed is not None:
        np.random.seed(seed)
    
    point_type = np.random.choice([0, 1, 2])  # 0--圆； 1--Gaussian； 2--月牙

    if point_type == 0:  # 同心圆形 默认k==2
        factor = np.random.uniform(0.5, 1)  
        noise = np.random.uniform(0, 0.5)    
        points, label = make_circles(n_samples=N, factor=factor, noise=noise, random_state=seed)
    
    elif point_type == 1:  # Gaussian
        cluster_std = np.random.uniform(0, 5, size=k)
        points, label = make_blobs(n_samples=N, centers=k, cluster_std=cluster_std, random_state=seed)
    
    elif point_type == 2:  # 月牙
        noise = np.random.uniform(0, 0.5)    
        points, label = make_moons(n_samples=N, noise=noise, random_state=seed)
    # # 随机缩放参数
    # scale_x = np.random.uniform(0.1, 100)
    # scale_y = np.random.uniform(0.1, 100)

    # # 随机旋转参数
    # angle = np.random.uniform(0, 2 * np.pi)

    # # 随机平移参数
    # dx = np.random.uniform(-1e4, 1e4)
    # dy = np.random.uniform(-1e4, 1e4)

    # # 对点进行缩放、旋转和平移
    # points = scale(points, scale_x, scale_y)
    # points = rotate(points, angle)
    # points = translate(points, dx, dy)

    return points, label, point_type

def generate_moons(N, k, n, seed=None):
    """
    生成月牙型数据。

    参数:
        N (int): 总样本数。
        k (int): 聚类数。
        seed (int): 随机种子。
    """
    noise = np.random.uniform(0, 0.1)    
    points, label = make_moons(n_samples=N, noise=noise, random_state=seed)
    
    return points, label

def generate_points_nd2(N, k, n, seed=None, min_distance=10, max_attempts=100):
    """
    生成不重合的聚类数据。

    参数:
        N (int): 总样本数。
        k (int): 聚类数。
        n (int): 特征维度。
        seed (int): 随机种子。
        min_distance (float): 聚类中心之间的最小距离。
        max_attempts (int): 最大尝试次数。

    返回:
        points (np.ndarray): 生成的数据点，形状为 [N, n]。
        label (np.ndarray): 每个数据点的标签，形状为 [N]。
    """
    if seed is not None:
        np.random.seed(seed)

    for _ in range(max_attempts):
        # 生成聚类中心
        centers = np.random.uniform(0, 100, size=(k, n))  # 假设数据范围在 [0, 100] 内

        # 检查聚类中心之间的距离
        dist_matrix = euclidean_distances(centers)
        np.fill_diagonal(dist_matrix, np.inf)  # 忽略对角线上的 0
        min_dist = dist_matrix.min()

        # 如果聚类中心之间的距离大于 min_distance，生成数据
        if min_dist > min_distance:
            cluster_std = np.random.uniform(0, 1, size=k)  # 减小标准差
            points, label = make_blobs(n_samples=N, centers=centers, cluster_std=cluster_std, random_state=seed)
            return points, label

    cluster_std = np.random.uniform(0, 5, size=k)
    points, label = make_blobs(n_samples=N, centers=k, cluster_std=cluster_std, n_features=n, random_state=seed)
    return points, label


            
def generate_points_nd(N, k, n, seed=None):
    """
    根据输入的点数 N 生成k cluster的点组, 并返回生成的点及其类型。

    :param N: 生成的点数
    :param k: cluster的数量
    :param n: 维度
    :param seed: 随机种子（可选）
    :return: points, point_type - 生成的点和点的类型
    """
    # 随机选择生成点的类型
    if seed is not None:
        np.random.seed(seed)    

    cluster_std = np.random.uniform(0, 5, size=k)
    points, label = make_blobs(n_samples=N, centers=k, cluster_std=cluster_std, n_features=n, random_state=seed)
    
    return points, label

def generate_circle_points(N, k, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    factor = np.random.uniform(0.2, 0.5)  
    noise = np.random.uniform(0, 0.2)    
    points, label = make_circles(n_samples=N, factor=factor, noise=noise, random_state=seed)
    
    return points, label
    


def plot_generated_points(points, point_labels, seed=None):
    # 将点转换为 NumPy 数组
    points_array = np.array(points)

    # 绘制散点图
    plt.figure(figsize=(8, 8))

    # 根据聚类标签绘制不同类型的点
    for label in range(k):
        # 选择对应标签的点
        cluster_points = points_array[point_labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')

    plt.title(f'Generated Points with {k} Clusters')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')  # 保持比例
    plt.grid()
    plt.legend()
    plt.savefig('generated_points.png')

def translate(points, dx, dy):
    """ 平移操作 """
    points[:, 0] += dx
    points[:, 1] += dy
    return points

def rotate(points, angle, center=(0,0)):
    """ 旋转操作 """
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    points = points - center
    points = np.dot(points, R.T)
    points = points + center
    return points


def scale(points, scale_x, scale_y):
    """ 缩放操作 """
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y
    return points

def generate_ellipsoidal_cluster(center, n_points, axes, seed=None):
    """
    生成一个高维椭球形聚类，中心为center，轴长为axes。
    
    :param center: 聚类的中心点，形状为 (n_dimensions,)
    :param n_points: 聚类的点数
    :param axes: 每个维度的轴长，形状为 (n_dimensions,)
    :param seed: 随机种子
    :return: 聚类的点，形状为 (n_points, n_dimensions)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_dimensions = len(center)
    
    # 生成高维标准正态分布的点
    points = np.random.randn(n_points, n_dimensions)  # N(0, 1) 分布的点
    
    # 计算变换矩阵
    transformation_matrix = np.diag(axes)  # 对角矩阵

    # 应用变换矩阵
    ellipsoidal_points = points @ transformation_matrix.T  # 变换矩阵应用到点
    
    # 平移到聚类中心
    ellipsoidal_points += center  # 聚类中心
    
    return ellipsoidal_points

def generate_two_ellipsoidal_clusters(n_points, n_dimensions=3, seed=None):
    """
    生成两个高维椭球形的聚类，中心和大小都随机，确保聚类不会重叠。
    
    :param n_points: 每个聚类的点数
    :param n_dimensions: 数据的维度
    :param seed: 随机种子
    :return: 两个聚类的点，形状为 (n_points, n_dimensions)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 随机生成第一个聚类的中心
    center_1 = np.random.uniform(-5, 5, size=n_dimensions)
    axes_1 = np.random.uniform(0, 6, size=n_dimensions)  # 每个维度的轴长
    
    # 随机生成第二个聚类的中心，确保与第一个不同并满足最小距离要求
    while True:
        center_2 = np.random.uniform(-5, 5, size=n_dimensions)
        # 计算两个聚类中心的距离
        distance_between_centers = np.linalg.norm(center_2 - center_1)
        
        # 根据最大半轴的尺寸来计算安全距离，避免重叠
        safe_distance = np.max(axes_1) + np.random.uniform(1, 3)  # 基于最大轴的安全距离
        if distance_between_centers > safe_distance:
            break
    
    axes_2 = np.random.uniform(0, 6, size=n_dimensions)  # 第二个聚类的每个维度的轴长
    
    # 生成两个聚类的点
    cluster_1 = generate_ellipsoidal_cluster(center_1, n_points, axes_1, seed)
    cluster_2 = generate_ellipsoidal_cluster(center_2, n_points, axes_2, seed)
    
    return cluster_1, cluster_2

def combine_clusters_with_labels(cluster_1, cluster_2):
    """
    将两个聚类的点随机组合成一个点集，并为每个点分配标签。
    
    :param cluster_1: 第一个聚类的点，形状为 (n_points, n_dimensions)
    :param cluster_2: 第二个聚类的点，形状为 (n_points, n_dimensions)
    :return: 
        - points: 组合后的点集，形状为 (2 * n_points, n_dimensions)
        - labels: 对应的标签，形状为 (2 * n_points,)
    """
    # 将两个聚类的点堆叠在一起
    points = np.vstack((cluster_1, cluster_2))
    
    # 生成对应的标签
    labels = np.hstack((np.zeros(len(cluster_1)), np.ones(len(cluster_2))))
    
    # 随机打乱点集和标签的顺序
    indices = np.arange(len(points))
    np.random.shuffle(indices)
    points = points[indices]
    labels = labels[indices]
    
    return points, labels


def generate_out_dis(N, n_cluster, n_dimensions, seed=None):
    """
    生成两个高维椭球形的聚类，并组合成一个点集。
    
    :param N: 总点数
    :param n_dimensions: 数据的维度
    :param seed: 随机种子
    :return: 
        - points: 组合后的点集，形状为 (N, n_dimensions)
        - labels: 对应的标签，形状为 (N,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n1 = N // 2
    cluster_1, cluster_2 = generate_two_ellipsoidal_clusters(n_points=n1, n_dimensions=n_dimensions, seed=seed)
    points, labels = combine_clusters_with_labels(cluster_1, cluster_2)
    
    return points, labels
    
def eval_data(strategy, n_points, b_size, n_dims, n_cluster):
    # 根据策略选择生成函数
    if strategy == "standard":
        generate_func = generate_points_nd
    elif strategy == "change":
        generate_func = generate_out_dis
    elif strategy == "moon":
        generate_func = generate_moons
    elif strategy == "circle":
        generate_func = generate_circle_points
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 调用通用生成函数
    return generate_eval_batch_data(generate_func, n_points, b_size, n_dims, n_cluster)

def generate_eval_batch_data(generate_func, n_points, b_size, n_dims, n_cluster):
    """
    通用批量数据生成函数。

    参数:
        generate_func: 数据生成函数。
        n_points (int): 每个样本的点数。
        b_size (int): 批量大小。
        n_dims (int): 数据的维度。
        n_cluster (int): 聚类数。

    返回:
        torch.Tensor: 生成的数据，形状为 [b_size, n_points, n_dims + 1]。
    """
    x_s = []
    for _ in range(b_size):
        # 调用生成函数生成数据
        points, label = generate_func(n_points, n_cluster, n_dims)
        label_expanded = label[:, np.newaxis]
        
        # 将点和标签拼接
        labeled_points = np.concatenate((points, label_expanded), axis=1)
        x_s.append(labeled_points)
    
    # 转换为 PyTorch 张量
    return torch.tensor(np.array(x_s), dtype=torch.float32)

def plot_decision_boundary(model, X, y, task, epoch, save_dict, resolution=30):
    
    X = X.cpu().numpy()
    y = y.cpu().numpy()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 2. 创建 2D 网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    # 3. 转换成模型输入格式
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # [1, N, 2]

    # 4. 将 grid_tensor 和 ys_dummy 移动到与模型相同的设备
    device = next(model.parameters()).device
    grid_tensor = grid_tensor.to(device)
    ys_dummy = torch.zeros_like(grid_tensor).to(device)

    # 5. 送入 Transformer 模型预测
    with torch.no_grad():
        preds = model(grid_tensor, ys_dummy)
        pred_labels = preds.argmax(dim=-1).squeeze().cpu().numpy().reshape(xx.shape)

    # 6. 绘图
    display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=pred_labels)
    # display.plot()
    # display.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor="black")
    display.plot(cmap='viridis')  # 将 DecisionBoundaryDisplay 的 cmap 设置为 'viridis'
    display.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor="black")
    
    plt.title(f"Transformer Decision Boundary {task}")
    path = f"/mnt/data/jinzi/numerical-in-context-learning-cluster/fig2/decision_boundary_{epoch}_{task}.png"
    
    output_dir = os.path.dirname(path)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path)
    return path
    
    
if __name__ == "__main__":
    N = 200  # 生成的点数
    k = 2    # cluster的数量
    seed = 42  # 随机种子
    n = 2  # 维度
    points, labels = generate_circle_points(N, k, n)
    
    # 将 points 和 labels 转换为 Python 原生类型
    data = {
        "points": points.tolist(),  # 将 NumPy 数组转换为列表
        "labels": labels.tolist()   # 将 NumPy 数组转换为列表
    }
    
    # 保存到 JSON 文件
    with open("generated_data.json", "w") as f:
        json.dump(data, f, indent=4)
        
    plot_generated_points(points, labels)
    
    print("Data saved to generated_data.json")