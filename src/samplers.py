import math

import torch
import random
import numpy as np
from graph import generate_points, generate_points_nd, generate_points_nd2

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError

def rand_select_sampler(sampler1, sampler2): # todo 添加随机选择sample
    if random.random() < 0.5:
        return sampler1
    else:
        return sampler2


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "uniform": UniformSampler,
        "cluster_2d": Cluster2DSampler,
        "cluster_nd": ClusterNDSampler,
        # add
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        if data_name == "cluster_2d":
            assert n_dims == 2
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError



def sample_transformation(eigenvalues, normalize=False): # 根据给定的特征值（eigenvalues）生成一个线性变换矩阵，用于对数据进行线性变换
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

class Cluster2DSampler(DataSampler):
    """
    通过graph.py中的generate_points函数随机生成k cluster个组的2d点集
    并且通过standardize函数对所有点集进行标准化(缩放和偏移)
    最终返回x_s
    x_s格式: [b_size, n_points, self.n_dims] 
        -b_size：batch_size
        -n_points：每个batch中点的数量
        -self.n_dims：维度, self.n_dims=(点+label) = self.n_dims+1
    """
    def __init__(self, n_dims, scale=None, bias=None, n_cluster=2):
        super().__init__(n_dims)
        self.scale = scale
        self.bias = bias
        self.n_cluster = n_cluster  # 存储 n_cluster 参数
            
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        assert self.n_dims == 2  # 二维点集
        x_s = []
        
        for _ in range(b_size):
            # points, label, point_type = generate_points_nd(n_points, self.n_cluster, seed=seeds)  # 生成点集
            points, label = generate_points_nd(n_points, self.n_cluster, 2, seed=seeds)  # 生成点集
            
            
            # # 生成 one-hot 向量
            # one_hot_labels = np.zeros((label.size, self.n_cluster))  # 创建一个全零的数组
            # one_hot_labels[np.arange(label.size), label] = 1  # 将对应的类别位置设为 1
            label_expanded = label[:, np.newaxis]
            
            # labeled_points = np.concatenate((points, one_hot_labels), axis=1)  # 在最后一个维度上组合
            labeled_points = np.concatenate((points, label_expanded), axis=1)  # 在最后一个维度上组合

            # 将每个批次的结果转换为一个 NumPy 数组
            x_s.append(labeled_points)  # 直接添加 NumPy 数组

        # 将所有批次的结果堆叠成一个张量
        x_s = np.array(x_s)  # 转换为 NumPy 数组，形状为 [b_size, n_points, n_dims]
        return torch.tensor(x_s, dtype=torch.float32)  # 确保数据类型为 float32
        
    def standardize(self, x_s):
        """
        将点集中心化 并且缩放成10*10的区域内（均匀缩放到10*10 x y方向缩放不一致）
        """
        center = np.mean(x_s, axis=0)
        # 中心化
        centered_points = x_s - center
        
        # 计算每个维度的最大绝对值
        max_abs_value = np.max(np.abs(centered_points), axis=0)
        
        # 计算缩放因子，确保在10x10区域内均匀缩放
        scale_factor = 1 / np.max(max_abs_value) if np.max(max_abs_value) > 0 else 1  # 防止除以零
        
        # 缩放点集
        standardized_points = centered_points * scale_factor
        
        return standardized_points
    
    
class ClusterNDSampler(DataSampler):
    """
    通过graph.py中的generate_points_nd函数随机生成k cluster个组的nd点集
    最终返回x_s
    x_s格式: [b_size, n_points, self.n_dims] 
        -b_size：batch_size
        -n_points：每个batch中点的数量
        -self.n_dims：维度, self.n_dims=点
    """
    def __init__(self, n_dims, scale=None, bias=None, n_cluster=2):
        super().__init__(n_dims)
        self.scale = scale
        self.bias = bias
        self.n_cluster = n_cluster  # 存储 n_cluster 参数
        self.n_dims = n_dims  # 存储 n_dims 参数
            
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        assert self.n_cluster <= self.n_dims
        x_s = []
        
        for _ in range(b_size):
            points, label= generate_points_nd(n_points, self.n_cluster, self.n_dims, seed=seeds)  # 生成点集
            label_expanded = label[:, np.newaxis]
            labeled_points = np.concatenate((points, label_expanded), axis=1)  # 在最后一个维度上组合
            # 将每个批次的结果转换为一个 NumPy 数组
            x_s.append(labeled_points)  # 直接添加 NumPy 数组

        # 将所有批次的结果堆叠成一个张量
        x_s = np.array(x_s)  # 转换为 NumPy 数组，形状为 [b_size, n_points, n_dims]
        return torch.tensor(x_s, dtype=torch.float32)  # 确保数据类型为 float32

class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None: #
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else: # 利用seeds
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
                # 缩放矩阵
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None: # 截断特征维度 将多余的维度置零
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class UniformSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            a = -1.96 #
            b=   1.96
            xs_b = a + (b - a) * torch.rand(b_size, n_points, self.n_dims)

            # xs_b = torch.rand(b_size, n_points, self.n_dims) # U [a,b]  [-+1.96]
        else: # 利用seed
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                np.random.seed(seed)
                xs_b[i] = torch.rand(n_points, self.n_dims, generator=generator)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None: # 截断特征维度 将多余的维度置零
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

