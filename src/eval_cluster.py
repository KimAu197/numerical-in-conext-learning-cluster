import torch
import os
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_blobs
from scipy.spatial import cKDTree  # 快速最近邻查询
import matplotlib.pyplot as plt
import json
import os
import sys
import models
from tasks import get_task_sampler
from samplers import get_data_sampler,rand_select_sampler
from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml
from curriculum import Curriculum
from graph import generate_points_nd, generate_points, generate_out_dis, generate_points_nd2, generate_moons, generate_circle_points
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler

# 1. 加载模型和配置
def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    print("run_path:", run_path)

    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp)) # todo 从yaml中读取conf
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf

def get_transformer_model(run_path, step=-1):
    model, config = get_model_from_run(run_path, step)
    model = model.eval().cuda()
    return model, config

# 2. 评估批次数据
def eval_batch(model, task_sampler, xs_o, n_dims, n_cluster):
    task = task_sampler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs = xs_o[:,:,:n_dims].to(device)
    ys = xs_o[:, :, -1].to(device)
    ys_ = torch.nn.functional.one_hot(ys.long(), num_classes=n_cluster).float()  # 形状为 [32, 200, 2]
    with torch.no_grad():
        pred = model(xs.to(device), ys_.to(device)).detach()
        pred_indices = torch.argmax(pred, dim=-1)  # 形状为 [batch_size, num_points]
        correct = (pred_indices == ys).sum().item()
    return pred_indices, ys, correct

def eval_batch_loss(model, task_sampler, xs_o, n_dims, n_cluster, prompting_strategy):
    task = task_sampler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs = xs_o[:,:,:n_dims].to(device)
    ys = xs_o[:, :, -1].to(device)

    ys_ = torch.nn.functional.one_hot(ys.long(), num_classes=n_cluster).float()  # 形状为 [32, 200, 2]
    with torch.no_grad():
        pred = model(xs.to(device), ys_.to(device)).detach()
        pred_indices = torch.argmax(pred, dim=-1)  # 形状为 [batch_size, num_points]
        correct = (pred_indices == ys).sum().item()
        # plot_decision_boundary(model, xs, ys, prompting_strategy)
        plot_decision_boundary2(model, xs, ys, ys_, prompting_strategy)
        
        
    
    return pred_indices, ys, correct

# 3. 计算评估指标（例如 ARI）
def calculate_ari(true_labels, pred_labels):
    return adjusted_rand_score(true_labels, pred_labels)

# 4. 生成数据
def generate_data(strategy, data_sampler,  n_points, b_size, n_dims, n_cluster):
    if strategy == "standard":
        return gen_standard(data_sampler, n_points, b_size, n_dims, n_cluster)
    elif strategy == "change":
        return gen_oval(data_sampler, n_points, b_size, n_dims, n_cluster)
    elif strategy == "moon":
        return gen_moon(data_sampler, n_points, b_size, n_dims, n_cluster)
    elif strategy == "circle":
        return gen_circle(data_sampler, n_points, b_size, n_dims, n_cluster)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def gen_standard(data_sampler, n_points, b_size, n_dims, n_cluster):
    x_s = []
    for _ in range(b_size):
        points, label = generate_points_nd(n_points, n_cluster, n_dims)
        label_expanded = label[:, np.newaxis]
        
        labeled_points = np.concatenate((points, label_expanded), axis=1)  

        x_s.append(labeled_points)  

    x_s = np.array(x_s)  
    
    return torch.tensor(x_s, dtype=torch.float32)  

def gen_oval(data_sampler, n_points, b_size, n_dims, n_cluster):
    x_s = []
    for _ in range(b_size):
        points, label = generate_out_dis(n_points, n_cluster, n_dims)
        label_expanded = label[:, np.newaxis]
        
        labeled_points = np.concatenate((points, label_expanded), axis=1)  

        x_s.append(labeled_points)  

    x_s = np.array(x_s)  
    return torch.tensor(x_s, dtype=torch.float32)  

def gen_moon(data_sampler, n_points, b_size, n_dims, n_cluster):
    x_s = []
    for _ in range(b_size):
        points, label = generate_moons(n_points, n_cluster, n_dims)
        label_expanded = label[:, np.newaxis]

        labeled_points = np.concatenate((points, label_expanded), axis=1)
        x_s.append(labeled_points)
    
    x_s = np.array(x_s)
    return torch.tensor(x_s, dtype=torch.float32)

def gen_circle(data_sampler, n_points, b_size, n_dims, n_cluster):
    x_s = []
    for _ in range(b_size):
        points, label = generate_circle_points(n_points, n_cluster, n_dims)
        label_expanded = label[:, np.newaxis]

        labeled_points = np.concatenate((points, label_expanded), axis=1)
        x_s.append(labeled_points)

    x_s = np.array(x_s)
    return torch.tensor(x_s, dtype=torch.float32)
    
    
# 5. 聚类比较函数（使用 K-means 或 DBSCAN）
def compare_with_baseline(data, true_labels, n_cluster):
    """
    对批次中的每个样本分别进行聚类，并计算评估指标的平均值。
    
    :param data: 输入数据，形状为 [batch_size, n_points, 3]
    :param true_labels: 真实标签，形状为 [batch_size, n_points]
    :return: 
        - kmeans_avg_metrics: KMeans评估指标的平均值
        - gmm_avg_metrics: GMM评估指标的平均值
        - agglo_avg_metrics: AgglomerativeClustering评估指标的平均值
        - dbscan_avg_metrics: DBSCAN评估指标的平均值
    """
    batch_size = data.shape[0]
    kmeans_metrics_list = []
    gmm_metrics_list = []
    agglo_metrics_list = []
    dbscan_metrics_list = []
    
    for i in range(batch_size):
        # 提取当前样本的数据和标签
        sample_data = data[i]  # 形状为 [n_points, 3]
        sample_labels = true_labels[i]  # 形状为 [n_points]
        
        # KMeans
        kmeans = KMeans(n_clusters=n_cluster, n_init = "auto")
        kmeans_labels = kmeans.fit_predict(sample_data)
        ari_kmeans = adjusted_rand_score(sample_labels, kmeans_labels)
        silhouette_kmeans = silhouette_score(sample_data, kmeans_labels)
        ch_kmeans = calinski_harabasz_score(sample_data, kmeans_labels)
        db_kmeans = davies_bouldin_score(sample_data, kmeans_labels)
        homogeneity_kmeans = homogeneity_score(sample_labels, kmeans_labels)
        completeness_kmeans = completeness_score(sample_labels, kmeans_labels)
        v_measure_kmeans = v_measure_score(sample_labels, kmeans_labels)
        
        kmeans_metrics = {
            "ARI": ari_kmeans,
            "Silhouette": silhouette_kmeans,
            "Calinski-Harabasz": ch_kmeans,
            "Davies-Bouldin": db_kmeans,
            "Homogeneity": homogeneity_kmeans,
            "Completeness": completeness_kmeans,
            "V-Measure": v_measure_kmeans
        }
        kmeans_metrics_list.append(kmeans_metrics)
        
        # GMM
        gmm = GaussianMixture(n_components=n_cluster)
        gmm_labels = gmm.fit_predict(sample_data)
        ari_gmm = adjusted_rand_score(sample_labels, gmm_labels)
        silhouette_gmm = silhouette_score(sample_data, gmm_labels)
        ch_gmm = calinski_harabasz_score(sample_data, gmm_labels)
        db_gmm = davies_bouldin_score(sample_data, gmm_labels)
        homogeneity_gmm = homogeneity_score(sample_labels, gmm_labels)
        completeness_gmm = completeness_score(sample_labels, gmm_labels)
        v_measure_gmm = v_measure_score(sample_labels, gmm_labels)
        
        gmm_metrics = {
            "ARI": ari_gmm,
            "Silhouette": silhouette_gmm,
            "Calinski-Harabasz": ch_gmm,
            "Davies-Bouldin": db_gmm,
            "Homogeneity": homogeneity_gmm,
            "Completeness": completeness_gmm,
            "V-Measure": v_measure_gmm
        }
        gmm_metrics_list.append(gmm_metrics)
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=2)  # 可调整 eps 和 min_samples
        dbscan_labels = dbscan.fit_predict(sample_data)
        
        if len(np.unique(dbscan_labels)) > 1:
            ari_dbscan = adjusted_rand_score(sample_labels, dbscan_labels)
            silhouette_dbscan = silhouette_score(sample_data, dbscan_labels)
            ch_dbscan = calinski_harabasz_score(sample_data, dbscan_labels)
            db_dbscan = davies_bouldin_score(sample_data, dbscan_labels)
            homogeneity_dbscan = homogeneity_score(sample_labels, dbscan_labels)
            completeness_dbscan = completeness_score(sample_labels, dbscan_labels)
            v_measure_dbscan = v_measure_score(sample_labels, dbscan_labels)
        else:
            ari_dbscan = silhouette_dbscan = ch_dbscan = db_dbscan = homogeneity_dbscan = completeness_dbscan = v_measure_dbscan = 0.0
        
        dbscan_metrics = {
            "ARI": ari_dbscan,
            "Silhouette": silhouette_dbscan,
            "Calinski-Harabasz": ch_dbscan,
            "Davies-Bouldin": db_dbscan,
            "Homogeneity": homogeneity_dbscan,
            "Completeness": completeness_dbscan,
            "V-Measure": v_measure_dbscan
        }
        dbscan_metrics_list.append(dbscan_metrics)
        
        
        # Agglomerative Hierarchical Clustering
        agglo = AgglomerativeClustering(n_clusters=n_cluster)
        agglo_labels = agglo.fit_predict(sample_data)
        ari_agglo = adjusted_rand_score(sample_labels, agglo_labels)
        silhouette_agglo = silhouette_score(sample_data, agglo_labels)
        ch_agglo = calinski_harabasz_score(sample_data, agglo_labels)
        db_agglo = davies_bouldin_score(sample_data, agglo_labels)
        homogeneity_agglo = homogeneity_score(sample_labels, agglo_labels)
        completeness_agglo = completeness_score(sample_labels, agglo_labels)
        v_measure_agglo = v_measure_score(sample_labels, agglo_labels)
        
        agglo_metrics = {
            "ARI": ari_agglo,
            "Silhouette": silhouette_agglo,
            "Calinski-Harabasz": ch_agglo,
            "Davies-Bouldin": db_agglo,
            "Homogeneity": homogeneity_agglo,
            "Completeness": completeness_agglo,
            "V-Measure": v_measure_agglo
        }
        agglo_metrics_list.append(agglo_metrics)
        
    # 计算KMeans和GMM指标的平均值
    kmeans_avg_metrics = {
        "ARI": np.mean([m["ARI"] for m in kmeans_metrics_list]),
        "Silhouette": np.mean([m["Silhouette"] for m in kmeans_metrics_list]),
        "Calinski-Harabasz": np.mean([m["Calinski-Harabasz"] for m in kmeans_metrics_list]),
        "Davies-Bouldin": np.mean([m["Davies-Bouldin"] for m in kmeans_metrics_list]),
        "Homogeneity": np.mean([m["Homogeneity"] for m in kmeans_metrics_list]),
        "Completeness": np.mean([m["Completeness"] for m in kmeans_metrics_list]),
        "V-Measure": np.mean([m["V-Measure"] for m in kmeans_metrics_list])
    }
    
    gmm_avg_metrics = {
        "ARI": np.mean([m["ARI"] for m in gmm_metrics_list]),
        "Silhouette": np.mean([m["Silhouette"] for m in gmm_metrics_list]),
        "Calinski-Harabasz": np.mean([m["Calinski-Harabasz"] for m in gmm_metrics_list]),
        "Davies-Bouldin": np.mean([m["Davies-Bouldin"] for m in gmm_metrics_list]),
        "Homogeneity": np.mean([m["Homogeneity"] for m in gmm_metrics_list]),
        "Completeness": np.mean([m["Completeness"] for m in gmm_metrics_list]),
        "V-Measure": np.mean([m["V-Measure"] for m in gmm_metrics_list])
    }
    
    agglo_avg_metrics = {
        "ARI": np.mean([m["ARI"] for m in agglo_metrics_list]),
        "Silhouette": np.mean([m["Silhouette"] for m in agglo_metrics_list]),
        "Calinski-Harabasz": np.mean([m["Calinski-Harabasz"] for m in agglo_metrics_list]),
        "Davies-Bouldin": np.mean([m["Davies-Bouldin"] for m in agglo_metrics_list]),
        "Homogeneity": np.mean([m["Homogeneity"] for m in agglo_metrics_list]),
        "Completeness": np.mean([m["Completeness"] for m in agglo_metrics_list]),
        "V-Measure": np.mean([m["V-Measure"] for m in agglo_metrics_list])
    }
    
    dbscan_avg_metrics = {
        "ARI": np.mean([m["ARI"] for m in dbscan_metrics_list]),
        "Silhouette": np.mean([m["Silhouette"] for m in dbscan_metrics_list]),
        "Calinski-Harabasz": np.mean([m["Calinski-Harabasz"] for m in dbscan_metrics_list]),
        "Davies-Bouldin": np.mean([m["Davies-Bouldin"] for m in dbscan_metrics_list]),
        "Homogeneity": np.mean([m["Homogeneity"] for m in dbscan_metrics_list]),
        "Completeness": np.mean([m["Completeness"] for m in dbscan_metrics_list]),
        "V-Measure": np.mean([m["V-Measure"] for m in dbscan_metrics_list])
    }
    
    return kmeans_avg_metrics, gmm_avg_metrics, agglo_avg_metrics, dbscan_avg_metrics

def calculate_cluster_metrics(pred_labels, true_labels, data):
    """
    计算聚类评估指标。
    
    :param pred_labels: 预测标签，形状为 [batch_size * n_points]
    :param true_labels: 真实标签，形状为 [batch_size * n_points]
    :param data: 输入数据，形状为 [batch_size, n_points, 3]
    :return: 聚类评估指标的字典
    """
    metrics = {
        "ARI": [],
        "Silhouette": [],
        "Calinski-Harabasz": [],
        "Davies-Bouldin": [],
        "Homogeneity": [],
        "Completeness": [],
        "V-Measure": []
    }

    for i in range(batch_size):
        # 提取当前 batch 的数据和标签
        batch_data = data[i]  # 形状为 [n_points, 3]
        batch_pred_labels = pred_labels[i]  # 形状为 [n_points]
        batch_true_labels = true_labels[i]  # 形状为 [n_points]

        # 计算 Silhouette Score
        silhouette = silhouette_score(batch_data, batch_pred_labels)
        metrics["Silhouette"].append(silhouette)

        # 计算 Adjusted Rand Index
        ari = adjusted_rand_score(batch_true_labels, batch_pred_labels)
        metrics["ARI"].append(ari)

        # 计算 Normalized Mutual Information
        chs = calinski_harabasz_score(batch_data, batch_pred_labels)
        metrics["Calinski-Harabasz"].append(chs)

        # 计算 Davies-Bouldin Index
        dbi = davies_bouldin_score(batch_data, batch_pred_labels)
        metrics["Davies-Bouldin"].append(dbi)

        # 计算 Homogeneity
        homogeneity = homogeneity_score(batch_true_labels, batch_pred_labels)
        metrics["Homogeneity"].append(homogeneity)

        # 计算 Completeness
        completeness = completeness_score(batch_true_labels, batch_pred_labels)
        metrics["Completeness"].append(completeness)

        # 计算 V-measure
        v_measure = v_measure_score(batch_true_labels, batch_pred_labels)
        metrics["V-Measure"].append(v_measure)
        
        avg_metrics = {
        "ARI": np.mean(metrics["ARI"]),
        "Silhouette": np.mean(metrics["Silhouette"]),
        "Calinski-Harabasz": np.mean(metrics["Calinski-Harabasz"]),
        "Davies-Bouldin": np.mean(metrics["Davies-Bouldin"]),
        "Homogeneity": np.mean(metrics["Homogeneity"]),
        "Completeness": np.mean(metrics["Completeness"]),
        "V-Measure": np.mean(metrics["V-Measure"]),
    }

    return avg_metrics

def plot_clusters_3d(xs, pred_labels, true_labels, save_path="clusters_3d.png"):
    """
    绘制三维聚类结果图，并将前 10 个样本绘制在一个画布上。

    参数:
        xs (np.ndarray): 三维数据，形状为 [batch_size, n_points, 3]。
        pred_labels (np.ndarray): 预测标签，形状为 [batch_size, n_points]。
        true_labels (np.ndarray): 真实标签，形状为 [batch_size, n_points]。
        save_path (str): 保存图像的路径。
    """
    xs = xs.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    # 确保只绘制前 10 个样本
    num_samples = min(10, xs.shape[0])
    xs = xs[:num_samples]
    pred_labels = pred_labels[:num_samples]
    true_labels = true_labels[:num_samples]

    # 创建画布
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("3D Clustering Results (Predicted vs True Labels)", fontsize=16)

    # 绘制每个样本的聚类结果
    for i in range(num_samples):
        # 预测标签图
        ax1 = fig.add_subplot(2, num_samples, i + 1, projection='3d')
        ax1.scatter(xs[i, :, 0], xs[i, :, 1], xs[i, :, 2], c=pred_labels[i], cmap='viridis', s=10)
        ax1.set_title(f"Sample {i+1} - Predicted")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # 真实标签图
        ax2 = fig.add_subplot(2, num_samples, num_samples + i + 1, projection='3d')
        ax2.scatter(xs[i, :, 0], xs[i, :, 1], xs[i, :, 2], c=true_labels[i], cmap='viridis', s=10)
        ax2.set_title(f"Sample {i+1} - True")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_clusters_2d(xs, pred_labels, true_labels, save_path="clusters_2d.png"):
    """
    绘制二维聚类结果图，并将前 10 个样本绘制在一个画布上。

    参数:
        xs (np.ndarray): 二维数据，形状为 [batch_size, n_points, 2]。
        pred_labels (np.ndarray): 预测标签，形状为 [batch_size, n_points]。
        true_labels (np.ndarray): 真实标签，形状为 [batch_size, n_points]。
        save_path (str): 保存图像的路径。
    """
    # 确保只绘制前 10 个样本
    xs = xs.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    
    num_samples = min(10, xs.shape[0])
    xs = xs[:num_samples]
    pred_labels = pred_labels[:num_samples]
    true_labels = true_labels[:num_samples]

    # 创建画布
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("2D Clustering Results (Predicted vs True Labels)", fontsize=16)

    # 绘制每个样本的聚类结果
    for i in range(num_samples):
        # 预测标签图
        ax1 = fig.add_subplot(2, num_samples, i + 1)
        ax1.scatter(xs[i, :, 0], xs[i, :, 1], c=pred_labels[i], cmap='viridis', s=10)
        ax1.set_title(f"Sample {i+1} - Predicted")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        # 真实标签图
        ax2 = fig.add_subplot(2, num_samples, num_samples + i + 1)
        ax2.scatter(xs[i, :, 0], xs[i, :, 1], c=true_labels[i], cmap='viridis', s=10)
        ax2.set_title(f"Sample {i+1} - True")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_clusters_2d(xs, pred_labels, true_labels, save_path="clusters_2d.png"):
    """
    绘制二维聚类结果图，并将前 10 个样本绘制在一个画布上。

    参数:
        xs (np.ndarray): 二维数据，形状为 [batch_size, n_points, 2]。
        pred_labels (np.ndarray): 预测标签，形状为 [batch_size, n_points]。
        true_labels (np.ndarray): 真实标签，形状为 [batch_size, n_points]。
        save_path (str): 保存图像的路径。
    """
    # 确保只绘制前 10 个样本
    xs = xs.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    
    num_samples = min(10, xs.shape[0])
    xs = xs[:num_samples]
    pred_labels = pred_labels[:num_samples]
    true_labels = true_labels[:num_samples]

    # 创建画布
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("2D Clustering Results (Predicted vs True Labels)", fontsize=16)

    # 绘制每个样本的聚类结果
    for i in range(num_samples):
        # 预测标签图
        ax1 = fig.add_subplot(2, num_samples, i + 1)
        ax1.scatter(xs[i, :, 0], xs[i, :, 1], c=pred_labels[i], cmap='viridis', s=10)
        ax1.set_title(f"Sample {i+1} - Predicted")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        # 真实标签图
        ax2 = fig.add_subplot(2, num_samples, num_samples + i + 1)
        ax2.scatter(xs[i, :, 0], xs[i, :, 1], c=true_labels[i], cmap='viridis', s=10)
        ax2.set_title(f"Sample {i+1} - True")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_decision_boundary(model, X, y, task, resolution=30):
    # 1. 将 X 和 y 从 GPU 复制到 CPU，并转换为 NumPy 数组
    # with open("/mnt/data/jinzi/numerical-in-context-learning/data.json", "r") as f:
    #     data = json.load(f)
        
    # X = torch.tensor(data[task]["points"], dtype=torch.float32)  # 转换为 PyTorch 张量
    # y = torch.tensor(data[task]["labels"], dtype=torch.float32)  # 转换为 PyTorch 张量
    
    X = X[0]
    y = y[0]
    
    X = X.cpu().numpy()
    y = y.cpu().numpy()
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init = "auto")
    kmeans.fit(grid)
    # 预测每个点的 cluster
    labels = kmeans.predict(grid)
    labels = torch.tensor(labels, dtype=torch.long)
    # print(labels)
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2)
    one_hot_labels = one_hot_labels.clone().detach()
    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # [1, N, 2]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(grid[:, 0], grid[:, 1], c=labels, cmap=plt.cm.Paired, marker='.', s=10)
    plt.title("KMeans Clustering Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("kmeans_clustering_result.png")  # 保存图像
    plt.close() 
    device = next(model.parameters()).device
    grid_tensor = grid_tensor.to(device)
    # TODO: 这里的ys_dummy是全部pad成0，还是用ture_label y的值？
    # ys_dummy = torch.zeros_like(grid_tensor).to(device) 
    ys_dummy = one_hot_labels.clone().detach().unsqueeze(0).to(device)

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
    plt.savefig(f"cluster_2d_{task}.png")
    
# ... existing code ...
def plot_decision_boundary2(model, X, y, ys_, task, resolution=30):
    X_np = X[0].cpu().numpy()
    y_np = y[0].cpu().numpy()
    device = next(model.parameters()).device

    # 生成网格点
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]  # (resolution^2, 2)

    # 预测每个 grid 点的概率
    probs = []
    for i in range(len(grid)):
        g = grid[i]
        g_tensor = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(device)
        x_comb = torch.cat([X[0], g_tensor], dim=0).unsqueeze(0).to(device)

        ys_dummy = torch.zeros(1, 2).to(device)
        ys_comb = torch.cat([ys_[0], ys_dummy], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_comb, ys_comb, inds=[x_comb.shape[1]-1])  # 只预测最后一个点
            prob = torch.softmax(pred, dim=-1)[0, 0, 1].item()  # class 1 概率
            probs.append(prob)

    # 转成 heatmap
    prob_map = np.array(probs).reshape(xx.shape)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, prob_map, levels=100, cmap='coolwarm')  # 红-蓝 colormap
    plt.colorbar(label='P(class=1)')
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='coolwarm', edgecolor='black', s=40)
    plt.title(f"Heatmap of Class Probabilities - {task}")
    plt.savefig(f"/mnt/data/jinzi/numerical-in-context-learning-cluster/src/fig/heatmap_prob_{task}.png")
    plt.close()
    
    with torch.no_grad():
        pred_real = model(X[0].unsqueeze(0).to(device), ys_[0].unsqueeze(0).to(device))
        pred_labels_real = pred_real.argmax(dim=-1)[0].cpu().numpy()  # shape: (N,)

    # 可视化真实数据点预测
    X_np = X[0].cpu().numpy()
    y_np = y[0].cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_np[:, 0], X_np[:, 1], c=pred_labels_real, cmap='coolwarm', edgecolor='black', s=40, label='Predicted')
    plt.title(f"Prediction on Real Data - {task}")
    plt.savefig(f"/mnt/data/jinzi/numerical-in-context-learning-cluster/src/fig/real_data_prediction_{task}.png")
    plt.close()

    
# 6. 模型评估和基准比较
def evaluate_model(model, task_sampler, data_sampler, n_points, batch_size, prompting_strategy, epoch, n_dims, n_cluster):
    model_metrics_list = []
    kmeans_metrics_list = []
    gmm_metrics_list = []
    agglo_metrics_list = []
    dbscan_metrics_list = []
    total_correct = 0
    total_samples = 0
    for _ in range(epoch):
        xs_o = generate_data(prompting_strategy, data_sampler, n_points, batch_size, n_dims, n_cluster)
        pred, labels, correct1 = eval_batch(model, task_sampler, xs_o, n_dims, n_cluster)
        xs = xs_o[:,:,:n_dims]
        if n_dims == 2:
            plot_clusters_2d(xs, pred, labels, f"2d_clusters_{epoch}.png")
        elif n_dims == 3:
            plot_clusters_3d(xs, pred, labels, f"3d_clusters_{epoch}.png")
        model_metrics = calculate_cluster_metrics(pred.cpu().numpy(), labels.cpu().numpy(), xs.cpu().numpy())
        model_metrics_list.append(model_metrics)
        kmeans_avg_metrics, gmm_avg_metrics, agglo_avg_metrics, dbscan_avg_metrics = compare_with_baseline(xs.cpu().numpy(), labels.cpu().numpy(), n_cluster)
        kmeans_metrics_list.append(kmeans_avg_metrics)
        gmm_metrics_list.append(gmm_avg_metrics)
        agglo_metrics_list.append(agglo_avg_metrics)
        dbscan_metrics_list.append(dbscan_avg_metrics)
        
        total_correct += correct1
        # print(correct1)
        total_samples += batch_size*n_points
        
    accuracy = total_correct / total_samples
    
    avg_model_metrics = {
        k: float(np.mean([m[k] for m in model_metrics_list])) for k in model_metrics_list[0]
    }
    avg_kmeans_metrics = {
        k: float(np.mean([m[k] for m in kmeans_metrics_list])) for k in kmeans_metrics_list[0]
    }
    avg_gmm_metrics = {
        k: float(np.mean([m[k] for m in gmm_metrics_list])) for k in gmm_metrics_list[0]
    }
    avg_agglo_metrics = {
        k: float(np.mean([m[k] for m in agglo_metrics_list])) for k in agglo_metrics_list[0]
    }
    avg_dbscan_metrics = {
        k: float(np.mean([m[k] for m in dbscan_metrics_list])) for k in dbscan_metrics_list[0]
    }
    
    return avg_model_metrics, avg_kmeans_metrics, avg_gmm_metrics, avg_agglo_metrics, avg_dbscan_metrics, accuracy

# 7. 运行评估
def run_evaluation(run_path, n_points, batch_size, prompting_strategy, epoch):
    model, config = get_transformer_model(run_path)
    curriculum = Curriculum(config.training.curriculum)
    # init sampler
    n_dims = model.n_dims
    # n_dims = args.training.n_dims
    bsize = config.training.batch_size
    n_cluster = config.training.n_cluster
    # 每个step采样一个bs进行训练
    data_sampler = get_data_sampler(config.training.data,  # gaussion or uniform
                                    n_dims=n_dims, 
                                    n_cluster = config.training.n_cluster # 添加n_cluster参数
                                    )
    
    task_sampler = get_task_sampler(
        config.training.task,
        n_dims,
        bsize,
        w_type = config.training.w_type, # 添加w_sample 参数
        num_tasks = config.training.num_tasks,
        n_cluster = config.training.n_cluster, # 添加 cluster的类别
        **config.training.task_kwargs,
    )
    
    avg_model_metrics, avg_kmeans_metrics, avg_gmm_metrics, avg_agglo_metrics, avg_dbscan_metrics, accuracy = evaluate_model(
        model, task_sampler, data_sampler, n_points, batch_size, prompting_strategy, epoch, n_dims, n_cluster
    )
    
    # # 打印模型评估指标
    # print("Model Evaluation Metrics:")
    # for metric, value in avg_model_metrics.items():
    #     print(f"{metric}: {value:.4f}")

    # # 打印KMeans评估指标
    # print("\nKMeans Evaluation Metrics:")
    # for metric, value in avg_kmeans_metrics.items():
    #     print(f"{metric}: {value:.4f}")

    # # 打印DBSCAN评估指标
    # print("\ngmm Evaluation Metrics:")
    # for metric, value in avg_dbscan_metrics.items():
    #     print(f"{metric}: {value:.4f}")
        
    # print("\nAgglomerativeClustering Evaluation Metrics:")
    # for metric, value in avg_agglo_metrics.items():
    #     print(f"{metric}: {value:.4f}")

    # 打印总正确率
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
        # 将结果存储为字典
    results = {
        "model_metrics": avg_model_metrics,
        "kmeans_metrics": avg_kmeans_metrics,
        "gmm_metrics": avg_gmm_metrics,
        "agglomerative_metrics": avg_agglo_metrics,
        "dbscan_metrics": avg_dbscan_metrics,
        "accuracy": accuracy
    }
    
    results = {k: float(v) if isinstance(v, np.float32) else v for k, v in results.items()}
    
    return results

def evaluate_loss_model(model, task_sampler, data_sampler, n_points, batch_size, prompting_strategy, epoch, n_dims, n_cluster):
    for _ in range(epoch):
        xs_o = generate_data(prompting_strategy, data_sampler, n_points, batch_size, n_dims, n_cluster)
        pred, labels, correct1 = eval_batch_loss(model, task_sampler, xs_o, n_dims, n_cluster, prompting_strategy)
        xs = xs_o[:,:,:n_dims]
        if n_dims == 2:
            plot_clusters_2d(xs, pred, labels, f"2d_clusters_{epoch}.png")
        elif n_dims == 3:
            plot_clusters_3d(xs, pred, labels, f"3d_clusters_{epoch}.png")
    
    return pred


def run_loss_evaluation(run_path, n_points, batch_size, prompting_strategy, epoch):
    model, config = get_transformer_model(run_path)
    curriculum = Curriculum(config.training.curriculum)
    # init sampler
    n_dims = model.n_dims
    # n_dims = args.training.n_dims
    bsize = config.training.batch_size
    n_cluster = config.training.n_cluster
    # 每个step采样一个bs进行训练
    data_sampler = get_data_sampler(config.training.data,  # gaussion or uniform
                                    n_dims=n_dims, 
                                    n_cluster = config.training.n_cluster # 添加n_cluster参数
                                    )
    
    task_sampler = get_task_sampler(
        config.training.task,
        n_dims,
        bsize,
        w_type = config.training.w_type, # 添加w_sample 参数
        num_tasks = config.training.num_tasks,
        n_cluster = config.training.n_cluster, # 添加 cluster的类别
        **config.training.task_kwargs,
    )
    
    loss = evaluate_loss_model(
        model, task_sampler, data_sampler, n_points, batch_size, prompting_strategy, epoch, n_dims, n_cluster
    )
    
    return loss

# if __name__ == "__main__":
#     run_path = "/mnt/data/jinzi/numerical-in-context-learning/result/model/cluster_3d/20e16d7e-8a0e-409b-8562-1cbcf9841bb5"
#     all_results = {}
#     for n_points in range(100,200,10):
#         print(f"Running evaluation for {n_points} points...")
#         batch_size = 32
#         prompting_strategy = "change"
#         epoch = 10
        
#         results = run_evaluation(run_path, n_points, batch_size, prompting_strategy, epoch)
#         print(results["accuracy"])
#         all_results[n_points] = results
    
#     # 将所有结果保存到一个JSON文件中
#     with open("all_results_o.json", "w") as f:
#         json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    run_path = "/mnt/data/jinzi/numerical-in-context-learning-cluster/result/model/cluster_2d/5b45ff5c-247b-41b2-99be-6c5ab9df80a4"
    task = 1
    if task == 0:
        all_results = {}
        n_points = 200
        batch_size = 32
        prompting_strategy = "standard"
        epoch = 1
        results = run_evaluation(run_path, n_points, batch_size, prompting_strategy, epoch)
        all_results[n_points] = results
        # 将所有结果保存到一个JSON文件中
        with open("all_results_o2.json", "w") as f:
            json.dump(all_results, f, indent=4)    
    else:
        all_results = {}
        n_points = 200
        batch_size = 32
        prompting_strategy = "standard"
        epoch = 1
        results = run_loss_evaluation(run_path, n_points, batch_size, prompting_strategy, epoch)
        # results = run_evaluation(run_path, n_points, batch_size, prompting_strategy, epoch)
        # all_results[n_points] = results
        # # 将所有结果保存到一个JSON文件中
        # with open("all_results_o2.json", "w") as f:
        #     json.dump(all_results, f, indent=4)   
    