import math
from scipy.integrate import simpson,trapz
import torch
import numpy as np
# 常用的评估函数
def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()
ce_loss = torch.nn.CrossEntropyLoss()

bce_loss2 = torch.nn.BCEWithLogitsLoss()

def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    return bce_loss(output, ys)

def ce_entropy(ys_pred, ys):
    ys_indices = torch.argmax(ys, dim=-1).view(-1)
    # ys_pred = ys_pred.view(-1)  # 形状为 [batch_size * n_points, num_classes]
    # ys_pred = ys_pred.unsqueeze(-1) 
    # ys = ys.view(-1)
    # print(ys_pred.shape, ys.shape)
    loss = ce_loss(ys_pred, ys_indices)
    return loss

def bi_entropy(ys_pred, ys):
    loss = bce_loss2(ys_pred, ys)
    return loss

# 抽象基类，所有具体任务（如线性回归、分类等）都从它继承
class Task:
    def __init__(self, n_dims, batch_size,w_type="gaussian", pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict # 任务参数池
        self.seeds = seeds
        self.w_type = w_type
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks,w_type):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError

# w sampler
def get_task_sampler(
    task_name, n_dims, batch_size,w_type="gaussian", pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression
    }
    ode_task_names_to_classes = {
        "ode_ivp_case1": ODEIVPCase1,
    }
    cluster_task_names_to_classes = {
        "cluster_2d": Cluster2DTask,
        "cluster_nd": ClusterNDTask,
    }
    
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, w_type,**kwargs)
        return lambda **args: task_cls(n_dims, batch_size,w_type, pool_dict, **args, **kwargs)
    elif task_name in ode_task_names_to_classes:
        task_cls = ode_task_names_to_classes[task_name]
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    elif task_name in cluster_task_names_to_classes:
        task_cls = cluster_task_names_to_classes[task_name]
        return lambda **args: task_cls(n_dims, batch_size, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError

class LinearRegression(Task):
    def __init__(self, n_dims, batch_size,w_type="gaussian", pool_dict=None, seeds=None, scale=1, ):
        """scale: a constant by which to scale the randomly sampled weights.
        w_sample_type: either "gaussian" or "uniform """
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.w_type = w_type
        if pool_dict is None and seeds is None:
            if self.w_type == "gaussian":
                self.w_b = torch.randn(self.b_size, self.n_dims, 1)
            elif self.w_type == "uniform":
                self.w_b = torch.tensor(np.random.uniform(-2, 2, size=(self.b_size, self.n_dims, 1)), dtype=torch.float32)
            elif self.w_type == "add":
                self.w_b = torch.randn(self.b_size, self.n_dims, 1) + torch.tensor(np.random.uniform(-2, 2, size=(self.b_size, self.n_dims, 1)), dtype=torch.float32)
            else:
                raise ValueError("Invalid w_type. Must be 'gaussian' or 'uniform'.")

        elif seeds is not None: # 利用生成的seeds
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                np.random.seed(seed)
                if self.w_type == "gaussian":
                    self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
                elif self.w_type == "uniform":
                    # mu sigma  np.normal  numpy.random.uniform
                    self.w_b[i] = torch.tensor( np.random.uniform(-2, 2, size=(n_dims, 1)), dtype=torch.float32 )
                # todo 添加 w1+w2
                elif self.w_type == "add":
                    self.w_b = torch.randn(self.b_size, self.n_dims, 1) + torch.tensor(
                        np.random.uniform(-2, 2, size=(self.b_size, self.n_dims, 1)), dtype=torch.float32
                    )
                else:
                    raise ValueError("Invalid w_type. Must be 'gaussian' or 'uniform' or 'add'.")
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
    def evaluate(self, xs_b):
        # w_b = self.w_b.to(xs_b.device)
        # ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        w_b = self.w_b.to(xs_b.device).float() # 统一为float32
        xs_b = xs_b.float()
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks,w_type, **kwargs):
        if w_type == "gaussian":
            return {"w": torch.randn(num_tasks, n_dims, 1)}
        elif w_type == "uniform":
            return {"w": torch.rand(num_tasks, n_dims, 1) * 2 - 1}
        else:
            raise ValueError("Invalid w_type. Must be 'gaussian' or 'uniform'.")

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class Cluster2DTask(Task):
    """
    2维cluster任务
    生成sample的时候 已经生成对应的label类别 所以这里只需要重新生成对应的one-hot向量即可
    以及对应的metric()
    """
    def __init__(self, n_dims, batch_size, n_cluster, seeds=None, scale=1):
        super(Cluster2DTask,self).__init__(n_dims, batch_size, seeds)
        self.n_cluster = n_cluster  
    
    def evaluate(self, xs):
        """
        截断最后一个one-hot向量
        """
        labels = xs[:, :, -1].long()  # 转换为整数类型
    
        # 将 labels 转换为 one-hot 编码，形状为 [batch_size, n_points, n_cluster]
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=self.n_cluster).float()
        
        return one_hot_labels  
    
    @staticmethod
    def get_metric():
        return squared_error
    @staticmethod
    def get_training_metric():
        return bi_entropy
    

class ClusterNDTask(Task):
    """
    高维cluster任务
    生成sample的时候 已经生成对应的label类别 所以这里只需要重新生成对应的one-hot向量即可
    以及对应的metric()
    """
    def __init__(self, n_dims, batch_size, n_cluster, seeds=None, scale=1):
        super(ClusterNDTask,self).__init__(n_dims, batch_size, seeds)
        self.n_cluster = n_cluster  
        self.n_dims = n_dims
    
    def evaluate(self, xs):
        """
        截断最后一个one-hot向量
        """
        labels = xs[:, :, -1].long()  # 转换为整数类型
    
        # 将 labels 转换为 one-hot 编码，形状为 [batch_size, n_points, n_cluster]
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=self.n_cluster).float()
        
        return one_hot_labels  
    
    @staticmethod
    def get_metric():
        return squared_error
    @staticmethod
    def get_training_metric():
        return ce_entropy       
        
class ODEIVPCase1(Task):
    """
    The problem is 
    dy/dt = ay+b,
    y(t0) = t0,
    t\in [t0,t1]
    要求x格式  x[i,j] = [a,b,y0,steps,0,0,0,...,0]
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """因为 dy/dt=ay+b中所有的数据都在x中传递了,没有其他的参数,所以这里的n_dims就是a和b的维度"""
        super(ODEIVPCase1,self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.t_0, self.t_e = 0, 5
 
    def evaluate(self, xs_b):
        ground_truth = lambda a, b, y0, t: (y0 + b / a) * np.exp(a * t) - b / a
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], self.n_dims, device=xs_b.device)
        for i in range(xs_b.shape[0]):
            for j in range(xs_b.shape[1]):
                a = xs_b[i, j, 0]
                b = xs_b[i, j, 1]
                y0 = xs_b[i, j, 2]
                steps = int(xs_b[i, j, 3].item())
                t = np.linspace(self.t_0, self.t_e, steps)
                ys_b[i, j] = torch.cat([ground_truth(a, b, y0, t),torch.zeros(self.n_dims - steps)],dim=0)
        return ys_b
    
    @staticmethod
    def get_metric():
        return squared_error
    @staticmethod
    def get_training_metric():
        return mean_squared_error