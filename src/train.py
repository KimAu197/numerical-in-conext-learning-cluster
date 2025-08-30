import os
from random import randint
import uuid
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import numpy as np
from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler,rand_select_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from graph import eval_data, plot_decision_boundary
import random
import wandb
import json

torch.backends.cudnn.benchmark = True

# 从环境变量中获取 API 密钥
wandb_api_key = os.getenv("WANDB_API_KEY")
# 登录到 wandb
wandb.login(key=wandb_api_key)
torch.cuda.set_device(3)

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 的 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU，也需要设置


# 设置随机种子
set_seed(52) # 开局可以固定seeds

def train_step(model, xs, ys, optimizer, loss_func): # 单个批次（batch）的前向传播、损失计算、反向传播和优化器更新
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def eval_step(model, xs, ys, loss_func): # 单个批次（batch）的前向传播和损失计算
    with torch.no_grad():
        output = model(xs, ys)
    loss = loss_func(output, ys)
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    global task_sampler1, task_sampler2
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate,weight_decay=args.training.weight_decay)  # add weight decay
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    # 检查checkpoints pt
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()
    # init sampler
    n_dims = model.n_dims
    # n_dims = args.training.n_dims
    bsize = args.training.batch_size
    # 每个step采样一个bs进行训练
    data_sampler = get_data_sampler(args.training.data,  # gaussion or uniform
                                    n_dims=n_dims, 
                                    n_cluster = args.training.n_cluster # 添加n_cluster参数
                                    )
    
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        w_type = args.training.w_type, # 添加w_sample 参数
        num_tasks = args.training.num_tasks,
        n_cluster = args.training.n_cluster, # 添加 cluster的类别
        **args.training.task_kwargs,
    )

    pbar = tqdm(range(starting_step, args.training.train_steps))
    train_steps = args.training.train_steps
    total_steps = train_steps - starting_step
    split_point = starting_step + total_steps // 2  # 前 50% 和后 50% 的分界点
    num_training_examples = args.training.num_training_examples
        
    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        # todo If_2distribution
        if args.training.If_two_distribution:
            if args.training.If_RandomShuffle_2distribution:
                task_sampler = rand_select_sampler(task_sampler1,task_sampler2) # random select
            else: # first half gaussian , half uniform
                if i < split_point:
                    task_sampler = task_sampler1
                else:
                    task_sampler = task_sampler2

        # 支持稀疏任务
        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None: # 训练集中样本的总数
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        # set_seed(42) # todo 不能直接固定seed 不然data全一样了

        xs_o = data_sampler.sample_xs( # 采样xs
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated, # 维度
            **data_sampler_args,
        )

        if args.training.save_sample:
            # 定义保存文件的路径
            save_path = os.path.join(args.out_dir, "save_data.npy")
            
            # 获取最后一个 batch 的点
            xs_o_last_batch = xs_o[-1].numpy()  # 转换为 NumPy 数组
            
            # 保存到文件
            if os.path.exists(save_path):
                # 如果文件已存在，读取现有数据并追加新数据
                existing_data = np.load(save_path, allow_pickle=True)  # 读取现有数据
                updated_data = np.append(existing_data, [xs_o_last_batch], axis=0)  # 追加新数据
                np.save(save_path, updated_data)  # 保存更新后的数据
            else:
                # 如果文件不存在，创建新文件并写入数据
                np.save(save_path, np.array([xs_o_last_batch]))  # 保存新数据
            
        
        xs = xs_o[:,:,:n_dims]

        task = task_sampler(**task_sampler_args)

        ys = task.evaluate(xs_o) # ground truth 变成one-hot向量
        
        # 计算损失
        loss_func = task.get_training_metric() # 损失metric
        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)  # train update 参数。 loss 为总误差
        # 点损失和baseline损失计算
        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric() # loss_func
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)# 个数据点的逐点平均误差 对bs求mean
        # baseline_loss：有效特征维度/数据点数量 表示任务的难度
        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )
        
        # # 多加一个在不同distribute数据集上eval的:
        # strategy_losses = {}
        # for strategy in ["moon", "circle", "change", "standard"]:
        #     xs_evo = eval_data(strategy, curriculum.n_points, bsize, n_dims, args.training.n_cluster)
        #     xs_ev = xs_evo[:,:,:n_dims]
        #     ys_ev = task.evaluate(xs_evo)
            
        #     loss_eval, output_eval = eval_step(model, xs_ev.cuda(), ys_ev.cuda(), loss_func)
        #     strategy_losses[f"overall_loss_eval_{strategy}"] = loss_eval
        
        # out_dir = "/mnt/data/jinzi/numerical-in-context-learning-cluster/fig"
        
        # # boundary plot
        # with open("/mnt/data/jinzi/numerical-in-context-learning-cluster/data.json", "r") as f:
        #     data = json.load(f)
        # if ((i <= 2000) and (i % 100 == 0)) or (i % 1000 == 0):
        #     for strategy in ["moon", "circle", "change", "standard"]:
        #         X = torch.tensor(data[strategy]["points"], dtype=torch.float32)  
        #         y = torch.tensor(data[strategy]["labels"], dtype=torch.float32)  
                
        #         path = plot_decision_boundary(model, X, y, strategy, i, args.out_dir)
        #         path = f"/mnt/data/jinzi/numerical-in-context-learning-cluster/fig2/decision_boundary_{i}_{strategy}.png"
        #         wandb.log({f"boundary_plot_{strategy}": wandb.Image(path)}, step=i)
        
        # 日记
        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss, # 相对损失
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    # **strategy_losses,
                },
                step=i,
            )
        # 更新
        curriculum.update()
        # 显示训练进度
        pbar.set_description(f"loss {loss}")
        # 保存模型状态 添加一个文件
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)
        # 保存检查点模型
        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run: # 测试模式 减少规模
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__, # config=args.__dict__ 用于将 args 对象的所有属性（即命令行参数和配置项）以字典形式传递给 WandB 作为配置
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()
    # 开始训练
    train(model, args)
    # 训练完成后计算评估指标 并保存至 metrics.json
    if not args.test_run: #
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema) #
    args = parser.parse_quinfig() # config
    assert args.model.family in ["gpt2", "lstm","gptJ"] #
    print(f"Running with: {args}")

    if not args.test_run: # 设置运行id
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
