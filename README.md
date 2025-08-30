# Numerical In-Context Learning for Clustering with Linear Transformers

This repository contains code developed for my undergraduate thesis on **in-context learning with linear Transformers** applied to **numerical clustering tasks**.  
The experiments investigate whether linear Transformers generalize clustering rules, or instead behave as **memory-driven template matchers**.

## 📂 Repository Structure

- **environment.yml** – dependencies and environment setup  
- **data.json**, **all_results_o2.json** – example datasets / experiment outputs  
- **non-linear-regression-IVP.ipynb**, **cluster.ipynb** – Jupyter notebooks for exploration and visualization  

### `src/`
- **train.py** – training entry point  
- **eval_cluster.py** – evaluation of clustering tasks  
- **graph.py** – synthetic data generation / sampling  
- **models.py**, **base_models.py** – model definitions (linear Transformer, etc.)  
- **curriculum.py**, **samplers.py**, **tasks.py** – task and data curriculum utilities  
- **plot_utils.py** – helper functions for plotting results  
- **conf/** – YAML configs for models and tasks  

---

## ⚙️ Setup

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate num-cluster


