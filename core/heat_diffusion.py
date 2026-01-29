import numpy as np
import networkx as nx
from scipy import sparse
import torch




def heat_diff(G, sources, device='cuda'):
    ctx = build_context(G, device=device)

    # 2. 极速计算
    return compute_field(ctx, sources)


def build_context(G, device='cuda'):
    """步骤1: 预处理。构建邻接矩阵并上传 GPU (只运行一次)"""
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 构建归一化转置矩阵 W = (D^-1 A)^T
    A = nx.adjacency_matrix(G, nodelist=nodes, weight='weight')
    degrees = np.array(A.sum(axis=1)).flatten()
    d_inv = np.divide(1.0, degrees, out=np.zeros_like(degrees, dtype=float), where=degrees != 0)
    P = sparse.diags(d_inv) @ A
    W_scipy = P.T.tocsr()

    # 转为 PyTorch GPU 稀疏张量
    W_torch = torch.sparse_csr_tensor(
        crow_indices=torch.tensor(W_scipy.indptr, dtype=torch.int64),
        col_indices=torch.tensor(W_scipy.indices, dtype=torch.int64),
        values=torch.tensor(W_scipy.data, dtype=torch.float32),
        size=W_scipy.shape,
        device=device
    )

    return {"W": W_torch, "nodes": nodes, "map": node_to_idx, "n": n, "dev": device}


def compute_field(ctx, seeds, alpha=0.1, max_iter=100, tol=1e-6):
    """
    步骤2: 计算核心。输入预处理好的上下文和种子权重，计算全图能量场。

    此函数基于 GPU 执行加权 Personalized PageRank (RWR) 算法。

    :param ctx: (dict) 上下文数据字典。
                必须包含 'W' (GPU稀疏矩阵), 'map' (节点索引), 'dev' (设备) 等。
                由 `build_context` 函数生成。

    :param seeds: (dict) 初始热源权重字典 {node_id: weight}。
                  例如: {'Apple': 5.0, 'Fruit': 1.0}。
                  权重会被自动归一化，数值越大代表该节点作为 Query 的重要性越高。

    :param alpha: (float, 0 < alpha <= 1) 重启概率 / 能量保留系数。
                  决定了能量场的“视野范围”：
                  - alpha 较大 (0.5 ~ 0.9): “近视”。能量被强力拉回种子点，只关注直接相连的强关系（适合同义词挖掘）。
                  - alpha 较小 (0.1 ~ 0.2): “远视”。能量可以传导到 3-4 跳之外，覆盖全图结构（适合隐式推理）。
                  - 默认 0.3 是一个平衡值。

    :param max_iter: (int) 最大迭代轮数。
                     通常 20-50 轮即可收敛。设置过大只会浪费计算资源。

    :param tol: (float) 收敛容忍度 (L1 范数)。
                当前后两次迭代的能量变化总量小于此值时，提前结束计算。

    :return: (dict) {node_id: energy_score}。
             全图节点的能量分布，数值介于 0~1 之间，总和为 1。
    """
    # 构建初始向量 q
    q_np = np.zeros(ctx['n'], dtype=np.float32)
    for node, weight in seeds.items():
        if node in ctx['map']:
            q_np[ctx['map'][node]] = weight

    if q_np.sum() > 0: q_np /= q_np.sum()  # 归一化

    # GPU 迭代
    q = torch.from_numpy(q_np).to(ctx['dev'])
    r = q.clone()
    damping = 1.0 - alpha

    for _ in range(max_iter):
        r_prev = r.clone()
        r = alpha * q + damping * (ctx['W'] @ r)
        if torch.norm(r - r_prev, p=1).item() < tol: break

    r_cpu = r.cpu().numpy()
    return {node: float(r_cpu[i]) for i, node in enumerate(ctx['nodes'])}


def compute_weighted_energy_field_torch(G, seed_weights, restart_prob=0.3, max_iter=100, tolerance=1e-6, device='cuda'):
    """
    基于 PyTorch (GPU) 加速的局部能量场计算 (Weighted PPR)
    """
    # 检查是否有可用的 GPU，如果没有则回退到 CPU
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，切换回 CPU 模式。")
        device = 'cpu'

    device = torch.device(device)

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # --- 1. 预处理 (CPU端) ---
    # 利用 Scipy 高效构建矩阵 (这一步在 CPU 上做通常够快且方便)
    A = nx.adjacency_matrix(G, nodelist=nodes, weight='weight')
    degrees = np.array(A.sum(axis=1)).flatten()

    with np.errstate(divide='ignore'):
        d_inv = 1.0 / degrees
    d_inv[degrees == 0] = 0

    # P = D^-1 * A
    P = sparse.diags(d_inv) @ A
    W_scipy = P.T.tocsr()  # 转置并转为 CSR 格式，PyTorch 对 CSR 的矩阵乘法优化最好

    # --- 2. 转换为 PyTorch Tensor 并移至 GPU ---
    # 构建 PyTorch 稀疏矩阵
    # 注意: PyTorch 的稀疏 CSR 需要 row_ptr, col_indices, values
    W_torch = torch.sparse_csr_tensor(
        crow_indices=torch.as_tensor(W_scipy.indptr, dtype=torch.int64),
        col_indices=torch.as_tensor(W_scipy.indices, dtype=torch.int64),
        values=torch.as_tensor(W_scipy.data, dtype=torch.float32),  # GPU上通常用 float32 速度更快
        size=W_scipy.shape,
        device=device
    )

    # 构建 Query 向量 q
    q_np = np.zeros(n, dtype=np.float32)
    total_weight = 0.0
    for node, weight in seed_weights.items():
        if node in node_to_idx:
            q_np[node_to_idx[node]] = weight
            total_weight += weight

    if total_weight == 0:
        return {node: 0.0 for node in nodes}

    q_np /= total_weight

    # 将 q 移至 GPU
    q = torch.from_numpy(q_np).to(device)

    # 初始化 r
    r = q.clone()

    # 预先定义常量
    alpha = restart_prob
    damping = 1.0 - alpha

    # --- 3. GPU 迭代计算 ---
    # print(f"开始 GPU 迭代 (Device: {device})...")

    for iteration in range(max_iter):
        r_prev = r.clone()

        # 核心公式: r = alpha * q + (1-alpha) * (W @ r)
        # torch.mv 是 稀疏矩阵 x 稠密向量 的推荐操作
        # 或者直接使用 @ 运算符: r = alpha * q + damping * (W_torch @ r)
        diffusion = W_torch @ r
        r = alpha * q + damping * diffusion

        # 检查收敛 (L1 范数)
        # torch.dist 或 torch.norm
        diff = torch.norm(r - r_prev, p=1).item()  # .item() 将标量转回 Python float

        if diff < tolerance:
            # print(f"收敛于第 {iteration + 1} 次迭代.")
            break

    # --- 4. 结果传回 CPU ---
    # .cpu().numpy() 将显存数据拉回内存
    r_cpu = r.cpu().numpy()

    return {node: float(r_cpu[node_to_idx[node]]) for node in nodes}