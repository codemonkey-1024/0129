from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Tuple, Any, Callable, Literal, Optional
from tqdm import tqdm
import json
from pathlib import Path
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
import json
import math
from sklearn.cluster import KMeans


try:
    from loguru import logger
except Exception:  # 兜底到标准 logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("llm_client")


def recursive_cluster_indices(data, indices=None, max_size=4):
    """
    递归聚类，返回每个簇对应的原始索引列表。
    """
    # 1. 初始化：如果是第一次调用，生成 0 到 N-1 的索引数组
    if indices is None:
        indices = np.arange(len(data))

    n_samples = data.shape[0]

    # 2. 终止条件：如果当前簇大小 <= 4，直接返回当前的索引组
    if n_samples <= max_size:
        return [indices.tolist()]  # 返回 Python list 格式的索引

    # 3. K-Means 二分
    try:
        # 强制分2类
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data)
    except Exception:
        # 容错处理：如果无法聚类（如数据完全相同），强制按顺序切分
        split_idx = int(np.ceil(n_samples / 2))
        return (recursive_cluster_indices(data[:split_idx], indices[:split_idx], max_size) +
                recursive_cluster_indices(data[split_idx:], indices[split_idx:], max_size))

    # 4. 获取掩码 (Mask)
    mask0 = (labels == 0)
    mask1 = (labels == 1)

    # 5. 关键步骤：同步切分数据 和 索引
    # 如果某一边为空（极端情况），强制按数量切分
    if np.sum(mask0) == 0 or np.sum(mask1) == 0:
        split_idx = int(np.ceil(n_samples / 2))
        return (recursive_cluster_indices(data[:split_idx], indices[:split_idx], max_size) +
                recursive_cluster_indices(data[split_idx:], indices[split_idx:], max_size))

    # 6. 递归调用，将结果合并
    results = []
    results.extend(recursive_cluster_indices(data[mask0], indices[mask0], max_size))
    results.extend(recursive_cluster_indices(data[mask1], indices[mask1], max_size))

    return results


def hash_text(input_string: str) -> str:
    """使用 MD5 生成哈希字符串并截取后15位

    Args:
        input_string: 输入字符串

    Returns:
        15位十六进制哈希值
    """
    return hashlib.md5(input_string.encode()).hexdigest()[-15:]

def generate_batches(data_list, batch_size):
    """Generate batches from a list with a specified batch size."""
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]


def read_jsonl(path: Union[str, Path], pb=False) -> List[Dict[str, Any]]:
    """
    读取 JSON Lines 文件，逐行解析为 dict。

    :param path: JSONL 文件路径，支持字符串或 Path 对象
    :return: 包含所有行的字典列表
    """
    # 统一转换为 Path 对象
    path_obj = Path(path) if isinstance(path, str) else path

    with path_obj.open("r", encoding="utf-8") as f:
        # 使用生成器表达式提高内存效率
        if pb:
            return [json.loads(line.strip()) for line in tqdm(f, desc=f"reading {path}") if line.strip()]
        else:
            return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl(path: Union[str, Path], rows: List[Dict[str, Any]], pbar=False) -> None:
    """
    将多行字典写入 JSONL 文件，并在目录不存在时自动创建目录。

    :param path: 输出文件路径，支持字符串或 Path 对象
    :param rows: 字典列表
    """
    # 统一转换为 Path 对象
    path_obj = Path(path) if isinstance(path, str) else path

    # ✅ 自动创建父目录
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open("w", encoding="utf-8") as f:
        if pbar:
            for row in tqdm(rows, desc=f"writing to {path}"):
                json_str = json.dumps(row, ensure_ascii=False)
                f.write(json_str + "\n")
        else:
            for row in rows:
                json_str = json.dumps(row, ensure_ascii=False)
                f.write(json_str + "\n")


def write_jsonl_stream(path: Union[str, Path], rows: List[Dict[str, Any]]) -> None:
    """
    流式写入 JSONL 文件，适用于大量数据。

    :param path: 输出文件路径，支持字符串或 Path 对象
    :param rows: 字典列表或可迭代对象
    """
    path_obj = Path(path) if isinstance(path, str) else path
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open("w", encoding="utf-8") as f:
        for row in rows:
            json_str = json.dumps(row, ensure_ascii=False)
            f.write(json_str + "\n")


def read_jsonl_stream(path: Union[str, Path]):
    """
    流式读取 JSONL 文件，逐行产生 dict。

    :param path: JSONL 文件路径，支持字符串或 Path 对象
    :yield: 每行对应的字典
    """
    path_obj = Path(path) if isinstance(path, str) else path

    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过空行
                yield json.loads(line.strip())

def parallel_apply(
    fn: Callable[..., Any],
    arg_tuples: List[Tuple],
    max_workers: Optional[int] = 4,
    backend: Literal["thread", "process"] = "thread",
    return_exceptions: bool = False,
    timeout: Optional[float] = None,
    desc: str = "Processing"
) -> List[Any]:
    """
    并行执行: 对于列表中的每个 Tuple 作为 *args 调用 fn(*args)，并带 tqdm 进度条。
    """

    Executor = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor

    results: List[Any] = [None] * len(arg_tuples)

    with Executor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, *args): idx for idx, args in enumerate(arg_tuples)}

        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i = futures[fut]
            try:
                results[i] = fut.result(timeout=timeout)
            except Exception as e:
                if return_exceptions:
                    results[i] = e
                else:
                    for other in futures:
                        other.cancel()
                    raise

    return results

def l1_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.sum(x)
    return x / (s + eps)

def cosine_sim(mat: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # mat: [n, dim], q: [dim]
    qn = q / (np.linalg.norm(q) + eps)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + eps)
    return (mn @ qn).astype(np.float32)  # in [-1,1]

def calculate_NDCG(
        rank_list: List[str],
        relevance_scores: Dict[str, float],
        k: int = None,
        method: str = "linear"
) -> float:
    """
    计算 NDCG@k，包含去重预处理。

    Args:
        rank_list: 预测列表。允许包含重复元素，函数内部会自动去重并保留首位。
        relevance_scores: 真实分数。
        k: 截断位置。
        method: 'linear' (推荐用于0-1分数) 或 'exponential'。
    """

    # --- 1. 去重逻辑 (保留排位最大值/首次出现) ---
    # 使用 dict.fromkeys() 是 Python 3.7+ 保留顺序去重的最快方式
    # 例如：['A', 'B', 'A', 'C'] -> ['A', 'B', 'C']
    # 'A' 的第二次出现（低排位）被丢弃，保留了第一次出现（高排位）
    rank_list = list(dict.fromkeys(rank_list))

    # --- 2. 参数处理 ---
    if k is None:
        k = len(rank_list)

    # 截断 (必须在去重之后进行)
    rank_list = rank_list[:k]

    # --- 3. 内部函数：计算 DCG ---
    def get_dcg(scores: List[float]) -> float:
        dcg_val = 0.0
        for i, rel in enumerate(scores):
            if rel <= 0:
                continue

            if method == "exponential":
                gain = 2 ** rel - 1
            else:
                gain = rel

            # i+2 因为排名从1开始，log(rank+1) -> log(i+1+1)
            dcg_val += gain / math.log2(i + 2)
        return dcg_val

    # --- 4. 计算实际 DCG ---
    # 获取预测结果对应的分数
    predicted_scores = [relevance_scores.get(doc, 0.0) for doc in rank_list]
    actual_dcg = get_dcg(predicted_scores)

    # --- 5. 计算理想 IDCG ---
    # 从所有真实分数中取 Top-K
    all_ideal_scores = sorted(relevance_scores.values(), reverse=True)
    ideal_top_k_scores = all_ideal_scores[:k]
    ideal_dcg = get_dcg(ideal_top_k_scores)

    # --- 6. 归一化 ---
    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg