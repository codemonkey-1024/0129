# kg_rl/reward_funcs.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from typing import Callable, Dict, Any, List, Tuple
import math
import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

from core.heat_diffusion import compute_field
from core.schema import GraphPath
from core.utils.tools import calculate_NDCG
from config import time_statistic


# ============================================================
# 类型定义
# ============================================================

RewardDetails = Dict[str, float]

BatchedRewardFn = Callable

def _score_global_diffusion(
        idx: Any,
        qitem: Dict[str, Any],
        id2score: Dict[Any, float],
        top_k_docs: int = 20,
) -> Tuple[float, RewardDetails]:
    """
    运行扩散算法 -> 计算文档 NDCG -> 应用立方锐化
    """
    gold_ent_ids = [ent[0]['id'] for ent in qitem['gold_ents']]
    gold_ent_recall = len(id2score.keys() & gold_ent_ids) / len(gold_ent_ids)

    iv_reward = sum([qitem['ent_iv'].get(id, 0.0) for id in id2score.keys()])

    rank_score = get_rank_score(idx, qitem, id2score, top_k_docs)
    final_reward = iv_reward + rank_score

    return float(final_reward), {"iv_reward": float(iv_reward), "ent_recall": float(gold_ent_recall), "rank_score": float(rank_score)}

def get_rank_score(
        idx: Any,
        qitem: Dict[str, Any],
        id2score: Dict[Any, float],
        top_k_docs: int = 20):
    try:
        heat = compute_field(idx.EP_ctx, id2score)
        sorted_ids = [id for id, v in sorted(heat.items(), key=lambda x: x[1], reverse=True)]
        gold_docs = qitem['gold_docs']
        ndcg = calculate_NDCG(sorted_ids, {doc['id']: 1.0 for doc in gold_docs})
        return ndcg
    except Exception as e:
        return 0.0


def make_traj_reward_fn(
        idx: Any,
        q2qitem: Dict[str, Dict[str, Any]],
        top_k_docs: int = 20,
) -> BatchedRewardFn:
    def reward_fn(
            query: str,
            paths: List[GraphPath]
    ) -> Tuple[float, RewardDetails]:

        qitem = q2qitem.get(query)

        # --- A. 聚合种子向量 (Seed Aggregation) ---
        id2score_global = defaultdict(float)
        id2count_global = defaultdict(int)

        for path in paths:
            # 给每条路径的第一个节点赋初始得分
            id2score_global[path.nodes[0]['id']] += 1.0
            id2count_global[path.nodes[0]['id']] += 1

            # 对路径中的每个节点加权得分
            for node, score in zip(path.nodes[1:], path.scores):
                id2score_global[node['id']] += score
                id2count_global[node['id']] += 1

        # 对每个节点的得分进行归一化
        id2score_global = {id: id2score_global[id] / id2count_global[id] for id in id2score_global}

        # --- B. 计算全局 NDCG (内部已包含立方锐化) ---
        R_global, details = _score_global_diffusion(
            idx, qitem, id2score_global, top_k_docs
        )

        # 返回计算得到的全局奖励和细节
        return float(R_global), details

    return reward_fn




