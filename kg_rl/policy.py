# kg_rl/policy.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from core.schema import GraphPath
from kg_rl.path_scorer import ScorerAdapter


class Policy(nn.Module):
    """
    策略网络：负责对候选路径进行打分并计算动作概率分布。
    """

    def __init__(self, scorer: ScorerAdapter, temperature: float = 1.0):
        super().__init__()
        self.scorer = scorer

    def forward(
            self,
            queries: List[str],
            candidates: List[List[GraphPath]]
    ) -> List[torch.Tensor]:
        """
        核心计算逻辑：Flatten -> Batch Score -> Split
        """
        device = next(self.parameters()).device
        batch_size = len(queries)

        # 1. 展平 (Flattening)
        # 将嵌套的 candidates 列表展平为一维，以便进行批处理
        flat_candidates: List[GraphPath] = []
        flat_queries: List[str] = []
        split_sizes: List[int] = []

        for q, cands in zip(queries, candidates):
            n_cands = len(cands)
            split_sizes.append(n_cands)
            if n_cands > 0:
                flat_candidates.extend(cands)
                flat_queries.extend([q] * n_cands)

        if not flat_candidates:
            return [torch.empty(0, device=device) for _ in range(batch_size)]

        # 2. 批量打分 (Batch Scoring)
        raw_logits = self.scorer.score_paths(flat_candidates, flat_queries)


        # 确保是 1D Tensor
        raw_logits = raw_logits.view(-1)

        # 3. 还原与处理 (Split & Post-process)
        # 按记录的 split_sizes 切分回每个样本
        scores_list = torch.split(raw_logits, split_sizes)

        logits_batch: List[torch.Tensor] = []
        for scores in scores_list:
            logits_batch.append(scores)

        return logits_batch