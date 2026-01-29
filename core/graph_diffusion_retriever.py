# -*- coding: utf-8 -*-
"""
基于二部图的扩散式文档检索器

包含：
- BaseGraphDiffusionRetriever: 抽象基类（支持在一个 personalization 中同时传入实体 / 文档权重）
- BiRankRetriever: 使用 BiRank 算法在 E-P 二部图上做文档排序
- PPRRetriever: 使用 networkx.pagerank 的 PPR 检索器
- CoHITSRetriever: 使用 Co-HITS 算法在 E-P 二部图上做文档排序
- HITSRetriever: 使用经典 HITS 算法在 E-P 二部图上计算 hub/authority
- PPRRetrieverFAST: 使用稀疏矩阵幂迭代实现的加速版 PPR
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import networkx as nx
from config import RunningConfig
from core.ragindex import RagIndex
import torch


# ============================================================
# 抽象基类：BaseGraphDiffusionRetriever
# ============================================================

class BaseGraphDiffusionRetriever(ABC):
    """
    在 E-P 图上，给定一个 personalization（其中可以同时包含实体节点和文档节点的权重），
    将“重要性”扩散到文档节点并返回 Top-K 文档节点属性。

    说明：
      - personalization: {node_id -> weight}，node_id 可以是实体 ID 或文档 ID
      - 在构造实体 / 文档种子向量时，会对整个 personalization 做一次统一归一化，
        然后再拆分到实体侧 u0 和文档侧 v0，而不是分别归一化。
    """

    def __init__(self, cfg: RunningConfig, idx: RagIndex):
        self.cfg = cfg
        self.idx = idx

        if idx.doc_embedding is None or idx.entity_embedding is None:
            raise RuntimeError("GraphIndex 的 embedding 尚未构建，请先调用 GraphIndex.index(...)。")

        # 二部图：实体节点 + 文档节点
        self.G_ep: nx.Graph = idx.EP_graph

        # 明确实体 / 文档节点 ID 列表（来自 embedding 对象）
        self.doc_ids: List[str] = idx.doc_embedding.ids
        self.ent_ids: List[str] = idx.entity_embedding.ids

        self.num_docs = len(self.doc_ids)
        self.num_ents = len(self.ent_ids)

        # 映射：id -> 索引（用于构造矩阵、向量）
        self.doc_id2idx = {did: i for i, did in enumerate(self.doc_ids)}
        self.ent_id2idx = {eid: i for i, eid in enumerate(self.ent_ids)}

    # ---------- 对外统一接口 ----------

    def retrieve_for_bi_graph(
        self,
        personalization: Optional[Dict[str, float]],
        *,
        top_k_docs: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        输入：
        - personalization: 节点权重 {node_id -> weight}，其中 node_id 可以是实体或文档

        输出：
        - Top-K 文档节点属性（含 "score"）
        """
        # 由子类实现具体扩散算法，得到文档得分向量
        doc_scores = self._compute_doc_scores(personalization)

        # 将得分与 doc_ids 绑定并排序
        ranked = sorted(
            [(doc_id, float(doc_scores[self.doc_id2idx[doc_id]])) for doc_id in self.doc_ids],
            key=lambda x: x[1],
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        for doc_id, score in ranked[:top_k_docs]:
            node_attr = dict(self.G_ep.nodes[doc_id])
            node_attr["score"] = score
            results.append(node_attr)

        return results

    # ---------- 子类必须实现的内部方法 ----------

    @abstractmethod
    def _compute_doc_scores(
        self,
        personalization: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """
        给定（可能同时包含实体和文档节点的） personalization，
        返回一个长度为 num_docs 的 numpy 向量，表示每个文档节点的得分。
        """
        raise NotImplementedError

    # ---------- 公共辅助函数：从统一 personalization 中拆分实体 / 文档种子 ----------

    def _split_personalization(
        self,
        personalization: Optional[Dict[str, float]],
        *,
        default_ent_uniform: bool = True,
        default_doc_uniform: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将一个统一的 personalization（实体 + 文档）拆分为：
          - 实体侧向量 u0（shape = [num_ents]）
          - 文档侧向量 v0（shape = [num_docs]）

        归一化规则：
          - 先在“整体上”对 personalization 做归一化（只考虑出现在实体 / 文档映射中的节点），
            再把归一化后的质量分别加到 u0 / v0 中。
          - 即：sum_over_entities_and_docs(weight) = 1，然后 u0、v0 各自拿到这 1 里的“份额”，
            而不是分别各自归一化为 1。

        回退规则：
          - 如果实体侧在 personalization 中完全缺失且 default_ent_uniform=True，
            则 u0 退回实体均匀分布；
          - 如果文档侧在 personalization 中完全缺失且 default_doc_uniform=True，
            则 v0 退回文档均匀分布。
        """
        u0 = np.zeros(self.num_ents, dtype=np.float32)
        v0 = np.zeros(self.num_docs, dtype=np.float32)

        if personalization:
            # 只考虑图中存在的实体 / 文档节点，且权重 > 0
            total = 0.0
            for nid, w in personalization.items():
                val = float(w)
                if val <= 0.0:
                    continue
                if (nid in self.ent_id2idx) or (nid in self.doc_id2idx):
                    total += val

            if total > 0.0:
                inv_total = 1.0 / total
                for nid, w in personalization.items():
                    val = float(w)
                    if val <= 0.0:
                        continue
                    val *= inv_total

                    ei = self.ent_id2idx.get(nid)
                    if ei is not None:
                        u0[ei] += val

                    dj = self.doc_id2idx.get(nid)
                    if dj is not None:
                        v0[dj] += val

        # 若某一侧没有任何个性化质量，则可退回均匀分布
        if u0.sum() <= 0.0 and default_ent_uniform and self.num_ents > 0:
            u0[:] = 1.0 / float(self.num_ents)

        if v0.sum() <= 0.0 and default_doc_uniform and self.num_docs > 0:
            v0[:] = 1.0 / float(self.num_docs)

        return u0, v0


# ============================================================
# BiRank 实现：BiRankRetriever
# ============================================================

class BiRankRetriever(BaseGraphDiffusionRetriever):
    """
    使用 BiRank 在 E-P 二部图上做文档排序的检索器。

    核心迭代公式（简化版 BiRank / Co-HITS）：
        u^{t+1} = (1 - alpha) * u0 + alpha * S_ed @ v^{t}
        v^{t+1} = (1 - beta)  * v0 + beta  * S_de @ u^{t}

    其中：
        - u: 实体侧得分向量（size = num_ents）
        - v: 文档侧得分向量（size = num_docs）
        - S_de, S_ed: 由 E-P 二部图构造的归一化矩阵

    这里：
        - personalization 中同时可以出现实体 ID 和文档 ID，
        - 归一化在整个 personalization 上做一次，然后拆到 u0/v0。
    """

    def __init__(
        self,
        cfg: RunningConfig,
        idx: RagIndex,
        *,
        alpha: float = 0.85,
        beta: float = 0.85,
        max_iter: int = 20,
        tol: float = 1e-6,
    ):
        super().__init__(cfg, idx)

        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol

        # 预先构造 E-P 二部图的稀疏矩阵以及归一化矩阵
        self._build_bipartite_matrices()

    # ---------- 构造二部图矩阵 ----------

    def _build_bipartite_matrices(self):
        """
        从 G_ep 构造实体-文档二部矩阵 W（实体为行，文档为列），
        再基于度信息构造归一化矩阵 S_de, S_ed。

        W_ij 表示实体 i 与文档 j 之间的边权（默认 1）。
        """
        rows = []
        cols = []
        data = []

        for u, v, attrs in self.idx.EP_graph.edges(data=True):
            # 只考虑 实体-文档 之间的边
            if u in self.ent_id2idx and v in self.doc_id2idx:
                ei = self.ent_id2idx[u]
                dj = self.doc_id2idx[v]
            elif u in self.doc_id2idx and v in self.ent_id2idx:
                ei = self.ent_id2idx[v]
                dj = self.doc_id2idx[u]
            else:
                # 不是 E-P 边（比如 E-E 或 P-P），在此忽略
                continue

            w = float(attrs.get("weight", 1.0))
            rows.append(ei)
            cols.append(dj)
            data.append(w)

        if len(data) == 0:
            raise RuntimeError("G_ep 中没有实体-文档边，无法构造 BiRank 矩阵。")

        # 实体 x 文档 的二部矩阵 W
        W = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_ents, self.num_docs),
            dtype=np.float32,
        )

        # 度：实体侧、文档侧
        ent_degree = np.asarray(W.sum(axis=1)).reshape(-1)  # shape = [num_ents]
        doc_degree = np.asarray(W.sum(axis=0)).reshape(-1)  # shape = [num_docs]

        # 避免除零
        ent_degree[ent_degree == 0] = 1.0
        doc_degree[doc_degree == 0] = 1.0

        # D^{-1/2} 形式的对称归一化
        ent_inv_sqrt = 1.0 / np.sqrt(ent_degree)
        doc_inv_sqrt = 1.0 / np.sqrt(doc_degree)

        # 对行列进行缩放： S_de = D_d^{-1/2} * W^T * D_e^{-1/2}
        #   实际上，我们需要两个矩阵：
        #   - S_de: 文档 <- 实体（docs x ents）
        #   - S_ed: 实体 <- 文档（ents x docs）
        #
        # 这里实现的是对称归一化版本：
        #   S_de = D_d^{-1/2} * W^T * D_e^{-1/2}
        #   S_ed = D_e^{-1/2} * W    * D_d^{-1/2}
        #
        # 注意：scipy.sparse 支持对行/列乘对角向量的广播。

        # 实体侧缩放：W_scaled_rows = D_e^{-1/2} * W
        W_scaled_rows = W.multiply(ent_inv_sqrt[:, None])          # (ents x docs)
        # 文档侧缩放：W_scaled_cols = W * D_d^{-1/2}
        W_scaled_cols = W.multiply(doc_inv_sqrt[None, :])          # (ents x docs)

        # S_ed: 实体 <- 文档
        #   S_ed = D_e^{-1/2} * W * D_d^{-1/2}
        self.S_ed = W_scaled_rows.multiply(doc_inv_sqrt[None, :])  # (ents x docs)

        # S_de: 文档 <- 实体
        #   S_de = D_d^{-1/2} * W^T * D_e^{-1/2}
        self.S_de = W_scaled_cols.T.multiply(ent_inv_sqrt[None, :])  # (docs x ents)

    # ---------- 核心：BiRank 扩散算法 ----------

    def _compute_doc_scores(
        self,
        personalization: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """
        给定统一的 personalization，使用 BiRank 迭代计算文档得分。
        返回 shape = [num_docs] 的 numpy 向量。
        """
        # 从统一 personalization 中拆分出实体侧 u0 和文档侧 v0
        u0, v0 = self._split_personalization(
            personalization,
            default_ent_uniform=True,
            default_doc_uniform=True,
        )

        # 初始化 u, v
        u = u0.copy()
        v = v0.copy()

        for _ in range(self.max_iter):
            u_prev = u
            v_prev = v

            # v -> u 更新：实体收集文档信息
            # S_ed: (ents x docs); v: (docs)
            u = (1.0 - self.alpha) * u0 + self.alpha * (self.S_ed @ v_prev)

            # u -> v 更新：文档收集实体信息
            # S_de: (docs x ents); u: (ents)
            v = (1.0 - self.beta) * v0 + self.beta * (self.S_de @ u_prev)

            # 收敛判断：u 和 v 都变化很小则提前停止
            du = np.linalg.norm(u - u_prev, ord=1)
            dv = np.linalg.norm(v - v_prev, ord=1)
            if du < self.tol and dv < self.tol:
                break

        # 最终返回文档侧得分 v
        if v.sum() > 0:
            v = v / v.sum()

        return v


class PPRRetriever(BaseGraphDiffusionRetriever):
    """
    使用 networkx.pagerank 在 E-P 图上做 Personalized PageRank，
    将节点个性化权重扩散到文档节点。

    这里的 personalization 可以包含任意图节点（实体 / 文档 / 其它）的权重；
    PageRank 的个性化向量在 networkx 内部会做全局归一化。
    """

    def __init__(self, cfg: RunningConfig, idx: RagIndex, alpha: float = 0.85):
        super().__init__(cfg, idx)
        self.alpha = alpha

    # --------------------------
    # 核心：返回文档侧 score 向量
    # --------------------------
    def _compute_doc_scores(
        self,
        personalization: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """
        输入：
          - personalization: {node_id: weight}，可以是实体 / 文档等任意节点
        输出：
          - 长度 num_docs 的 numpy 向量（按 doc_ids 顺序）
        """
        personalization_nx = None

        if personalization:
            # 只保留图中存在的节点；networkx 会自动归一化成概率分布
            tmp: Dict[str, float] = {}
            for nid, w in personalization.items():
                if nid in self.G_ep:
                    tmp[nid] = tmp.get(nid, 0.0) + float(w)
            if tmp:
                personalization_nx = tmp

        # 直接在 E-P 图上做 PPR
        pr = nx.pagerank(
            self.G_ep,
            personalization=personalization_nx,
            alpha=self.alpha,
        )

        # 只提取文档节点得分
        scores = np.zeros(self.num_docs, dtype=np.float32)
        for doc_id, idx in self.doc_id2idx.items():
            scores[idx] = float(pr.get(doc_id, 0.0))

        # 归一化（可选）
        if scores.sum() > 0:
            scores /= scores.sum()

        return scores


class CoHITSRetriever(BaseGraphDiffusionRetriever):
    """
    使用 Co-HITS 算法在 E-P 二部图上做文档排序。

    简化版 Co-HITS 迭代（实体 = hub，文档 = authority）：
        u^{t+1} = (1 - alpha) * u0 + alpha * P_ed @ v^{t}
        v^{t+1} = (1 - beta)  * v0 + beta  * P_de @ u^{t+1}

    其中：
        - u: 实体节点得分向量（长度 num_ents）
        - v: 文档节点得分向量（长度 num_docs）
        - P_ed: 从文档到实体的归一化矩阵（ents x docs，按列归一化）
        - P_de: 从实体到文档的归一化矩阵（docs x ents，按列归一化）
        - u0, v0: 由统一 personalization 拆分得到的实体 / 文档种子向量
    """

    def __init__(
        self,
        cfg: RunningConfig,
        idx: RagIndex,
        *,
        alpha: float = 0.85,
        beta: float = 0.85,
        max_iter: int = 20,
        tol: float = 1e-6,
    ):
        super().__init__(cfg, idx)

        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol

        # 构造二部图的 W, P_de, P_ed
        self._build_bipartite_matrices()

    # ---------- 构造 Co-HITS 用的二部图矩阵 ----------

    def _build_bipartite_matrices(self):
        """
        从 G_ep 构造实体-文档二部矩阵 W（实体为行，文档为列），
        再基于度信息构造列归一化矩阵：
            - P_ed: 从文档到实体（ents x docs）
            - P_de: 从实体到文档（docs x ents）
        """
        rows = []
        cols = []
        data = []

        # 仅使用实体-文档之间的边
        for u, v, attrs in self.idx.EP_graph.edges(data=True):
            if u in self.ent_id2idx and v in self.doc_id2idx:
                ei = self.ent_id2idx[u]
                dj = self.doc_id2idx[v]
            elif u in self.doc_id2idx and v in self.ent_id2idx:
                ei = self.ent_id2idx[v]
                dj = self.doc_id2idx[u]
            else:
                # 非 E-P 边（比如 E-E / P-P），忽略
                continue

            w = float(attrs.get("weight", 1.0))
            rows.append(ei)
            cols.append(dj)
            data.append(w)

        if len(data) == 0:
            raise RuntimeError("G_ep 中没有实体-文档边，无法构造 Co-HITS 矩阵。")

        # W：实体 x 文档
        W = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_ents, self.num_docs),
            dtype=np.float32,
        )

        # ---------- 文档侧 / 实体侧度 ----------
        ent_degree = np.asarray(W.sum(axis=1)).reshape(-1)  # [num_ents]
        doc_degree = np.asarray(W.sum(axis=0)).reshape(-1)  # [num_docs]

        ent_degree[ent_degree == 0.0] = 1.0
        doc_degree[doc_degree == 0.0] = 1.0

        inv_ent_deg = 1.0 / ent_degree
        inv_doc_deg = 1.0 / doc_degree

        # ---------- 列归一化 ----------
        # P_ed: 从文档到实体（ents x docs）
        #   对 W 的每一列 j 做归一化：P_ed[:, j] = W[:, j] / deg_doc[j]
        self.P_ed = W.multiply(inv_doc_deg[None, :])          # ents x docs

        # P_de: 从实体到文档（docs x ents）
        #   对 W^T 的每一列 i（实体）做归一化：P_de[:, i] = W^T[:, i] / deg_ent[i]
        self.P_de = W.T.multiply(inv_ent_deg[None, :])        # docs x ents

    # ---------- 内部：Co-HITS 迭代 ----------

    def _compute_doc_scores(
        self,
        personalization: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """
        输入：
          - personalization: 统一的节点个性化权重 {node_id: weight}
        输出：
          - 长度 num_docs 的文档得分向量（按 self.doc_ids 顺序）
        """
        # 从统一 personalization 中拆分实体种子 u0、文档先验 v0
        u0, v0 = self._split_personalization(
            personalization,
            default_ent_uniform=True,
            default_doc_uniform=True,
        )

        # 初始化 u, v
        u = u0.copy()
        v = v0.copy()

        for _ in range(self.max_iter):
            u_prev = u
            v_prev = v

            # v -> u：实体从文档收集权重
            # P_ed: (ents x docs); v_prev: (docs)
            u = (1.0 - self.alpha) * u0 + self.alpha * (self.P_ed @ v_prev)

            # u -> v：文档从实体收集权重
            # P_de: (docs x ents); 这里用 u（当前步）做更新
            v = (1.0 - self.beta) * v0 + self.beta * (self.P_de @ u)

            # 收敛检测（L1 范数变化）
            du = np.linalg.norm(u - u_prev, ord=1)
            dv = np.linalg.norm(v - v_prev, ord=1)
            if du < self.tol and dv < self.tol:
                break

        # 归一化一下文档得分（非必需，但通常更稳定）
        if v.sum() > 0:
            v = v / v.sum()

        return v



# =========================
# PPR 幂迭代
# =========================
def ppr_power_iteration(
    Tt: sp.csr_matrix,
    personalization: np.ndarray,
    *,
    alpha: float = 0.85,
    dangling_mask: Optional[np.ndarray] = None,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    解 PageRank 方程：
        r = alpha * T^T * r + (1 - alpha) * p + alpha * dangling_mass * p

    其中：
      - T 是行随机矩阵（每行出边分布），Tt = T^T
      - p 是个性化向量（会在这里归一化为概率分布）
      - dangling_mass = sum_{i 是悬空节点} r_i
      - 返回的 r 是一个概率分布（L1 范数为 1）
    """
    n = Tt.shape[0]
    # 归一化 p（如果全 0，则退回均匀分布）
    p = personalization.astype(np.float64, copy=True)
    s = p.sum()
    if s > 0:
        p /= s
    else:
        p[:] = 1.0 / n

    # 初始向量：均匀分布（与 networkx.pagerank 默认一致）
    r = np.full(n, 1.0 / n, dtype=np.float64)

    has_dangling = bool(dangling_mask is not None and np.any(dangling_mask))
    beta = 1.0 - alpha

    for _ in range(max_iters):
        # 基本 PageRank 更新
        r_new = alpha * (Tt @ r) + beta * p

        # 悬空节点：把其质量按 p 分布回流
        if has_dangling:
            r_new += alpha * r[dangling_mask].sum() * p

        # 收敛检测（L1 范数，与 networkx 一致的思路）
        if np.linalg.norm(r_new - r, 1) < tol:
            # 再归一化一次，防止累积误差
            r_new /= r_new.sum()
            return r_new

        r = r_new

    # max_iters 内没达到 tol，也返回当前值（与 networkx 语义一致）
    r /= r.sum()
    return r


class PPRRetrieverFAST:
    """
    使用 E-P 图上的 PPR 做文档召回的加速实现（稀疏矩阵 + 幂迭代）。

    目标：在输入/输出和数学语义上，都与下面调用完全等价：

        nx.pagerank(self.G_ep, personalization=personalization, alpha=0.85)

    其中：
      - personalization: dict[node_id -> weight] 或 None
      - 我们只在 doc_ids 上取 Top-K 结果，但 PageRank 计算覆盖 G_ep 的所有节点。

    这里的 personalization 与 PPRRetriever 保持一致：
      - 可以同时包含实体 / 文档 / 其它节点；
      - 会在幂迭代内部做统一归一化。
    """

    def __init__(self, cfg: RunningConfig, graph_index: RagIndex):
        self.cfg = cfg
        self.graph_index = graph_index

        if graph_index.doc_embedding is None or graph_index.entity_embedding is None:
            raise RuntimeError("GraphIndex 的 embedding 尚未构建，请先调用 GraphIndex.index(...)。")

        self.G_ep: nx.Graph = graph_index.EP_graph
        self.doc_ids: List[str] = graph_index.doc_embedding.ids
        self.ent_ids: List[str] = graph_index.entity_embedding.ids

        self.doc_id2idx = {did: i for i, did in enumerate(self.doc_ids)}
        self.ent_id2idx = {eid: i for i, eid in enumerate(self.ent_ids)}

        # 构建转移矩阵
        self._build_transition_matrices()

    # ---------- 构建 E-P 图上的 T / Tᵀ ----------

    def _build_transition_matrices(self) -> None:
        """
        基于 E-P 图一次性构建：
          - nodes_all：G_ep 中的所有节点（与 networkx.pagerank 完全一致）
          - CSR 邻接 A：
              * 有向图 -> 只保留原有方向
              * 无向图 -> 无向边视作双向（u->v, v->u）
          - 行随机转移矩阵 T 及其转置 Tt
          - 悬空掩码 dangling_mask
        """
        # 1) 所有节点（顺序固定）
        self.nodes_all: List[Any] = list(self.G_ep.nodes())
        self.nid2idx: Dict[Any, int] = {nid: i for i, nid in enumerate(self.nodes_all)}
        n = len(self.nodes_all)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        is_directed = self.G_ep.is_directed()

        # 2) 构造邻接矩阵（带权重）
        for u, v, d in self.G_ep.edges(data=True):
            w = float(d.get("weight", 1.0))
            iu = self.nid2idx[u]
            iv = self.nid2idx[v]

            if is_directed:
                # 有向图：保留方向 u -> v
                rows.append(iu)
                cols.append(iv)
                data.append(w)
            else:
                # 无向图：视为双向 u<->v
                rows.extend([iu, iv])
                cols.extend([iv, iu])
                data.extend([w, w])

        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

        # 3) 归一化为行随机矩阵 T
        out = np.asarray(A.sum(axis=1)).ravel()
        self.dangling_mask = (out == 0)
        # 防止除零：让悬空节点的 out 度暂时视为 1（后续用 dangling_mask 单独处理）
        out[self.dangling_mask] = 1.0

        self.T: sp.csr_matrix = A.multiply(1.0 / out[:, None]).tocsr()
        self.Tt: sp.csr_matrix = self.T.transpose().tocsr()

    # ---------- personalization(dict/None) → 向量 ----------

    def _personalization_to_vector(self, personalization: Optional[Dict[Any, float]]) -> np.ndarray:
        """
        将 personalization 映射到与 nodes_all 对齐的向量 p：

          - 如果 personalization 为 None 或空：等价于均匀分布（在幂迭代内部处理）
          - 否则：只对图中存在的节点赋值，其余忽略，后续统一归一化。
        """
        n = len(self.nodes_all)
        p = np.zeros(n, dtype=np.float64)

        if not personalization:
            # personalization=None -> 均匀分布
            # 这里先返回全 0，后续会在 ppr_power_iteration 中退回均匀分布
            return p

        for nid, w in personalization.items():
            idx = self.nid2idx.get(nid)
            if idx is not None:
                p[idx] += float(w)

        return p

    # ---------- PPR ----------

    def _run_ppr(
        self,
        personalization_vec: np.ndarray,
        *,
        alpha: float = 0.85,
        max_iters: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        输入：personalization 向量（未归一化也可以）
        输出：PageRank 向量 r（在数值上与 networkx.pagerank 一致）
        """
        r = ppr_power_iteration(
            self.Tt,
            personalization_vec,
            alpha=alpha,
            dangling_mask=self.dangling_mask,
            max_iters=max_iters,
            tol=tol,
        )
        return r

    # ---------- personalization(dict/None) → 文档 Top-K ----------

    def retrieve_from_entities(
        self,
        personalization: Optional[Dict[Any, float]],
        *,
        top_k_docs: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        输入：
          - personalization: 节点权重（node_id -> weight），或 None

        输出：
          - Top-K 文档节点属性（含 "score"），语义对齐 networkx.pagerank
        """
        # 1) dict/None -> 向量
        p_vec = self._personalization_to_vector(personalization)

        # 2) PPR
        r = self._run_ppr(p_vec, alpha=0.85, max_iters=100, tol=1e-6)

        # 3) 只取文档节点的得分（与 PPRRetriever 保持一致）
        scores: List[tuple] = []
        for did in self.doc_ids:
            if did not in self.nid2idx:
                continue
            idx = self.nid2idx[did]
            scores.append((did, float(r[idx])))

        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:top_k_docs]

        results: List[Dict[str, Any]] = []
        for doc_id, score in scores:
            node_attr = dict(self.G_ep.nodes[doc_id])
            node_attr["score"] = score
            results.append(node_attr)

        return results
