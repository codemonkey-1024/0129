# kg_rl/env.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from torch.distributions import Categorical

from core.schema import GraphPath
from core.ragindex import RagIndex
from .policy import Policy
from config import TrainingConfig, logger
from config import *
from tqdm import tqdm


# =========================================================================
# 2. 图环境 (Graph Environment)
# =========================================================================
@dataclass
class StepState:
    """环境的单步状态。**已增加 candidates 字段用于缓存**。"""
    query: str
    path: GraphPath
    depth: int
    done: bool = False
    # 缓存：当前状态下所有可能的下一步路径 (GraphPath)
    candidates: List[GraphPath] = field(default_factory=list)


@dataclass
class Trajectory:
    """完整的一条采样轨迹"""
    start_node: Dict
    query: str

    # 核心序列数据
    states: List[StepState] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)

    # 策略相关的 Log数据 (用于 PPO update)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    entropies: List[torch.Tensor] = field(default_factory=list)

    # 分析与统计数据
    step_cand_probs: List[List[Tuple[str, float]]] = field(default_factory=list)
    chosen_step_probs: List[float] = field(default_factory=list)
    logp_sum: float = 0.0

    # 结果与奖励 (由 Trainer 后续填充)
    final_path: Optional[GraphPath] = None
    reward: Optional[float] = None
    R_q: Optional[float] = None
    R_path: Optional[float] = None
    reward_details: Optional[Dict] = None

    # 前缀分数 (用于 Reward Shaping)
    prefix_scores: Optional[List[float]] = None
    prefix_delta_scores: Optional[List[float]] = None


class GraphEnv:
    """
    基于 RAG 索引的图遍历环境。
    负责：状态管理、邻居节点扩展 (Candidates)、启发式剪枝。
    """

    def __init__(self, cfg: TrainingConfig, idx: RagIndex, max_depth: int = 4, avoid_cycles: bool = True):
        self.cfg = cfg
        self.idx = idx
        self.G_ee = idx.EE_graph  # 实体-实体 关系图
        self.G_ep = idx.EP_graph  # 实体-文档 共现图
        self.max_depth = max_depth
        self.avoid_cycles = avoid_cycles
        self.explore_sources = cfg.explore_sources
        self.max_cand = cfg.max_cand

        self.ee_neighbors_out = defaultdict(list)
        self.ee_neighbors_in = defaultdict(list)
        self._build_ee_neighbors()

        self.ep_neighbors_out = defaultdict(list)
        self.ep_neighbors_in = defaultdict(list)
        self.co_cur_neighbors = defaultdict(list)
        self._build_ep_neighbors()

    def _build_ee_neighbors(self):
        """构建图的邻接表，只在图结构变化时调用一次"""
        for u, v, attr in self.G_ee.edges(data=True):
            self.ee_neighbors_out[u].append((v, attr))
            self.ee_neighbors_in[v].append((u, attr))

    def _build_ep_neighbors(self, max_p_degree=64):
        """构建 EP 图的邻接表，只在图结构变化时调用一次"""
        for u, v in self.G_ep.edges():
            # 构建实体 -> 文档 出边
            self.ep_neighbors_out[u].append((v))
            # 构建文档 -> 实体 入边
            self.ep_neighbors_in[v].append((u))

        for e in tqdm([node[0] for node in self.G_ep.nodes(data=True) if node[1]['type'] == 'E'],
                      desc="building co-cur neighbors cache"):
            p_neighbors = self.ep_neighbors_out.get(e, []) + self.ep_neighbors_in.get(e, [])
            for p in p_neighbors:
                co_cur_es = self.ep_neighbors_out.get(p, []) + self.ep_neighbors_in.get(p, [])
                doc2ent = [(p, e) for e in co_cur_es]
                self.co_cur_neighbors[e].extend(list(set(doc2ent)))


    def reset(self, start_node: Dict, query: str, warm_start: Optional[GraphPath] = None) -> StepState:
        """重置环境到初始状态，并计算初始候选集"""
        path = warm_start.copy() if warm_start else GraphPath(start_node)
        depth = len(path.nodes) - 1

        initial_state = StepState(query=query, path=path, depth=depth, done=False)
        # 初始状态也需要计算候选集，以便第一个动作的采样
        initial_state.candidates = self._calculate_candidates(initial_state)

        # 如果初始状态就已达最大深度，则标记 done=True
        if initial_state.depth >= self.max_depth:
            initial_state.done = True

        # 如果没有候选集，则标记 done=True (死胡同)
        if not initial_state.candidates and not initial_state.done:
            initial_state.done = True

        return initial_state

    def step(self, state: StepState, action_idx: int) -> StepState:
        """
        执行一步动作。

        Args:
            state: 当前状态 (包含已缓存的 candidates)。
            action_idx: 动作索引，对应 state.candidates 中的路径。

        Returns:
            新的状态 StepState，并计算了下一状态的 candidates。
        """
        # 1. 确定下一个路径
        if not state.candidates:
            # 这应该在采样时被捕获，但作为安全措施
            return state

        next_path = state.candidates[action_idx]
        new_depth = state.depth + 1

        # 2. 检查是否终止
        done = new_depth >= self.max_depth

        # 3. 构造新的状态
        new_state = StepState(query=state.query, path=next_path, depth=new_depth, done=done)

        # 4. 预计算下一状态的 candidates (仅在未终止时)
        if not done:
            new_state.candidates = self._calculate_candidates(new_state)

            # 如果下一状态是死胡同，则标记 done=True
            if not new_state.candidates:
                new_state.done = True

        return new_state

    # 将原来的 candidates 方法重命名，并在 reset/step 中调用
    def _calculate_candidates(self, state: StepState) -> List[GraphPath]:
        """核心逻辑：获取当前状态的所有合法下一步路径（经过剪枝）"""
        current_node = state.path.current_node

        s = time.time()
        query_emb = self.idx.query_embedding.gather([state.query])
        e = time.time()
        time_statistic['retrieve_time'].append(e - s)

        all_cands = []

        if "EE" in self.explore_sources:
            # 1. 扩展 EE 图谱邻居 (Entity-Relation-Entity)

            s = time.time()
            paths_ee = self._get_ee_neighbors(state.path)
            e = time.time()
            time_statistic["_get_ee_neighbors"].append(e - s)

            s = time.time()
            paths_ee = self._heuristic_prune(paths_ee, query_emb, self.max_cand, mode='relation')
            e = time.time()
            time_statistic['_heuristic_prune'].append(e - s)
            all_cands += paths_ee

        if "EP" in self.explore_sources:
            # 2. 扩展 EP 图谱邻居 (Entity-Doc-Entity)
            s = time.time()
            paths_ep = self._get_ep_neighbors(state.path)
            e = time.time()
            time_statistic["_get_ep_neighbors"].append(e - s)

            s = time.time()
            paths_ep = self._heuristic_prune(paths_ep, query_emb, self.max_cand, mode='entity_doc')
            e = time.time()
            time_statistic['EP_heuristic_prune'].append(e - s)
            all_cands += paths_ep

        # 3. 合并并按格式排序 (保持确定性)
        all_cands.sort(key=lambda p: p.format_string)
        return all_cands

    # --- 内部扩展逻辑 (保持不变) ---

    def _get_ee_neighbors(self, p: GraphPath) -> List[GraphPath]:
        """从 G_ee (KB) 扩展一跳"""
        node_id = p.current_node["id"]
        from_ids = {n["id"] for n in p.nodes} if self.avoid_cycles else set()
        cands = []

        # 1. Outgoing: Node -> Obj
        for neighbor_id, attr in self.ee_neighbors_out[node_id]:
            if neighbor_id in from_ids: continue
            neighbor_node = {"id": neighbor_id, **self.G_ee.nodes[neighbor_id]}
            rel_label = attr.get("relation", attr.get("r", "rel"))
            triple = {"begin": p.current_node, "r": rel_label, **attr, "end": neighbor_node}
            cands.append(p.copy().add_node(neighbor_node, triple))

        # 2. Incoming: Subj -> Node
        for neighbor_id, attr in self.ee_neighbors_in[node_id]:
            if neighbor_id in from_ids: continue
            neighbor_node = {"id": neighbor_id, **self.G_ee.nodes[neighbor_id]}
            rel_label = attr.get("relation", attr.get("r", "rel"))
            # 注意三元组方向：邻居 -> 当前
            triple = {"begin": neighbor_node, "r": rel_label, **attr, "end": p.current_node}
            cands.append(p.copy().add_node(neighbor_node, triple))

        return cands

    def _get_ep_neighbors(self, p: GraphPath) -> List[GraphPath]:
        """从 G_ep (Co-occurrence) 扩展两跳：Entity -> Doc -> Entity"""
        source_id = p.current_node["id"]
        from_ids = {n["id"] for n in p.nodes} if self.avoid_cycles else set()
        cands = []

        for doc_id, target_id in self.co_cur_neighbors.get(source_id, []):
            if target_id == source_id:
                continue  # 排除回头路
            if target_id in from_ids:
                continue  # 排除环路 (注意：这里是否包含 Doc 节点取决于 avoid_cycles 语义，通常只排斥 Entity)

            target_node = self.G_ep.nodes[target_id]
            doc_node = self.G_ep.nodes.get(doc_id)
            if target_node.get('type') != 'E':  # 确保终点是 Entity 类型
                continue

            target_node_dict = {"id": target_id, **target_node}

            # 构造虚拟边：Co-occurrence
            triple = {
                "begin": p.current_node,
                "r": "co-cur",
                "relation": "co-cur",
                "doc": doc_node,  # 记录中间文档信息
                "end": target_node_dict
            }
            cands.append(p.copy().add_node(target_node_dict, triple))

        return cands

    def _heuristic_prune(self, paths: List[GraphPath], query_emb: np.ndarray, top_k: int, mode: str) -> List[GraphPath]:
        """基于 Embedding 相似度对候选路径进行剪枝"""
        # 将查询嵌入加载到 GPU 上
        query_emb = torch.tensor(query_emb).to(self.cfg.device)

        if len(paths) <= top_k:
            return paths

        if mode == 'relation':
            r_ids = [p.relations[-1]['id'] for p in paths]
            s = time.time()
            emb_matrix = self.idx.triple_embedding.gather(r_ids)
            e = time.time()
            time_statistic["triple_embedding.gather"].append(e - s)

            s = time.time()
            emb_matrix = torch.tensor(emb_matrix).to(self.cfg.device)  # 将嵌入矩阵转移到 GPU 上
            scores = torch.matmul(emb_matrix, query_emb.T).flatten()
            e = time.time()
            time_statistic["matmul"].append(e - s)

        elif mode == 'entity_doc':
            end_ids = [p.current_node['id'] for p in paths]
            doc_ids = [p.relations[-1]['doc']['id'] for p in paths]

            end_embs = self.idx.entity_embedding.gather(end_ids)
            doc_embs = self.idx.doc_embedding.gather(doc_ids)

            # 加和融合并转移到 GPU
            combined_embs = torch.tensor(end_embs).to(self.cfg.device) + torch.tensor(doc_embs).to(self.cfg.device)
            scores = torch.matmul(combined_embs, query_emb.T).flatten()

        else:
            return paths  # 未知模式不剪枝

        # 取 TopK
        top_indices = torch.argsort(scores, descending=True)[:top_k]
        return [paths[i.item()] for i in top_indices]  # 转换为原始路径列表


# =========================================================================
# 3. 采样流程 (Rollout)
# =========================================================================
import time


def rollout(
        env: GraphEnv,
        policy: Policy,
        start_nodes: List[Dict],
        queries: List[str],
        warm_starts: Optional[List[GraphPath]] = None,
        greedy: bool = False,
        use_grad: bool = False,
        samples_per_start: int = 1,
        temperature: float = 1.5,
        **kwargs
) -> List[Trajectory]:
    """
    基于策略模型 `policy` 在图环境 `env` 中进行批量路径采样。

    Args:
        env: 图环境实例，负责状态转移和候选集计算。
        policy: 路径采样策略模型。
        start_nodes: 路径搜索的起始实体列表。
        queries: 对应的查询文本列表。
        warm_starts: 可选，已有的前缀路径，用于热启动。
        greedy: 是否采用贪婪策略 (argmax) 而非采样。
        use_grad: 是否启用梯度计算 (用于收集 On-policy 梯度，如 REINFORCE)。
        samples_per_start: 每个 (start_node, query) 对采样的轨迹数量。
        temperature: 采样温度，影响探索程度。

    Returns:
        List[Trajectory]: 完整的采样轨迹列表。
    """

    # 获取策略模型所在设备（CPU 或 GPU）
    device = next(policy.parameters()).device

    # 如果没有提供热启动路径，则默认为 None
    warm_starts = warm_starts or [None] * len(start_nodes)

    # 存储已完成的轨迹列表
    completed_trajs: List[Trajectory] = []

    # 1. 扁平化初始化：一次性生成所有待运行轨迹
    # 对每个起始节点和查询生成初始轨迹，并执行环境重置
    active_trajs: List[Trajectory] = [
        Trajectory(
            start_node=s, query=q,
            states=[env.reset(s, q, w)]  # 通过 env.reset 获得初始状态
        )
        for s, q, w in zip(start_nodes, queries, warm_starts)
        for _ in range(samples_per_start)  # 每个 (start_node, query) 对采样 samples_per_start 次
    ]

    # 2. 将初始状态已完成的轨迹移动到 completed_trajs
    next_active_trajs = []
    for traj in active_trajs:
        # 如果初始状态已经完成，直接添加到已完成的轨迹列表中
        if traj.states[-1].done:
            traj.final_path = traj.states[-1].path
            completed_trajs.append(traj)
        else:
            next_active_trajs.append(traj)

    # 更新 active_trajs，移除已经完成的轨迹
    active_trajs = next_active_trajs

    # 3. 向量化循环：直到所有轨迹完成
    while active_trajs:
        # --- A. 构造 Batch 数据 ---
        # 提取当前需要策略推理的 (traj, state, candidates) 三元组
        batch_items: List[Tuple[Trajectory, StepState, List[GraphPath]]] = []

        for traj in active_trajs:
            state = traj.states[-1]  # 当前轨迹的最后一个状态
            candidates = state.candidates  # 当前状态下的候选路径集合

            # 如果状态已经完成，或者候选路径为空，说明这条轨迹已经终止
            if state.done or not candidates:
                traj.final_path = state.path
                completed_trajs.append(traj)
            else:
                batch_items.append((traj, state, candidates))  # 将需要继续采样的轨迹保存到 batch_items 中

        if not batch_items:
            break  # 所有轨迹都已完成，退出循环

        # --- B. 模型推理 (Policy Inference) ---
        # 使用 Python 的 zip(*) 技巧进行转置
        trajs, states, batch_cands = zip(*batch_items)
        batch_paths = [s.path for s in states]
        batch_queries = [s.query for s in states]

        # 启用/禁用梯度计算
        ctx = torch.enable_grad if use_grad else torch.no_grad

        s = time.time()
        with ctx():
            # 批量计算策略模型的 logits
            logits_batch = policy(batch_queries, list(batch_cands))
        e = time.time()
        time_statistic['policy'].append(e - s)
        # --- C. 批量采样与环境推进 ---
        next_active = []  # 用于存储下一轮需要继续采样的轨迹

        # 这里的 zip 完美对齐：input_item[i] 对应 logits[i]
        for (traj, state, cands), logits in zip(batch_items, logits_batch):
            # 异常保护：如果 logits 为零，说明此轨迹已终止
            if logits.numel() == 0:
                state.done = True
                traj.final_path = state.path
                completed_trajs.append(traj)
                continue

            # 动作采样 (使用温度和贪婪参数)
            if not greedy:
                logits = logits / temperature  # 温度调节，控制探索程度

            s = time.time()
            probs = F.softmax(logits, dim=-1)  # 计算动作的概率分布
            dist = Categorical(probs=probs)  # 创建一个类别分布对象

            # 如果使用贪婪策略，则选择最大概率的动作，否则进行采样
            if greedy:
                action_idx = int(torch.argmax(probs).item())  # 选择最大概率的动作
            else:
                action_idx = int(dist.sample().item())  # 从概率分布中采样

            e = time.time()
            time_statistic['sample_time'].append(e - s)

            # 记录策略数据
            traj.step_cand_probs.append([  # 记录每个候选动作的概率
                (c.current_node.get("mention", "?"), float(p))
                for c, p in zip(cands, probs)
            ])
            traj.chosen_step_probs.append(float(probs[action_idx]))  # 记录选择的动作的概率
            traj.actions.append(action_idx)  # 记录选择的动作

            # 记录 LogProb 和 Entropy (用于 PPO)
            log_p = dist.log_prob(torch.tensor(action_idx, device=device))  # 计算动作的对数概率
            traj.log_probs.append(log_p)  # 记录 log_prob
            traj.entropies.append(dist.entropy())  # 记录熵

            # 推进环境状态。新状态已经预计算了下一轮的候选集。
            s = time.time()
            new_state = env.step(state, action_idx)
            e = time.time()
            time_statistic['ent.step'].append(e - s)
            traj.states.append(new_state)

            if new_state.done:
                traj.final_path = new_state.path  # 如果新状态已完成，记录路径
                completed_trajs.append(traj)
            else:
                next_active.append(traj)  # 继续下一轮循环

        active_trajs = next_active  # 更新 active_trajs，继续采样未完成的轨迹

    # 4. 将采样概率作为节点分数
    for traj in completed_trajs:
        traj.final_path.scores = traj.chosen_step_probs  # 使用选择的动作概率作为路径的评分

    # 返回已完成的轨迹列表
    return completed_trajs
