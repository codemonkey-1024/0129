# retriever.py

from config import RunningConfig
from core.ragindex import RagIndex  # 如果不在同一目录，改成你的实际导入路径
from core.llm_functions import *
from core.schema import GraphPath
from loguru import logger
import numpy as np
from core.path_ranker import PathRanker
import networkx as nx
from typing import Any, List
from kg_rl.path_scorer import DynamicPathScorer



def two_hop_paths_with_attrs(G, source):
    """
    返回从 source 出发的所有“恰好两跳”的路径：
    每条是 (u, v, w) 三个节点，并且附带节点属性。

    返回格式示例：
    [
      {
        "path": (u, v, w),
        "attrs": (attrs_u, attrs_v, attrs_w)
      },
      ...
    ]
    """
    if source not in G:
        return []

    # 一跳邻居
    neighbors_1 = set(G.neighbors(source))
    # 为了判重：记录已加入的 (中间节点, 终点节点) 组合
    seen = set()
    results = []

    for v in neighbors_1:  # 第一跳
        for w in G.neighbors(v):  # 第二跳
            if w == source:
                # 回到源节点，路径长度还是2，但是通常不算“两跳到别的节点”，可按需要保留/去掉
                continue
            if w in neighbors_1:
                # w 是一跳邻居（直接邻居），用户要求只要“恰好两跳”的，
                # 通常不希望把一跳节点当作结果（否则可能存在 source-w 的边）
                continue

            key = (v, w)
            if key in seen:
                continue
            seen.add(key)

            # 取节点属性（如果没属性，G.nodes[x] 就是空 dict）
            attrs_u = dict(G.nodes[source])
            attrs_v = dict(G.nodes[v])
            attrs_w = dict(G.nodes[w])
            assert attrs_v['type'] == "P"
            results.append({
                "path": (source, v, w),
                "attrs": (attrs_u, attrs_v, attrs_w),
            })

    return results


# =========================
# BeamRetriever：query → 实体路径 & 实体集合
# =========================
class BeamRetriever:
    """
    使用实体嵌入 + E-E 图做 Beam Search：
      query → 起始实体 → 若干条实体路径 → 实体集合
    """

    def __init__(self, cfg: RunningConfig, index: RagIndex, scorer: DynamicPathScorer = None):
        self.cfg = cfg
        self.idx = index
        self.G_ee = index.EE_graph
        self.ent_vec_store = index.entity_embedding

        self.embedding_model = cfg.embedding_model
        self.llm = cfg.llm_model

        self.path_ranker = PathRanker(cfg, index, device=cfg.device, scorer=scorer, rank_mode=cfg.rank_mode)

        self.explore_sources = cfg.explore_sources


    def _expand_node_from_G_ee(self, p: GraphPath):
        from_ids = {n["id"] for n in p.nodes}
        node = p.current_node
        node_id = node["id"]
        expanded_paths: List[GraphPath] = []

        # 出边：node -> obj
        for _, obj_id, rel_attrs in self.G_ee.out_edges(node_id, data=True):
            if obj_id in from_ids:
                continue
            obj_node = {"id": obj_id, **self.G_ee.nodes[obj_id]}
            triple = {
                "begin": node,
                "r": rel_attrs.get("relation", rel_attrs.get("r", "rel")),
                **rel_attrs,
                "end": obj_node,
            }
            expanded_paths.append(p.copy().add_node(obj_node, triple))

        # 入边：subj -> node
        for subj_id, _, rel_attrs in self.G_ee.in_edges(node_id, data=True):
            if subj_id in from_ids:
                continue
            subj_node = {"id": subj_id, **self.G_ee.nodes[subj_id]}
            triple = {
                "begin": subj_node,
                "r": rel_attrs.get("relation", rel_attrs.get("r", "rel")),
                **rel_attrs,
                "end": node,
            }
            expanded_paths.append(p.copy().add_node(subj_node, triple))

        return expanded_paths

    def _expand_node_from_G_ep(self, p: GraphPath):
        from_ids = {n["id"] for n in p.nodes}
        node = p.current_node
        node_id = node["id"]
        expanded_paths: List[GraphPath] = []
        paths = two_hop_paths_with_attrs(self.idx.EP_graph, node_id)
        for path in paths:
            obj_node = path['attrs'][-1]
            if obj_node['id'] in from_ids and obj_node['type'] != "E":
                continue
            triple = {
                "begin": node,
                "r": 'co-cur',
                "relation": 'co-cur',
                'doc': path['attrs'][1],
                "end": obj_node,
            }
            expanded_paths.append(p.copy().add_node(obj_node, triple))
        return expanded_paths

    def _expand(self, p: GraphPath, query, expand_EP_Graph:bool = True, max_cand=8) -> List[GraphPath]:
        expanded_paths: List[GraphPath] = []

        query_embedding = self.idx.query_embedding.gather([query])


        if "EE" in self.explore_sources:
            # 从EE图谱中拓展
            paths_from_G_ee = self._expand_node_from_G_ee(p)
            if len(paths_from_G_ee) > max_cand:
                triples = [p.relations[-1] for p in paths_from_G_ee]
                r_ids = [tri['id'] for tri in triples]
                triple_embeddings = self.idx.triple_embedding.gather(r_ids)
                scores = (triple_embeddings @ query_embedding.T).flatten()
                top_k_indices = np.argsort(scores)[-max_cand:][::-1]
                paths_from_G_ee = [paths_from_G_ee[i] for i in top_k_indices]
            expanded_paths += paths_from_G_ee

        if "EP" in self.explore_sources:
            if expand_EP_Graph:
                paths_from_G_ep = self._expand_node_from_G_ep(p)
                if len(paths_from_G_ep) > max_cand:
                    end_node_ids = [p.current_node['id'] for p in paths_from_G_ep]
                    end_node_embeddings = self.idx.entity_embedding.gather(end_node_ids)
                    scores = (end_node_embeddings @ query_embedding.T).flatten()

                    doc_ids = [p.relations[-1]['doc']['id'] for p in paths_from_G_ep]
                    doc_embeddings = self.idx.doc_embedding.gather(doc_ids)
                    doc_scores = (doc_embeddings @ query_embedding.T).flatten()
                    scores = doc_scores + scores

                    top_k_indices = np.argsort(scores)[-max_cand:][::-1]
                    paths_from_G_ep = [paths_from_G_ep[i] for i in top_k_indices]
                expanded_paths += paths_from_G_ep

        return expanded_paths



    def greedy_search_per_seed(
            self,
            query: str,
            seeds: List[Any],
            max_depth: int = 3,
            samples_per_start: int = 1,
    ) -> List[GraphPath]:
        """
        对每个 seed 做一次简单的贪心搜索，返回评分最高的 samples_per_start 条路径。
        """

        if not seeds or samples_per_start <= 0:
            return []

        results: List[GraphPath] = []

        for seed in seeds:
            paths: List[GraphPath] = [GraphPath(seed)]  # 初始化为每个 seed 创建路径

            for depth in range(max_depth):
                candidates = []

                # 扩展每条路径
                for path in paths:
                    expanded_paths = self._expand(path, query, max_cand=self.cfg.max_cand)  # 扩展路径
                    candidates.extend(expanded_paths if expanded_paths else [path])

                # 如果没有有效的候选路径，提前终止
                if not candidates:
                    break

                # 筛选出有效的路径（至少有一个关系）
                candidates = [path for path in candidates if len(path.relations) > 0]

                # 如果筛选后没有有效的候选路径，提前终止
                if not candidates:
                    break

                # 对候选路径进行评分
                scores = self.path_ranker.rank(candidates, query)

                # 为每条路径添加评分
                for score, path in zip(scores, candidates):
                    path.scores.append(score)

                # 选出得分最高的 samples_per_start 条路径
                top_k = min(samples_per_start, len(candidates))
                top_k_indices = np.argsort(scores)[-top_k:][::-1]  # 从高到低排序
                paths = [candidates[i] for i in top_k_indices]

            # 将结果加入到最终返回的结果列表
            results.extend(paths)

        return results

    @staticmethod
    def collect_entities_from_paths(paths: List[GraphPath]) -> List[str]:
        """
        将多条路径上的实体去重汇总。
        """
        ent_set = set()
        for p in paths:
            for node in p.nodes:
                ent_set.add(node['id'])
        return list(ent_set)

    def preprocess_query(self, query: str, init_topk_entities: int):
        # 1. 抽主题实体，没有就用一个空占位，后面会回退用 query
        topic_ents = topic_entity_extraction(self.llm, query) or [{}]

        batch_match_ents = []
        for ent in topic_ents:
            mention = ent.get("mention", "")
            desc = ent.get("description", "")
            text = f"{mention}, {desc}".strip(", ") or query

            emb = np.asarray(self.embedding_model.embed_query(text), dtype=np.float32)
            batch_match_ents.append(
                self.ent_vec_store.search(emb, top_k=2 * init_topk_entities)
            )

        if not batch_match_ents:
            return []

        seeds, seen_ids = [], set()
        max_len = max(len(m) for m in batch_match_ents)
        i = 0

        # 2. 轮询各个 topic 的检索结果 + 去重
        while len(seeds) < init_topk_entities and i < max_len:
            for matches in batch_match_ents:
                if len(seeds) >= init_topk_entities or i >= len(matches):
                    continue

                candidate = matches[i]  # 假设为 (entity, score)
                entity = candidate[0] if isinstance(candidate, (list, tuple)) else candidate

                # 用 entity 的 id 去重；没有 id 就直接用 entity 本身
                ent_id = entity.get("id") if isinstance(entity, dict) else entity
                if ent_id in seen_ids:
                    continue

                seen_ids.add(ent_id)
                seeds.append(entity)
            i += 1

        return seeds


    def _search_related_entities(self, text: str, top_k=5):
        emb = np.asarray(self.embedding_model.embed_query(text), dtype=np.float32)
        ents = self.ent_vec_store.search(emb, top_k=top_k)
        return ents

    def search_paths_and_entities(
        self,
        query: str,
        start_ents: List[Any] = None,
        init_top_k_entities: int = 5,
        beam_width: int = 5,
        max_depth: int = 3,
    ):
        """
        一次性返回：
          - paths: 多条实体路径
          - ent_ids: 路径上所有实体的去重集合
        """

        # 1) 用嵌入找到起始实体

        if start_ents is not None and len(start_ents) >= init_top_k_entities:
            seeds = start_ents[:init_top_k_entities]
        else:
            seeds = self.preprocess_query(query, init_top_k_entities)
        # paths = self.beam_search(
        #     query=query,
        #     seeds=seeds,
        #     beam_width=beam_width,
        #     max_depth=max_depth,
        # )
        paths = self.greedy_search_per_seed(
            query=query,
            seeds=seeds,
            samples_per_start=1,
            max_depth=max_depth,
        )
        ent_ids = self.collect_entities_from_paths(paths)
        return paths, ent_ids
