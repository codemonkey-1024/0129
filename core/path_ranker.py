import json
import threading
import torch
import torch.nn.functional as F
from typing import List
from config import RunningConfig, logger
from kg_rl.path_scorer import DynamicPathScorer, ScorerAdapter
from core.schema import GraphPath
from core.ragindex import RagIndex
import numpy as np


class PathRanker:

    def __init__(self, cfg: RunningConfig, idx: RagIndex, scorer: DynamicPathScorer = None, device: str = None, rank_mode: str= "embedding"):
        self.cfg = cfg
        self.idx = idx
        self.scorer = scorer
        self._init_lock = threading.Lock()  # 初始化锁
        self.embed_model = cfg.embedding_model
        self.rank_mode = rank_mode
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.path_scores = []


    def init_scorer(self):
        # 第一层检查：已经有 scorer 了就直接返回（无锁，快速路径）
        if self.scorer is not None:
            return

        # 只有真的需要初始化时才加锁
        with self._init_lock:
            # 第二层检查：可能已经有别的线程初始化完了
            if self.scorer is not None:
                return

            base_scorer = DynamicPathScorer(
                model_name=self.cfg.pretrained_model_name,
                use_lora=getattr(self.cfg, "use_lora", True)
            )
            base_scorer.load_state_dict(
                torch.load(self.cfg.checkpoint, map_location="cpu")
            )
            base_scorer.to(self.device)
            self.scorer = ScorerAdapter(base_scorer).to(self.device)
            self.scorer.eval()

    def _rank_by_scorer(self, paths: List[GraphPath], query: str):
        # 延迟初始化模型（线程安全）
        if self.scorer is None:
            self.init_scorer()
        self.scorer.eval()
        with torch.no_grad():
            scores = self.scorer.score_paths(paths, [query] * len(paths))
            result = F.softmax(scores, dim=0).squeeze()
            result_list = [result.item()] if result.dim() == 0 else result.tolist()
        return result_list

    import numpy as np

    def _rank_by_embedding(self, candidates: List[GraphPath], query: str) -> List[float]:
        if not candidates: return []

        # 1. 准备 Query [D, 1]
        q_vec = self.idx.query_embedding.gather([query])
        q_vec = q_vec.T

        # 2. 分类收集 ID
        tri_idxs, tri_ids = [], []
        doc_idxs, doc_ids, ent_ids = [], [], []

        for i, p in enumerate(candidates):
            rel = p.relations[-1]
            # 判断依据：知识图谱边有 'id'，文档边没有
            if rel.get('id'):
                tri_idxs.append(i)
                tri_ids.append(rel['id'])
            else:
                doc_idxs.append(i)
                doc_ids.append(rel['doc']['id'])
                ent_ids.append(rel['end']['id'])

        # 3. 定义通用计算函数 (核心简化点)
        def calc_sim(store, ids):
            if not ids: return 0.0
            vecs = store.gather(ids)
            return (vecs @ q_vec).flatten()

        # 4. 计算并填回
        final_scores = np.zeros(len(candidates), dtype=np.float32)

        if tri_ids:
            final_scores[tri_idxs] = calc_sim(self.idx.triple_embedding, tri_ids)

        if doc_ids:
            # Doc Path Score = Sim(Doc) + Sim(EndEntity)
            score_doc = calc_sim(self.idx.doc_embedding, doc_ids)
            score_ent = calc_sim(self.idx.entity_embedding, ent_ids)
            final_scores[doc_idxs] = (score_doc + score_ent)/2

        return final_scores.tolist()



    def rank(self, graph_paths: List[GraphPath], query: str):
        if len(graph_paths) > 512:
            logger.warning(f"number of paths exceeds 256: {len(graph_paths)}")

        if self.rank_mode == "scorer":
            return self._rank_by_scorer(graph_paths, query)
        elif self.rank_mode == "embedding":
            scores = self._rank_by_embedding(graph_paths, query)
            self.path_scores.append((query, graph_paths, scores))
            return scores
        else:
            raise ValueError(f"{self.rank_mode} Error!")
