from __future__ import annotations

import json
import pickle
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from core.utils.tools import (
    read_jsonl,
    write_jsonl,
    parallel_apply,
    cosine_sim,
    hash_text,
)
from core.llm_functions import (
    coref_disambiguation,
    extract_entities_relations_from_doc,
    reasoning as llm_reasoning,
)
from core.utils.text_spliter import SemanticTextSplitterV2
from config import RunningConfig


class GraphBuilder:
    """
    负责：
      - 从数据集读取文档
      - 文本切分
      - 指代消解（可选）
      - 实体 / 关系抽取
      - 结果写回到 ./output/{dataset}/Corpus.json
    """

    def __init__(self, cfg: RunningConfig):
        self.cfg = cfg
        self.llm = self.cfg.llm_model
        self.short_spliter = SemanticTextSplitterV2(
            max_tokens=128,
            min_tokens=8,
            target_tokens=48,
            overlap_tokens=0,
            model_name="gpt-4",
        )
        self.long_spliter = SemanticTextSplitterV2()

    def index(self, dataset: str) -> List[Dict[str, Any]]:
        """
        对原始语料做实体/关系抽取，如果已有缓存则直接读取。

        返回：带有 entities / relations 字段的文档列表。
        """
        dataset_path = Path(f"./dataset/{dataset}/Corpus.json")
        output_path = Path(f"./output/{dataset}/Corpus.json")

        if output_path.exists():
            print(f"[GraphBuilder] Cache exist, load from {output_path}...")
            corpus = read_jsonl(output_path)
        else:
            print("[GraphBuilder] Indexing...")
            corpus = read_jsonl(dataset_path)

            # 如需指代消解，解除下面注释
            # corpus = parallel_apply(
            #     fn=self.coref_disambiguation,
            #     arg_tuples=[(doc,) for doc in corpus],
            #     max_workers=self.cfg.max_workers,
            #     desc="Coref Disambiguation",
            # )

            # 实体 / 关系抽取（长切分）
            corpus = parallel_apply(
                fn=self.extract_entities_and_relations,
                arg_tuples=[(doc,) for doc in corpus],
                max_workers=self.cfg.max_workers,
                desc="Extracting Entities & Relations",
            )
            write_jsonl(output_path, corpus)

        return corpus

    def extract_entities_and_relations(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        长切分 → 抽取 → 统一实体ID → 回填关系端点
        """
        chunks = self.long_spliter.split_text(doc.get("context", ""))
        doc["entities"] = []
        doc["relations"] = []

        for chunk in chunks:
            try:
                entities, relations = extract_entities_relations_from_doc(
                    self.llm, chunk
                )

                # 原始 id → mention 哈希
                eid2id = {
                    ent["id"]: hash_text(ent.get("mention", "")) for ent in entities
                }

                # 替换实体 id
                for ent in entities:
                    ent["id"] = hash_text(ent.get("mention", ""))

                # 替换关系端点
                for rel in relations:
                    if rel["source"] in eid2id:
                        rel["source"] = eid2id[rel["source"]]
                    if rel["target"] in eid2id:
                        rel["target"] = eid2id[rel["target"]]

                doc["entities"].extend(entities)
                doc["relations"].extend(relations)
            except Exception:
                traceback.print_exc()

        return doc

    def coref_disambiguation(
        self, doc: Dict[str, Any], pre_chunk_num: int = 1
    ) -> Dict[str, Any]:
        """
        短切分 + 结合前文窗口做指代消解，失败则回退原文。
        """
        chunks = self.short_spliter.split_text(doc.get("context", ""))
        pre_chunk = deque(maxlen=pre_chunk_num)
        out: List[str] = []

        for chunk in chunks:
            try:
                disamb = coref_disambiguation(
                    self.llm, doc.get("title", ""), "\n".join(pre_chunk), chunk
                )
            except Exception:
                disamb = chunk

            out.append(disamb)
            pre_chunk.append(disamb)

        doc["context"] = "\n".join(out)
        return doc

