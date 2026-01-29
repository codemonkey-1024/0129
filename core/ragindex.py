# index.py
import os
import copy
import pickle
import traceback
import networkx as nx
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
from tqdm import tqdm

from core.llm_functions import extract_entities_relations_from_doc, coref_disambiguation
from core.utils.text_spliter import SemanticTextSplitterV2
from core.utils.tools import hash_text, read_jsonl, parallel_apply, write_jsonl, recursive_cluster_indices
from core.heat_diffusion import build_context
from config import RunningConfig


# =========================
# Helper Functions
# =========================

def calculate_average_degree_by_type(G, node_type):
    """
    Optimized calculation of average degree for a specific node type.
    """
    nodes = [n for n, attr in G.nodes(data="type") if attr == node_type]
    if not nodes:
        return 0.0
    # G.degree(nodes) returns an iterator of (node, degree)
    total_degree = sum(d for _, d in G.degree(nodes))
    return total_degree / len(nodes)


# =========================
# GraphBuilder
# =========================
class GraphBuilder:
    """
    Responsible for: Reading Dataset -> Splitting -> Entity Extraction -> Saving Raw Corpus
    """

    def __init__(self, cfg: RunningConfig):
        self.cfg = cfg
        self.llm = self.cfg.llm_model
        # Initialize splitters
        self.short_spliter = SemanticTextSplitterV2(
            max_tokens=256, min_tokens=48, target_tokens=128, overlap_tokens=0, model_name="gpt-4"
        )
        self.long_spliter = SemanticTextSplitterV2(
            max_tokens=256, min_tokens=48, target_tokens=128, overlap_tokens=0, model_name="gpt-4"
        )

    def index(self, dataset: str) -> List[Dict[str, Any]]:
        dataset_path = Path(f"./dataset/{dataset}/corpus.jsonl")
        output_path = Path(f"./output/{dataset}/corpus.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            print(f"[GraphBuilder] Cache exist, loading from {output_path}...")
            corpus = read_jsonl(output_path)
        else:
            print("[GraphBuilder] Indexing raw corpus...")
            corpus = read_jsonl(dataset_path)

            # Optional: Coreference Disambiguation
            corpus = parallel_apply(
                fn=self.coref_disambiguation,
                arg_tuples=[(doc,) for doc in corpus],
                max_workers=self.cfg.max_workers,
                desc="Coref Disambiguation",
            )

            # Extract Entities & Relations
            extract_res = parallel_apply(
                fn=self.extract_entities_and_relations,
                arg_tuples=[(doc,) for doc in corpus],
                max_workers=self.cfg.max_workers,
                desc="Extracting Entities & Relations",
            )
            corpus = [doc for docs in extract_res for doc in docs]
            write_jsonl(output_path, corpus)
            print(f"[GraphBuilder] Saved raw corpus to {output_path}")

        return corpus

    def _extract_and_normalize(self, llm, chunk: str) -> Tuple[List[Dict], List[Dict]]:
        entities, relations = extract_entities_relations_from_doc(llm, chunk)

        # Normalize IDs using hash
        eid2hid = {ent["id"]: hash_text(ent.get("mention", "")) for ent in entities}

        valid_entities = []
        for ent in entities:
            # Ensure ID is updated
            ent["id"] = eid2hid[ent["id"]]
            valid_entities.append(ent)

        valid_relations = []
        for rel in relations:
            # Only keep relations where both endpoints exist
            if rel.get("source") in eid2hid and rel.get("target") in eid2hid:
                rel["source"] = eid2hid[rel["source"]]
                rel["target"] = eid2hid[rel["target"]]
                valid_relations.append(rel)

        return valid_entities, valid_relations

    def extract_entities_and_relations(self, doc: Dict[str, Any], return_multi_doc: bool = True) -> List[Any]:
        context = doc.get("context", "") or ""
        chunks = self.long_spliter.split_text(context)

        if not chunks:
            doc["entities"] = []
            doc["relations"] = []
            return [doc] if not return_multi_doc else []

        if return_multi_doc:
            docs = []
            for chunk in chunks:
                new_doc = deepcopy(doc)
                new_doc["context"] = chunk
                try:
                    ents, rels = self._extract_and_normalize(self.llm, chunk)
                    new_doc["entities"] = ents
                    new_doc["relations"] = rels
                except Exception:
                    traceback.print_exc()
                    new_doc["entities"] = []
                    new_doc["relations"] = []
                docs.append(new_doc)
            return docs
        else:
            # If not splitting docs, aggregate everything (logic omitted for brevity as usually True)
            pass

    def coref_disambiguation(self, doc: Dict[str, Any], pre_chunk_num: int = 1) -> Dict[str, Any]:
        chunks = self.short_spliter.split_text(doc.get("context", ""))
        pre_chunk = deque(maxlen=pre_chunk_num)
        out = []
        for chunk in chunks:
            try:
                disamb = coref_disambiguation(self.llm, doc.get("title", ""), "\n".join(pre_chunk), chunk)
            except Exception:
                disamb = chunk
            out.append(disamb)
            pre_chunk.append(disamb)
        doc["context"] = "\n".join(out)
        return doc


# =========================
# Embedding Container
# =========================
@dataclass
class EmbeddingItems:
    ids: List[str]
    embeddings: np.ndarray
    id2item: Dict[str, Any]
    id2index: Dict[str, int]

    def __init__(self, ids, id2item, embeddings):
        self.ids = ids
        self.id2item = id2item

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.embeddings = embeddings / norms

        self.id2index = {id_: idx for idx, id_ in enumerate(self.ids)}

    def search(self, query_vec: np.ndarray, top_k: int = 10):
        q = query_vec.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0: q_norm = 1.0
        q = q / q_norm

        sims = self.embeddings @ q
        idx = np.argsort(-sims)[:top_k]
        return [(self.id2item[self.ids[i]], float(sims[i])) for i in idx]

    def gather(self, requested_ids: List[str]) -> np.ndarray:
        indices = [self.id2index[id_] for id_ in requested_ids if id_ in self.id2index]
        if not indices:
            return np.array([])
        return self.embeddings[indices]


# =========================
# RagIndex: Main Entry
# =========================
class RagIndex:
    def __init__(self, cfg: RunningConfig):
        self.cfg = cfg
        self.EE_graph = nx.DiGraph()
        self.EP_graph = nx.Graph()

        # Contexts
        self.EE_ctx = None
        self.EP_ctx = None

        # Embeddings
        self.triple_embedding: Optional[EmbeddingItems] = None
        self.entity_embedding: Optional[EmbeddingItems] = None
        self.doc_embedding: Optional[EmbeddingItems] = None
        self.query_embedding: Optional[EmbeddingItems] = None

        self.cache_dir = Path(f"./output/{self.cfg.dataset_name}/cache/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def index(self, dataset_name: str):
        print(f"Indexing dataset {dataset_name}")
        self.cfg.dataset_name = dataset_name
        self.cache_dir = Path(f"./output/{dataset_name}/cache/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load or Build Raw Corpus (Chunked)
        graph_builder = GraphBuilder(self.cfg)
        corpus = graph_builder.index(dataset_name)

        # 2. Merge Documents (Optional but recommended for semantic density)
        if getattr(self.cfg, "merge_doc", False):
            corpus = self._get_merged_corpus(corpus)

        # 3. Build Graphs (EE & EP)
        self._build_graphs(corpus)

        # 4. Build Embeddings (Doc, Entity, Triple)
        self._build_embeddings()

        # 5. Build Query Embeddings
        self._build_query_embeddings()

        # 6. Graph Contexts (Heat Diffusion)
        print(f"[Index] Building Graph Contexts via Heat Diffusion...")
        self.EE_ctx = build_context(self.EE_graph.to_undirected())
        self.EP_ctx = build_context(self.EP_graph.to_undirected())

        # Statistics
        print("=" * 30)
        EP_E_avage_degree = calculate_average_degree_by_type(self.EP_graph, "E")
        EP_P_avage_degree = calculate_average_degree_by_type(self.EP_graph, "P")
        print(f"[Index] Final Stats:")
        print(f"[Index] doc num in graph: {len(corpus)}")
        print(f"[Index] doc num in vector_store: {len(self.doc_embedding.ids)}")
        print(f"[Index] entity num in vector_store: {len(self.entity_embedding.ids)}")
        print(f"[Index] triple num in vector_store: {len(self.triple_embedding.ids)}")
        print(f"[Index] EE graph: {len(self.EE_graph.nodes())} nodes; {len(self.EE_graph.edges())} edges")
        print(f"[Index] EP graph: {len(self.EP_graph.nodes())} nodes; {len(self.EP_graph.edges())} edges, E degree: {round(EP_E_avage_degree, 4)}; P degree: {round(EP_P_avage_degree, 4)}")

    print("=" * 30)

    # ---------- Document Merging Logic ----------
    def _get_merged_corpus(self, raw_corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merge_path = Path(f"./output/{self.cfg.dataset_name}/merged_corpus.jsonl")

        if merge_path.exists():
            print(f"[Index] Loading Merged Corpus from {merge_path}...")
            return read_jsonl(merge_path)

        print("[Index] Merging documents (Clustering)...")

        # 1. 去重 (Deduplication)
        unique_corpus = []
        seen_hashes = set()
        for doc in raw_corpus:
            doc_hash = hash_text(doc.get("context", ""))
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_corpus.append(doc)

        # 2. 聚类 (Clustering)
        doc_texts = [doc.get("context", "") for doc in unique_corpus]
        doc_vecs = np.array(self.cfg.embedding_model.embed_documents(doc_texts), dtype=np.float32)
        clusters = recursive_cluster_indices(doc_vecs)

        merged_corpus = []

        for cluster_indices in tqdm(clusters, desc="Merging Clusters"):
            if not cluster_indices: continue

            base_idx = cluster_indices[0]
            merged_doc = deepcopy(unique_corpus[base_idx])

            # 辅助去重集合
            existing_ent_ids = {e['id'] for e in merged_doc.get('entities', [])}
            existing_rel_ids = {hash_text(str((r['source'], r['relation'], r['target']))) for r in
                                merged_doc.get('relations', [])}

            # --- [关键优化]：初始化通用字段去重集合 ---
            # 用来存储除了核心字段外的其他字段（如 url, author 等）
            # 结构: {"url": {"http://google.com"}, "author": {"Alice"}}
            meta_sets = {}
            # 先把 base_doc 的元数据放进去
            for k, v in merged_doc.items():
                if k not in ["context", "entities", "relations", "title", "id", "type"]:
                    if isinstance(v, (str, int, float)):
                        meta_sets[k] = {str(v)}
                    elif isinstance(v, list):
                        # 如果元数据本身是列表（例如 tags: ["A", "B"]），则转为字符串集合处理
                        try:
                            meta_sets[k] = set(str(item) for item in v)
                        except:
                            meta_sets[k] = set()

            # --- [Title 特殊处理] ---
            title_set = set()
            if merged_doc.get("title"): title_set.add(merged_doc["title"].strip())

            # 3. 遍历合并
            for i in cluster_indices[1:]:
                doc_to_merge = unique_corpus[i]

                # (A) 核心字段：Context
                if "context" in doc_to_merge:
                    merged_doc["context"] += "\n\n" + doc_to_merge["context"]

                # (B) 核心字段：Entities
                for ent in doc_to_merge.get("entities", []):
                    if ent['id'] not in existing_ent_ids:
                        merged_doc["entities"].append(ent)
                        existing_ent_ids.add(ent['id'])

                # (C) 核心字段：Relations
                for rel in doc_to_merge.get("relations", []):
                    rel_hash = hash_text(str((rel['source'], rel['relation'], rel['target'])))
                    if rel_hash not in existing_rel_ids:
                        merged_doc["relations"].append(rel)
                        existing_rel_ids.add(rel_hash)

                # (D) 核心字段：Title
                if doc_to_merge.get("title"):
                    title_set.add(doc_to_merge["title"].strip())

                # (E) [新增] 通用元数据字段自动合并
                for k, v in doc_to_merge.items():
                    # 跳过已处理的核心字段和 ID
                    if k in ["context", "entities", "relations", "title", "id", "type"]:
                        continue

                    if k not in meta_sets:
                        meta_sets[k] = set()

                    # 尝试将值加入集合（自动去重）
                    if isinstance(v, (str, int, float)):
                        meta_sets[k].add(str(v))
                    elif isinstance(v, list):
                        for item in v:
                            meta_sets[k].add(str(item))

            # 4. 收尾：将集合转换回字符串或列表

            # Title 收尾
            merged_doc["title"] = " | ".join(sorted(list(title_set)))

            # 通用元数据收尾
            for k, v_set in meta_sets.items():
                if not v_set: continue
                # 如果只有一个值，直接赋值；如果有多个值，用 "; " 拼接
                # 你也可以选择变成 list: merged_doc[k] = list(v_set)
                sorted_vals = sorted(list(v_set))
                if len(sorted_vals) == 1:
                    merged_doc[k] = sorted_vals[0]  # 保持原样
                else:
                    merged_doc[k] = "; ".join(sorted_vals)  # 拼接，例如 url1; url2

            # ID 重算
            merged_doc['id'] = hash_text(merged_doc['context'])

            merged_corpus.append(merged_doc)

        print(f"[Index] Merging complete. Reduced {len(unique_corpus)} unique chunks to {len(merged_corpus)} docs.")
        write_jsonl(merge_path, merged_corpus)
        return merged_corpus

    # ---------- Graph Building ----------
    def _build_graphs(self, corpus: List[Dict[str, Any]]) -> None:
        ee_path = self.cache_dir / "EE_graph.pkl"
        ep_path = self.cache_dir / "EP_graph.pkl"

        # Check if graphs exist on disk
        if ee_path.exists() and ep_path.exists():
            print(f"[Index] Loading cached graphs from {self.cache_dir}...")
            with open(ee_path, "rb") as f: self.EE_graph = pickle.load(f)
            with open(ep_path, "rb") as f: self.EP_graph = pickle.load(f)
            return

        print("[Index] Building Graphs from scratch...")
        self.EE_graph.clear()
        self.EP_graph.clear()

        # Temporary lists for batch processing
        ent_nodes = {}  # ID -> Data
        doc_nodes = []
        ep_edges = []
        ee_edges = []

        for doc in tqdm(corpus, desc="Processing Corpus for Graphs"):
            # Doc Node
            doc_id = doc.get("id") or hash_text(doc.get("context", ""))
            doc["id"] = doc_id
            doc["type"] = "P"
            doc_nodes.append(doc)

            ctx_lower = (doc.get("context") or "").lower()

            # Process Entities & EP Edges
            for ent in doc.get("entities", []):
                ent_id = ent["id"]
                if ent_id not in ent_nodes:
                    ent["type"] = "E"
                    ent_nodes[ent_id] = ent

                # Calculate weight (frequency of mention)
                mention = (ent.get("mention") or "").lower()
                weight = 1.0
                if mention:
                    weight = float(ctx_lower.count(mention) + 1e-3)  # Avoid zero

                ep_edges.append((doc_id, ent_id, {"weight": weight, "relation": "contain"}))

            # Process EE Edges
            for rel in doc.get("relations", []):
                # Ensure edge has an ID
                rel_id = hash_text(str((rel['source'], rel['relation'], rel['target'])))
                rel['id'] = rel_id
                ee_edges.append(rel)

        # Add Nodes/Edges to NetworkX
        # 1. Entity Nodes
        for eid, data in ent_nodes.items():
            self.EP_graph.add_node(eid, **data)
            self.EE_graph.add_node(eid, **data)

        # 2. Doc Nodes
        for d in doc_nodes:
            self.EP_graph.add_node(d["id"], **d)

        # 3. EP Edges
        self.EP_graph.add_edges_from(ep_edges)

        # 4. EE Edges
        for r in ee_edges:
            self.EE_graph.add_edge(r["source"], r["target"], **r)

        # Save to Disk
        print(f"[Index] Saving graphs to {self.cache_dir}...")
        with open(ee_path, "wb") as f:
            pickle.dump(self.EE_graph, f)
        with open(ep_path, "wb") as f:
            pickle.dump(self.EP_graph, f)

    # ---------- Embeddings ----------
    def _build_embeddings(self) -> None:
        """
        Builds or loads embeddings for Docs, Entities, and Triples based on the current graphs.
        """
        doc_path = self.cache_dir / "docs_embeddings.npz"
        ent_path = self.cache_dir / "entity_embeddings.npz"
        trip_path = self.cache_dir / "triple_embeddings.npz"

        # Check Cache
        if doc_path.exists() and ent_path.exists() and trip_path.exists():
            print(f"[Index] Loading cached embeddings from {self.cache_dir}...")

            d_data = np.load(doc_path, allow_pickle=True)
            self.doc_embedding = EmbeddingItems(d_data["ids"],
                                                {nid: self.EP_graph.nodes[nid] for nid in d_data["ids"] if
                                                 nid in self.EP_graph},
                                                d_data["vectors"])

            e_data = np.load(ent_path, allow_pickle=True)
            self.entity_embedding = EmbeddingItems(e_data["ids"],
                                                   {nid: self.EP_graph.nodes[nid] for nid in e_data["ids"] if
                                                    nid in self.EP_graph},
                                                   e_data["vectors"])

            # For triples, we need to reconstruct the map
            t_data = np.load(trip_path, allow_pickle=True)
            # Re-extract triples from graph to build ID map
            triples_map = {edge[-1]['id']: edge[-1] for edge in self.EE_graph.edges(data=True)}
            self.triple_embedding = EmbeddingItems(t_data["ids"], triples_map, t_data["vectors"])
            return

        print("[Index] Computing Embeddings (LLM Call)...")
        embedding_model = self.cfg.embedding_model

        # 1. Documents
        doc_nodes = [n for n, d in self.EP_graph.nodes(data=True) if d.get('type') == 'P']
        doc_texts = [self.EP_graph.nodes[n].get("context", "") for n in doc_nodes]
        doc_vecs = np.array(embedding_model.embed_documents(doc_texts), dtype=np.float32)

        # 2. Entities
        ent_nodes = [n for n, d in self.EP_graph.nodes(data=True) if d.get('type') == 'E']
        ent_texts = [f"{self.EP_graph.nodes[n].get('mention', '')}: {self.EP_graph.nodes[n].get('description', '')}" for
                     n in ent_nodes]
        ent_vecs = np.array(embedding_model.embed_documents(ent_texts), dtype=np.float32)

        # 3. Triples
        edges = list(self.EE_graph.edges(data=True))
        triple_ids = [e[-1]['id'] for e in edges]
        triple_texts = [f"{e[-1]['source_name']} {e[-1]['relation']} {e[-1]['target_name']}" for e in edges]
        triple_vecs = np.array(embedding_model.embed_documents(triple_texts), dtype=np.float32)

        # Save
        np.savez_compressed(doc_path, ids=np.array(doc_nodes), vectors=doc_vecs)
        np.savez_compressed(ent_path, ids=np.array(ent_nodes), vectors=ent_vecs)
        np.savez_compressed(trip_path, ids=np.array(triple_ids), vectors=triple_vecs)

        # Build Objects
        self.doc_embedding = EmbeddingItems(doc_nodes, {n: self.EP_graph.nodes[n] for n in doc_nodes}, doc_vecs)
        self.entity_embedding = EmbeddingItems(ent_nodes, {n: self.EP_graph.nodes[n] for n in ent_nodes}, ent_vecs)

        triples_map = {e[-1]['id']: e[-1] for e in edges}
        self.triple_embedding = EmbeddingItems(triple_ids, triples_map, triple_vecs)

    def _build_query_embeddings(self):
        q_path = self.cache_dir / "query_embeddings.npz"
        q_file = Path(f"dataset/{self.cfg.dataset_name}/questions.jsonl")


        if q_path.exists():
            print(f"[Index] Loading query embeddings from {q_path}...")
            data = np.load(q_path, allow_pickle=True)
            self.query_embedding = EmbeddingItems(data["ids"], {i: i for i in data["ids"]}, data["vectors"])
            return

        print("[Index] Building Query Embeddings...")
        questions = [q['question'] for q in read_jsonl(q_file)]
        vecs = np.array(self.cfg.embedding_model.embed_documents(questions), dtype=np.float32)

        np.savez_compressed(q_path, ids=np.array(questions), vectors=vecs)
        self.query_embedding = EmbeddingItems(questions, {q: q for q in questions}, vecs)