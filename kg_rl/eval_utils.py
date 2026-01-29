# kg_rl/eval_utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any, List
import math
import torch
from collections import defaultdict
from core.graph_diffusion_retriever import CoHITSRetriever
from core.retriever import BeamRetriever
from core.utils.tools import parallel_apply
import random
from core.utils.tools import calculate_NDCG

def get_interleaved_batches(runtimes, batch_size: int, max_width: int):
    """
    生成混合 Batch 迭代器。
    将所有数据集切块后打乱，防止模型对特定数据集过拟合。
    """
    tasks = []
    for rt in runtimes:
        random.shuffle(rt.questions)
        # 切分 chunks
        for i in range(0, len(rt.questions), batch_size):
            chunk = rt.questions[i: i + batch_size]
            tasks.append((rt, chunk))

    random.shuffle(tasks)  # 全局打乱任务顺序

    for rt, chunk in tasks:
        queries, start_nodes = [], []
        for q in chunk:
            # 获取起始实体，限制最大宽度
            nodes = q.get("start_ents", [])[:max_width]
            for node in nodes:
                queries.append(q["question"])
                start_nodes.append(node)

        if queries:
            yield rt, queries, start_nodes




def compute_node_importance(paths: List[object]):
    node2importance = defaultdict(lambda: 1.0)
    node2count = defaultdict(lambda: 0)
    for p in paths:
        node2importance[p.nodes[0]['id']] = 1.0
        node2count[p.nodes[0]['id']] += 1
        for s, node in zip(p.scores, p.nodes[1:]):
            node2importance[node['id']] = s
            node2count[node['id']] += 1

    return {k: s / node2count[k] for k, s in node2importance.items()}


def retrieve(cfg, qitem, beam_ret, docs_ret, top_k_docs):
    query = qitem['question']
    start_ents = qitem['start_ents']

    # 1. Beam Search for Paths
    paths, ent_ids = beam_ret.search_paths_and_entities(
        query,
        start_ents=start_ents,
        init_top_k_entities=cfg.init_top_k_entities,
        beam_width=cfg.beam_width,
        max_depth=cfg.max_depth
    )

    # 2. Calculate Importance for Diffusion
    node2importance = compute_node_importance(paths)

    # 3. Diffusion Retrieval
    docs = docs_ret.retrieve_for_bi_graph(
        personalization=node2importance,
        top_k_docs=top_k_docs,
    )
    return docs, node2importance


def eval_on_one_dataset(cfg, policy, ds_rt) -> float:
    policy.eval()
    idx = ds_rt.idx

    # Use Scorer from Policy
    beam_ret = BeamRetriever(cfg=ds_rt.cfg, index=idx, scorer=policy.scorer)
    docs_ret = CoHITSRetriever(ds_rt.cfg, idx)
    questions = ds_rt.questions

    q2personalization = {}
    retrieved_results = []

    with torch.no_grad():
        # Parallel execution
        args = [(cfg, q, beam_ret, docs_ret, ds_rt.cfg.top_k_docs) for q in questions]
        results = parallel_apply(retrieve, args, max_workers=ds_rt.cfg.max_workers)

        for q, res in zip(questions, results):
            # res[0] is docs, res[1] is node_importance
            retrieved_results.append(res[0])
            q2personalization[q['question']] = res[1]

    # Compute Metrics (Focusing on Ranking)
    metric = compute_metrics(
        questions,
        retrieved_results,
        q2personalization,
        top_k=ds_rt.cfg.top_k_docs
    )

    print(f"[Eval Metric] Dataset: {ds_rt.name} | NDCG: {metric['doc_ndcg']:.4f} | Recall: {metric['doc_recall']:.4f}")

    # Return NDCG as the primary metric for optimization
    return metric['doc_ndcg']



from tqdm import tqdm
from kg_rl.env import rollout
from core.heat_diffusion import compute_field

def compute_metrics(qitems, retrieved_results, q2personalization):
    average_answer_recall = 0
    average_sent_recall = 0
    average_doc_recall = 0
    average_ent_recall = 0
    doc_mrr_total = 0.0
    doc_ndcg_total = 0.0
    n = 0
    for qitem, docs in zip(qitems, retrieved_results):
        text = "\n".join(
            [f"\n{i + 1}: {doc.get('title', '')} \n {doc.get('context', '')}" for i, doc in enumerate(docs)])
        gold_docs = {fact['id']: 1.0 for fact in qitem['gold_docs']}
        doc_rank = [doc["id"] for doc in docs]

        doc_ndcg_total += calculate_NDCG(doc_rank, gold_docs)

        average_answer_recall += 1 if qitem['answer'].lower() in text.lower() else 0

        retrieved_text = "\n".join([doc['context'] for doc in docs])
        retrieved_titles = "\n".join([doc['title'] for doc in docs])

        average_doc_recall += sum([1 for fact in qitem['support_facts'] if fact['title'] in retrieved_titles]) / len(
            qitem['support_facts'])
        average_sent_recall += sum(
            [1 for fact in qitem['support_facts'] if fact['sentence'] in retrieved_text]) / len(
            qitem['support_facts'])

        gold_end_ids = set([ent[0]['id'] for ent in qitem['gold_ents']])
        ret_end_ids = set([ent for ent in q2personalization[qitem['question']]])
        average_ent_recall += len(gold_end_ids & ret_end_ids) / len(gold_end_ids)

    return {
        "doc_recall": round(average_doc_recall / len(retrieved_results), 4),
        "sent_recall": round(average_sent_recall / len(retrieved_results), 4),
        "entity_recall": round(average_ent_recall / len(retrieved_results), 4),
        "answer_recall": round(average_answer_recall / len(retrieved_results), 4),
        "ndcg": round(doc_ndcg_total / len(retrieved_results), 4),
        "mrr": round(doc_mrr_total / len(retrieved_results), 4)
    }


def eval_one_epoch(cfg, policy, rt) -> tuple:
    policy.eval()
    idx = rt.idx

    q2paths = defaultdict(list)
    batch_iter = get_interleaved_batches([rt], batch_size=32, max_width=cfg.beam_width)
    for rt, queries, start_nodes in tqdm(batch_iter, total=len(rt.questions) // 32 + 1):
        trajs = rollout(
            env=rt.env,
            policy=policy,
            start_nodes=start_nodes,
            queries=queries,
            warm_starts=None,
            greedy=True,
            use_grad=False,
            samples_per_start=1
        )
        for traj in trajs:
            q2paths[traj.query].append(traj.final_path)

    q2personalization = {}
    retrieved_results = []
    questions = []
    for q, paths in tqdm(q2paths.items(), desc="Diff to doc"):
        node2importance = compute_node_importance(paths)
        q2personalization[q] = node2importance
        heat = compute_field(rt.idx.EP_ctx, node2importance)
        sorted_ids = [id for id, v in sorted(heat.items(), key=lambda x: x[1], reverse=True)]
        top_k_doc_ids = []

        for id in sorted_ids:
            if len(top_k_doc_ids) < cfg.top_k_docs and id in rt.idx.doc_embedding.ids:
                top_k_doc_ids.append(id)

        docs = [rt.idx.doc_embedding.id2item[id] for id in top_k_doc_ids]
        retrieved_results.append(docs)
        questions.append(q)

    q2items = {qitem['question']: qitem for qitem in rt.questions}
    recalls = compute_metrics([q2items[q] for q in questions], retrieved_results, q2personalization)
    return recalls["doc_recall"], recalls["ndcg"]