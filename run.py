import json
from collections import defaultdict
from core.generator import Generator
from core.evaluate import Evaluator
from config import RunningConfig
from core.utils.tools import *
from core.heat_diffusion import *
import random

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


def calculate_MRR(rank_list: List[str], relevance_scores: Dict, k=None):
    mrr_total = 0.0
    for i, id in enumerate(rank_list):
        mrr_total += relevance_scores.get(id, 0.0) / (i + 1)

    return mrr_total / (len(relevance_scores.keys()) + 1e-5)


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

        doc_mrr_total += calculate_MRR(doc_rank, gold_docs)
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


from train_on_multi_dataset import load_dataset_runtime, get_interleaved_batches
from kg_rl.env import rollout
from kg_rl.path_scorer import *
from kg_rl.policy import Policy
from config import Config, RunningConfig


def rollout_path(rt, policy, cfg):
    q2paths = defaultdict(list)
    batch_iter = get_interleaved_batches([rt], batch_size=cfg.batch_size, max_width=cfg.beam_width)
    for rt, queries, start_nodes in tqdm(batch_iter, total=len(rt.questions) // cfg.batch_size + 1):
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
    return q2paths


def main(cfg: RunningConfig, rt=None):
    if rt == None:
        # --- 1. 初始化 (Setup) ---
        print(f"Loading dataset: {cfg.dataset_name}...")
        rt = load_dataset_runtime(cfg.dataset_name, cfg.device, mode="running", cfg_input=cfg)

    if getattr(cfg, 'sample', False):
        rt.questions = rt.questions[:500]

    # --- 2. 个性化计算与缓存 (Personalization & Caching) ---
    cache_path = Path(f'./output/{cfg.dataset_name}/q2personalization.json')
    cache_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    if getattr(cfg, 'ablation', True):
        q2personalization = {}
        for qitem in tqdm(rt.questions, desc="filtering start nodes", disable= not cfg.enable_tqdm):
            q = qitem['question']
            gold_ids = [ent[0]['id'] for ent in qitem['gold_ents']]

            distri = {id: 1.0 for id in gold_ids}
            noise_distri = {id: cfg.eval_step * 0.005 * random.random() for id in rt.idx.entity_embedding.ids}
            for k, v in distri.items():
                if noise_distri.get(k):
                    noise_distri[k] += v
                else:
                    noise_distri[k] = v
            q2personalization[q] = noise_distri

    elif getattr(cfg, 'use_cache', True) and cache_path.exists():
        print("Loading personalization from cache...")
        with open(cache_path, encoding='utf-8') as f:
            q2personalization = json.load(f)
    else:
        print("Cache not found. Computing rollout paths...")
        # 加载模型与策略
        base_scorer = DynamicPathScorer(cfg.pretrained_model_name)
        if cfg.checkpoint:
            print(f"Loading checkpoint: {cfg.checkpoint}")
            base_scorer.load_state_dict(torch.load(cfg.checkpoint, map_location='cpu'))

        base_scorer.to(cfg.device)
        policy = Policy(ScorerAdapter(base_scorer).to(cfg.device), temperature=0.9)
        q2personalization = {}
        q2paths = rollout_path(rt, policy, cfg)

        for qitem in tqdm(rt.questions, desc="Computing Importance", disable= not cfg.enable_tqdm):
            q = qitem['question']
            q2personalization[q] = compute_node_importance(q2paths[q])

        # 立即保存缓存，防止后续步骤崩溃导致计算丢失
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(q2personalization, f, ensure_ascii=False)
        print("Personalization computed and saved.")

    # --- 3. 文档检索 (Retrieval) ---
    retrieved_results = []
    questions_list = []
    q2items = {qitem['question']: qitem for qitem in rt.questions}

    for qitem in tqdm(rt.questions, desc="Retrieving Docs", disable= not cfg.enable_tqdm):
        q = qitem['question']

        # 计算热度场
        heat = compute_field(rt.idx.EP_ctx, q2personalization[q])

        sorted_candidates = sorted(heat.items(), key=lambda x: x[1], reverse=True)
        top_docs = []

        for doc_id, _ in sorted_candidates:
            if len(top_docs) >= cfg.top_k_docs:
                break
            if doc_id in rt.idx.doc_embedding.ids:
                top_docs.append(rt.idx.doc_embedding.id2item[doc_id])

        retrieved_results.append(top_docs)
        questions_list.append(q)


    recalls = compute_metrics([q2items[q] for q in questions_list], retrieved_results, q2personalization)
    print(f"Recall Metrics: {recalls}")


    if getattr(cfg, 'do_reasoning', True):
        generator = Generator(cfg)


        gen_args = [
            (q, docs, cfg.max_context_token)
            for q, docs in zip(questions_list, retrieved_results)
        ]

        answers = parallel_apply(
            fn=generator.reasoning,
            arg_tuples=gen_args,
            max_workers=cfg.max_workers,
            desc="Generating Answers"
        )


        predictions = [
            {
                "gold_answer": q2items[q]['answer'],
                "pred_answer": ans,
                "retrieved_docs": docs
            }
            for q, ans, docs in zip(questions_list, answers, retrieved_results)
        ]


        print("Evaluating predictions...")
        evaluator = Evaluator(llm_model=cfg.llm_model, predictions=predictions)
        metrics = evaluator.evaluate(max_workers=cfg.max_workers)
        print(f"Final Evaluation Metrics: {metrics}")
    return recalls

if __name__ == "__main__":
    datasets = ["wikiQA", "hotpotqa", "musique"]
    cfg = RunningConfig()
    for dataset in datasets:
        rt = load_dataset_runtime(dataset, cfg.device, mode="running", cfg_input=cfg)
        main(cfg, rt)


