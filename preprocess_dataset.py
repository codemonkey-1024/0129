import json
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import List, Any

from tqdm import tqdm

from config import RunningConfig
from core.ragindex import RagIndex
from core.utils.tools import read_jsonl, parallel_apply, write_jsonl
from core.generator import Generator
from core.evaluate import Evaluator
from core.graph_diffusion_retriever import *
from core.retriever import BeamRetriever
from config import RunningConfig
from core.utils.tools import *
from core.llm_functions import extract_evidence_entities
from config import logger
import numpy as np
from scipy.sparse import diags
import numpy as np
import networkx as nx
from scipy import sparse
from core.heat_diffusion import *
import time


def extract_gold_entities_from_doc(llm, question):
    evidence_docs = []
    for evidence in question['support_facts']:
        evidence_text = f"{evidence['title']}\n{evidence['doc']}"
        evidence_docs.append(evidence_text)
    entities = extract_evidence_entities(llm, question, '\n\n'.join(evidence_docs))
    return entities


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


def  preprocess_question(idx, question, llm, beam_ret, cfg):
    try:
        idx = beam_ret.idx
        embed_model = cfg.embedding_model
        # entities = extract_gold_entities_from_doc(llm, question)
        # question['support_ents'] = entities

        gold_ents = []
        gold_doc_titles = []
        for fact in question['support_facts']:
            text = fact['sentence'] if fact['sentence'].strip() != "" else fact['doc']
            related_triples = idx.triple_embedding.search(np.array(embed_model.embed_query(text), dtype=np.float32))
            fact['related_triples'] = related_triples
            last_evi = related_triples[0][0]['evidence']
            for triple, s in related_triples:
                if triple['evidence'] != last_evi:
                    break
                source = idx.entity_embedding.id2item[triple['source']]
                target = idx.entity_embedding.id2item[triple['target']]
                gold_ents.append((source, s))
                gold_ents.append((target, s))
                last_evi = triple['evidence']

            gold_doc_titles.append(fact['title'])



        question['gold_ents'] = list({ent['id']: (ent, s) for ent, s  in gold_ents}.values())
        question['start_ents'] = beam_ret.preprocess_query(question['question'], 20)


        # 计算每个实体相对query的信息量，以gold ents为热中心，扩散到其他实体
        graph = idx.EE_graph

        triple_scores = idx.triple_embedding.embeddings @ np.array(cfg.embedding_model.embed_query(question['question']), dtype=np.float32)
        for id, triple_score in zip(idx.triple_embedding.ids, triple_scores):
                triple = idx.triple_embedding.id2item[id]
                graph[triple['source']][triple['target']]['weight'] = triple_score
        sources = {ent[0]['id']: 1.0 for ent in question['gold_ents']}


        ee_heats = heat_diff(graph, sources, 'cuda')
        ee_heats = {k: v for k, v in normalize_dict_values(ee_heats).items() if v > 1e-3}

        question['ent_iv'] = ee_heats

        doc_ids = set([n[0] for n in idx.EP_graph.nodes(data=True) if n[1]['type'] == "P"])
        gold_titles_set = set(gold_doc_titles)  # 将gold_doc_titles转换为set，避免在循环中多次计算
        gold_docs = [
            n[1]
            for n in idx.EP_graph.nodes(data=True)
            if n[1]['type'] == 'P' and set(title.strip() for title in n[1]['title'].split('|')) & gold_titles_set
        ]
        question['gold_docs'] = gold_docs
        ep_heats = heat_diff(idx.EP_graph, {doc['id']:1.0  for doc in gold_docs}, 'cuda')
        ep_heats = {k: v for k, v in ep_heats.items() if v > 1e-4 and v not in doc_ids}
        ep_heats = normalize_dict_values(ep_heats)
        question['ent_iv'] = merge_dicts(ee_heats, ep_heats)
        # sorted(heats.values(), reverse=True)
        return question
    except Exception as e:
        traceback.print_exc()
        return None

def merge_dicts(dict1, dict2):
    result = dict1.copy()  # 先复制dict1，避免修改原字典
    for key, value in dict2.items():
        if key in result:
            result[key] += value  # 如果键在dict1中，值相加
        else:
            result[key] = value  # 如果键不在dict1中，补充该键值对
    return result

def normalize_dict_values(d):
    # 获取字典中的所有值并转为numpy数组
    values = np.array(list(d.values()))

    # 计算最大值和最小值
    max_value = np.max(values)
    min_value = np.min(values)

    # 进行归一化
    normalized_values = (values - min_value) / (max_value - min_value)

    # 创建归一化后的字典
    normalized_dict = {key: normalized_values[i] for i, key in enumerate(d.keys())}

    return normalized_dict


def compute_recalls(questions, retrieved_results, q2personalization):
    average_answer_recall = 0
    average_sent_recall = 0
    average_doc_recall = 0
    average_ent_recall = 0

    n = 0
    for question, docs in zip(questions, retrieved_results):
        text = "\n".join(
            [f"\n{i + 1}: {doc.get('title', '')} \n {doc.get('context', '')}" for i, doc in enumerate(docs)])

        average_answer_recall += 1 if question['answer'].lower() in text.lower() else 0

        retrieved_text = "\n".join([doc['context'] for doc in docs])
        retrieved_titles = "\n".join([doc['title'] for doc in docs])

        average_doc_recall += sum([1 for fact in question['support_facts'] if fact['title'] in retrieved_titles]) / len(
            question['support_facts'])
        average_sent_recall += sum(
            [1 for fact in question['support_facts'] if fact['sentence'] in retrieved_text]) / len(
            question['support_facts'])

        gold_end_ids = set([ent[0]['id'] for ent in question['gold_ents']])
        ret_end_ids = set([ent for ent in q2personalization[question['question']]])
        average_ent_recall += len(gold_end_ids & ret_end_ids) / len(gold_end_ids)

    return {
        "doc_recall": average_doc_recall / len(retrieved_results),
        "sent_recall": average_sent_recall / len(retrieved_results),
        "entity_recall": average_ent_recall / len(retrieved_results),
        "answer_recall": average_answer_recall / len(retrieved_results)
    }


def pre_process_questions(idx, cfg, beam_ret, questions):
    # 预处理，包括使用LLM抽取出gold实体， 链接到图谱中，构建起始节点（主要是为给训练用）
    q_path = Path(f'output/{cfg.dataset_name}/questions.jsonl')
    if not q_path.exists():
        questions = parallel_apply(fn=preprocess_question,
                                   arg_tuples=[(idx, q, cfg.llm_model, beam_ret, cfg) for q in questions],
                                   max_workers=cfg.max_workers,
                                   desc="Preprocessing")
        write_jsonl(Path(f'output/{cfg.dataset_name}/questions.jsonl'), [q for q in questions if q])
    else:
        questions = read_jsonl(q_path)
    return questions



def main(cfg):
    idx = RagIndex(cfg)
    idx.index(cfg.dataset_name)

    idx.EE_graph = idx.EE_graph.to_undirected()
    idx.EP_graph = idx.EP_graph.to_undirected()

    beam_ret = BeamRetriever(cfg, idx)

    questions = read_jsonl(Path(f'dataset/{cfg.dataset_name}/questions.jsonl'))
    pre_process_questions(idx, cfg, beam_ret, questions)


if __name__ == "__main__":
    cfg = RunningConfig()
    datasets = ["wikiQA"]
    for dataset in datasets:
        try:
            cfg.dataset_name = dataset
            main(cfg)
        except Exception as e:
            print(e)
