import json
import os

# 先加载已有的 title2doc


def process_samples(samples, batch_index):
    corpus = []
    questions = []
    d_index = 0
    q_index = 0
    title2doc = {}
    for d in samples:
        id2doc = {}
        for para in d['paragraphs']:
            context = para['paragraph_text']
            title = para['title']
            title2doc[title] = context
            id2doc[para['idx']] = para
            corpus.append({
                "title": title,
                "context": context,
                "id": para['idx'],
                "meta": {"title": title}
            })

        # 构造支持句（带安全检查）
        support_facts = []
        for t in d['question_decomposition']:
            support_facts.append({
                "title": id2doc[t['paragraph_support_idx']]['title'],
                "sentence": id2doc[t['paragraph_support_idx']]['paragraph_text'],
                "doc": id2doc[t['paragraph_support_idx']]['paragraph_text']
            })

        questions.append({
            "question": d["question"],
            "answer": d["answer"],
            "id": q_index,
            "support_facts": support_facts
        })
        q_index += 1


    corpus = list({p['title']+p['context']:p  for p in corpus}.values())

    with open( "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open( "questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


# 加载样本数据
with open("sample.json", encoding="utf-8") as f:
    data = json.load(f)

process_samples(data, 0)


