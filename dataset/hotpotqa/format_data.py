import json
import os

# 先加载已有的 title2doc
with open('title2doc.json', encoding='utf-8') as fp:
    title2doc = json.load(fp)


def process_samples(samples, batch_index):
    corpus = []
    questions = []
    d_index = 0
    q_index = 0

    for d in samples:
        for title, sents in zip(d['context']['title'], d['context']['sentences']):
            context = "\n".join(sents)
            corpus.append({
                "title": title,
                "context": context,
                "id": d_index,
                "meta": {"title": title}
            })
            d_index += 1

        # 构造支持句（带安全检查）
        support_facts = []
        for title, sent_idx in zip(d["supporting_facts"]['title'], d["supporting_facts"]['sent_id']):
            sents = title2doc.get(title)

            if sents is not None and isinstance(sents, list) and 0 <= sent_idx < len(sents):
                sentence = sents[sent_idx]
            else:
                # 如果想检查是哪些数据有问题，可以打印或记录日志
                print(f"[WARN] support_fact 索引越界或缺失: title={title}, idx={sent_idx}")
                sentence = ""

            support_facts.append({
                "title": title,
                "sentence": sentence,
                "doc": "\n".join(title2doc.get(title))
            })

        questions.append({
            "question": d["question"],
            "answer": d["answer"],
            "id": q_index,
            "support_facts": support_facts
        })
        q_index += 1

    corpus = list({p['title'] + p['context']: p for p in corpus}.values())
    with open( "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open("questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


# 加载样本数据
with open("sample.json", encoding="utf-8") as f:
    data = json.load(f)

process_samples(data, 0)


