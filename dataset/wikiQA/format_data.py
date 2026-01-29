import json

with open('sample.json', encoding='utf-8') as f:
    data = json.load(f)

corpus = []
questions = []
d_index = 0
q_index = 0
title2sents = {}
for d in data:
    for title, sents in d['context']:
        title2sents[title] = sents
        context = "\n".join(sents)
        if context.strip() == "":
            continue
        if len(context) < 50:
            continue
        if context.endswith("to:"):
            continue
        corpus.append({
            "title": title,
            "context": context,
            "id": d_index,
            'meta': {
                "title": title
            }
        })
        d_index += 1

    # support_ents = []
    # for evi in d['evidences']:
    #     support_ents.append(evi[0])
    #     support_ents.append(evi[2])
    support_facts = []

    for t in d['supporting_facts']:
        support_facts.append({
            'title': t[0],
            "sentence": title2sents[t[0]][t[1]],
            "doc": '\n'.join(title2sents[t[0]])
        })

    questions.append({
        "question": d['question'],
        "answer": d['answer'],
        "id": q_index,
        'support_facts': support_facts
    })
    q_index += 1




with open('corpus.jsonl', 'w', encoding='utf-8') as f:
    for d in corpus:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

with open('questions.jsonl', 'w', encoding='utf-8') as f:
    for q in questions:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

