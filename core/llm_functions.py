from core.prompt_list import *
import re

from core.utils.my_llm import LLMClient
import re
from typing import List, Dict, Tuple, Any
import traceback
import json
import re

def deduplicates_list(lst: List[Any]) -> List[Any]:
    """
    去除列表中的重复元素，并保持原有顺序。

    参数:
    lst (List[Any]): 需要去重的列表，列表中的元素可以是任何可序列化的类型。

    返回:
    List[Any]: 去重后的列表，保持原有顺序。
    """
    if not isinstance(lst, list):
        raise TypeError("输入参数必须是一个列表")

    unique_dict = {}
    for item in lst:
        # 将每个元素序列化为字符串作为键，确保字典的键是唯一的
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in unique_dict:
            unique_dict[key] = item

    # 返回去重后的列表
    return list(unique_dict.values())

def remove_think_tags(text):
    # 使用正则表达式匹配 <think> 标签及其内容
    pattern = r'<think>.*?</think>'
    # 使用 re.DOTALL 使 . 匹配换行符
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text


def _parse_llm_response_for_triples_extraction(response_text):
    """
    解析优化后的知识图谱响应文本，支持带数字序号的格式

    参数:
        response_text (str): 原始响应文本

    返回:
        tuple: (entities, relations) 实体列表和关系列表

    示例输入:
        1.[entity | ID | Type | "Exact Mention" | Contextual Description]
        2.[relation | SourceID | RelationType | TargetID | "EvidenceSpan"]
    """
    try:
        entities = []
        relations = []
        entity_ids = set()
        entity_name_map = {}

        response_text = remove_think_tags(response_text)
        # 预处理文本
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        for line in lines:
            # 去除行首的数字序号（如 "1."）
            modified_line = re.sub(r'^\d+\.', '', line)
            # 去除方括号并清理格式
            clean_line = modified_line.replace("[", "", 1).replace("]", "", 1).strip()
            elements = [e.strip() for e in clean_line.split("| ")]

            if not elements:
                continue

            if elements[0] == "entity":
                if len(elements) != 5:
                    raise ValueError(f"实体格式错误: {line}")

                entity = {
                    "id": elements[1],
                    "type": elements[2],
                    "mention": elements[3].strip('"'),
                    "description": elements[4]
                }
                entities.append(entity)
                entity_ids.add(entity["id"])
                entity_name_map[entity["id"]] = entity["mention"]

            elif elements[0] == "relation":
                if len(elements) != 5:
                    raise ValueError(
                        f"{line} -> Format Error. Should be [relation | SubjectEntityID | Predicate | ObjectEntityID | \"Evidence or supporting phrase\"].")

                source_id, rel_type, target_id, evidence = elements[1:5]

                if source_id not in entity_ids:
                    raise ValueError(f"SubjectEntityID '{source_id}' not in Entities")
                if target_id not in entity_ids:
                    raise ValueError(f"ObjectEntityID '{target_id}' not in Entities")

                relations.append({
                    "source": source_id,
                    "target": target_id,
                    "relation": rel_type,
                    "evidence": evidence.strip('"'),
                    "source_name": entity_name_map[source_id],
                    "target_name": entity_name_map[target_id]
                })

        if len(entity_ids) != len(entities):
            raise ValueError("存在重复的实体ID")

        return entities, relations
    except Exception as e:
        raise e


def extract_triples_from_doc(llm, text):
    _token = dict()
    # 抽取实体
    stage_1_prompt = prompt_for_extract_entities(text)
    entity_list = []
    for i in range(3):
        entities = llm.call_llm(stage_1_prompt, post_process=_parse_entities_from_response, token_counter=_token,
                                cache_token=i)
        entity_list.extend(entities)
    entity_list = list({entity['mention']: entity for entity in entity_list}.values())
    for i, ent in enumerate(entity_list):
        ent['id'] = f"E{i + 1}"

    relations = []
    if len(entity_list) > 0:
        # 抽取关系
        entities_block = "\n".join(
            [f'[entity | {ent["id"]} | {ent["type"]} | "{ent["mention"]}" | {ent["description"]}]' for ent in
             entity_list])
        stage_2_prompt = prompt_for_extract_relations(text, entities_block)
        relations = llm.call_llm(stage_2_prompt, post_process=_parse_relations_from_response, token_counter=_token,
                                 entities=entity_list)

    return entity_list, relations


def extract_triples_from_doc_old(llm, text):
    prompt = prompt_for_KG_contruction_auto_type(text)
    _token = dict()
    response = llm.call_llm(prompt, post_process=_parse_llm_response_for_triples_extraction, token_counter=_token)
    return response


def _format_context(paths, use_path=True, use_text=True):
    # 提取三元组并格式化，保持顺序去重
    triples = [
        f"<{triple['begin']['mention']} | {triple['r'].replace('_', ' ')} | {triple['end']['mention']}>"
        for path in paths for triple in path.get('relations', [])
    ]
    format_triples = list(dict.fromkeys(triples))

    # 提取相关句子，保持顺序去重
    related_sents = [
        sent for path in paths for sent in path.get('context_sentences', [])
    ]
    related_sents = list(dict.fromkeys(related_sents))

    # 拼接为字符串，只在有内容时添加标题
    parts = []
    if use_path and format_triples:
        parts.append("Triples:\n" + "\n".join(format_triples))
    if use_text and related_sents:
        parts.append("Related Text:\n" + "\n".join(related_sents))
    context_info = "\n\n".join(parts)
    return context_info


def _parse_llm_response_for_topic_entity_extraction(response_text):
    """
    Parameters:
        response_text (str): The original response text.

    Returns:
        tuple: (entities, relations) A list of entities and a list of relations.

    Example input:
        1.[entity | ID | Type | "Exact Mention" | Contextual Description]
        2.[relation | SourceID | RelationType | TargetID | "EvidenceSpan"]
    """
    entities = []
    relations = []
    entity_ids = set()
    entity_name_map = {}

    lines = [line.strip() for line in response_text.split("\n") if line.strip()]

    for line in lines:
        # Remove the numerical index at the beginning of the line(e.g.,"1.").
        modified_line = re.sub(r'^\d+\.', '', line)
        # Remove square brackets and clean up the format
        clean_line = modified_line.replace("[", "", 1).replace("]", "", 1).strip()
        elements = [e.strip() for e in clean_line.split("|")]

        if not elements:
            continue

        if elements[0] == "entity":
            if len(elements) != 5:
                raise ValueError(f"The entity format is incorrect:{line}")

            entity = {
                "id": elements[1],
                "type": elements[2],
                "mention": elements[3].strip('"'),
                "description": elements[4]
            }
            entities.append(entity)
            entity_ids.add(entity["id"])
            entity_name_map[entity["id"]] = entity["mention"]

        elif elements[0] == "relation":
            if len(elements) != 5:
                raise ValueError(f"The relationship format is incorrect:{line}")

            source_id, rel_type, target_id, evidence = elements[1:5]

            if source_id not in entity_ids:
                raise ValueError(f"The source entity ID'{source_id}'does not exist.")
            if target_id not in entity_ids:
                raise ValueError(f"The target entity ID'{target_id}'does not exist.")

            relations.append({
                "source": source_id,
                "target": target_id,
                "relation": rel_type,
                "evidence": evidence.strip('"'),
                "source_name": entity_name_map[source_id],
                "target_name": entity_name_map[target_id]
            })

    if len(entity_ids) != len(entities):
        raise ValueError("There are duplicate entity IDs.")

    return entities


def _parse_llm_response_for_score_triples(response_text, **kwargs):
    try:
        candidates = kwargs.get('candidates', [])

        # 支持任意浮点数（如 0, 0.0, 0.25, 1, 1.0）
        pattern = re.compile(
            r'<([^>]+)>\s*:\s*(\d+(?:\.\d+)?)'
        )

        results = []
        for match in pattern.finditer(response_text):
            triplet_part, score_str = match.groups()

            # 安全 split，避免 LLM 输出不标准导致 ValueError
            parts = [p.strip() for p in triplet_part.split('|')]
            if len(parts) != 3:
                continue  # 或者 raise ValueError("Invalid triple format")

            head, relation, tail = parts
            score = float(score_str)

            results.append({
                'head': head,
                'relation': relation,
                'tail': tail,
                'score': score
            })

        # 如果数量完全匹配，直接返回
        if len(results) == len(candidates):
            return results

        # 否则补全候选，保证输出与 candidates 对齐
        outputs = [
            {
                'head': candidate['begin']['mention'],
                'relation': candidate['r'],
                'tail': candidate['end']['mention'],
                'score': 0.0
            }
            for candidate in candidates
        ]

        # 建立映射表
        tri2scores = {
            (r['head'], r['relation'], r['tail']): r['score']
            for r in results
        }

        # 覆盖已有分数（注意 0.0 也要覆盖）
        for tri in outputs:
            key = (tri['head'], tri['relation'], tri['tail'])
            if key in tri2scores:
                tri['score'] = tri2scores[key]

        return outputs

    except Exception as e:
        raise e


def score_triples(llm, query, paths, batch_size=20):
    all_results = []
    # Process candidate triples in batches
    for i in range(0, len(paths), batch_size):
        batch_path = paths[i:i + batch_size]
        container = [r for p in batch_path for r in p.relations[:-1]]
        candidates = [p.relations[-1] for p in batch_path]
        batch_results = _score_triples_in_batch(llm, query, container, candidates)
        all_results.extend(batch_results)
    assert len(all_results) == len(paths)
    return all_results


def _score_triples_in_batch(llm, query, container, candidates):
    format_container = [f"<{triple['begin']['mention']} | {triple['r']} | {triple['end']['mention']}>" for triple in
                        container]
    format_candidates = [f"<{triple['begin']['mention']} | {triple['r']} | {triple['end']['mention']}>" for triple in
                         candidates]
    existed_text = "\n".join(deduplicates_list(format_container)) if len(format_container) else "None"
    text = "\n".join(format_candidates)
    prompt = prompt_for_score_triples(query, text, existed_text)

    _token = {'input_token': 0, 'output_token': 0}
    results = llm.call_llm(prompt, post_process=_parse_llm_response_for_score_triples, candidates=candidates,
                           token_counter=_token)

    return results


def _parse_llm_response_for_eval_sufficiency_with_llm(response_text, **kwargs):
    if 'yes' in response_text.lower():
        return True
    elif 'no' in response_text.lower():
        return False
    else:
        raise ValueError("The answer format is incorrect!")


def eval_sufficiency_with_llm(llm, paths, query):
    context_info = _format_context(paths)
    prompt = prompt_for_eval_sufficiency(context_info, query)
    _token = {'input_token': 0, 'output_token': 0}
    results = llm.call_llm(prompt, post_process=_parse_llm_response_for_eval_sufficiency_with_llm, token_counter=_token)

    return results


def _parse_llm_response_for_missing_knowledge_identify(response_text, **kwargs):
    """
    解析不同格式的 Final Answer，支持星号、下划线、加粗等变体。

    参数:
    output_text (str): 模型生成的完整输出文本。

    返回:
    str: 提取的答案文本，若未找到则返回空字符串。
    """
    # 正则表达式匹配标签前后的星号/下划线，并允许跨行内容
    questions = response_text.strip().split('\n')
    assert len(questions) > 0
    return questions


def missing_knowledge_identify(llm, question, path):
    context_info = _format_context(path, use_text=False)
    prompt = prompt_for_missing_knowledge_identify(context_info, question)
    sub_questions = llm.call_llm(prompt, post_process=_parse_llm_response_for_missing_knowledge_identify)
    return sub_questions


def missing_knowledge_extraction(llm, question, path, sub_questions, context, **kwargs):
    try:
        context_info = _format_context(path, use_text=False)
        prompt = prompt_for_missing_knowledge_extraction_refine(context_info, sub_questions, context)
        entities, relations = llm.call_llm(prompt, post_process=_parse_llm_response_for_triples_extraction)
        return entities, relations
    except Exception as e:
        raise e


def _parse_llm_response_for_multi_relation(response_text: str, **kwargs) -> list:
    """
    解析实体对关系抽取模型输出，将其转为结构化对象列表。
    此版本能处理包含多个由'|'分割的关系。
    """
    relations_data = []
    lines = response_text.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
        try:
            # 确保这里的分隔符 '||' 与你生成 entity_pairs_str 时使用的一致
            pair_str, relation_str = line.rsplit(':', 1)
            if "||" in pair_str:
                head, tail = [x.strip() for x in pair_str.split('||', 1)]
            elif "|" in pair_str:
                head, tail = [x.strip() for x in pair_str.split('|', 1)]

            relation_str = relation_str.strip()

            if 'none' in relation_str.lower():
                relations_list = []
            else:
                relations_list = [rel.strip() for rel in relation_str.split('|')]

            relations_data.append({
                'head': head,
                'tail': tail,
                'relations': relations_list
            })
        except Exception as e:
            print(f"Skipping line due to parse error: '{line}'. Error: {e}")

    return relations_data


def complete_relations(llm, text, entity_pairs, query):
    entity_pairs_str = ""
    for entity_pair in entity_pairs:
        entity_pairs_str += f"{entity_pair[0].strip()} || {entity_pair[1].strip()}\n"
    prompt = prompt_for_focused_multi_relation_completion(text, entity_pairs_str, query)
    return llm.call_llm(prompt, post_process=_parse_llm_response_for_multi_relation)


def summary_answer(llm, text, question):
    prompt = prompt_for_summary_answer(text, question)
    return llm.call_llm(prompt)


def extract_content(keyword: str, text: str, mode: str = "last"):
    """
    Extract content inside <keyword>...</keyword> tags.
    mode:
        - "last": extract the last matched block
        - "outer": extract the outermost block
        - "inner": extract the innermost block
    """
    # Pattern to find all <keyword>...</keyword> blocks, including nested
    pattern = fr"<{keyword}>(.*?)</{keyword}>"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    if not matches:
        return None

    if mode == "last":
        return matches[-1].strip()

    elif mode == "outer":
        # Match the longest span (greedy)
        pattern_outer = fr"<{keyword}>(.*)</{keyword}>"
        match = re.search(pattern_outer, text, flags=re.DOTALL)
        return match.group(1).strip() if match else None

    elif mode == "inner":
        # The innermost is simply the shortest match, i.e., the first in non-greedy matches list
        return matches[0].strip()

    else:
        raise ValueError("mode must be one of: 'last', 'outer', 'inner'")


def _parse_llm_response_for_coref_disambiguation(response_text, **kwargs):
    res = extract_content("TEXT", response_text)
    if res:
        return res
    else:
        raise Exception("Response format error.")


def topic_entity_extraction(llm, text):
    # prompt = prompt_for_entity_extraction(text)
    prompt = prompt_for_preprocess_query_more_ent(text)
    _token = {'input_token': 0, 'output_token': 0}
    entities = llm.call_llm(prompt, post_process=_parse_entities_from_response, token_counter=_token)
    return entities


def coref_disambiguation(llm: LLMClient = None, title: str = None, pre_chunk: str = None, current_chunk: str = None):
    prompt = prompt_for_coref_disambiguation(title, pre_chunk, current_chunk)
    res = llm.call_llm(prompt, post_process=_parse_llm_response_for_coref_disambiguation)
    return res


def _parse_entities_from_response(response_text: str, **kwargs) -> List[Dict]:
    """
    仅解析“实体”行，返回实体列表（不解析关系）。
    允许行首带数字序号，如 `1.[entity | ...]`。
    会做：
      - 去除 <think> 等思考标签（依赖 remove_think_tags）
      - 去除行首数字序号
      - 严格校验 entity 行的 5 段结构
      - 检查实体 ID 重复
    返回:
      [
        {"id": "...", "type": "...", "mention": "...", "description": "..."},
        ...
      ]
    """
    try:
        entities: List[Dict] = []
        entity_ids = set()

        # 清洗

        text = extract_content("ENTITY", response_text)
        if not text:
            return []
        text = remove_think_tags(text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for line in lines:
            # 去掉行首数字序号（如 "1."）
            modified = re.sub(r'^\d+\.\s*', '', line)

            # 只处理以 [entity 开头的行（更稳健）
            if not modified.lstrip().startswith("[entity"):
                continue

            # 取出方括号内的内容；如果没有匹配，就退回到整行去掉首尾[]的处理
            m = re.search(r'\[(.+?)\]\s*$', modified)
            inside = m.group(1) if m else modified.strip().lstrip('[').rstrip(']')
            # 使用 '|' 分隔再逐个 strip
            parts = [p.strip() for p in inside.split('|')]

            if not parts or parts[0] != "entity":
                continue

            if len(parts) != 5:
                raise ValueError(f"实体格式错误: {line}")

            _, ent_id, ent_type, ent_mention, ent_desc = parts
            ent_mention = ent_mention.strip('"')

            if ent_id in entity_ids:
                raise ValueError(f"存在重复的实体ID: {ent_id}")

            entities.append({
                "id": ent_id,
                "type": ent_type,
                "mention": ent_mention,
                "description": ent_desc
            })
            entity_ids.add(ent_id)

        # 终态校验：去重检查
        if len(entity_ids) != len(entities):
            raise ValueError("存在重复的实体ID")

        return entities

    except Exception as e:
        raise e


def _parse_relations_from_response(response_text: str, **kwargs) -> List[Dict]:
    """
    仅解析“关系”行，返回关系列表。
    依赖上一步的 entities 进行严格 EID 校验与 name 映射。
    允许行首带数字序号，如 `2.[relation | ...]`。
    返回:
      [
        {
          "source": "E1", "target": "E2", "relation": "approved", "evidence": "...",
          "source_name": "...", "target_name": "..."
        },
        ...
      ]
    """
    try:
        entities = kwargs.get('entities')
        relations: List[Dict] = []

        # 用 entities 构造 id 集与 name 映射
        entity_ids = {e["id"] for e in entities}
        entity_name_map = {e["id"]: e.get("mention", "") for e in entities}

        # 清洗
        text = remove_think_tags(response_text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for line in lines:
            # 去掉行首数字序号（如 "2."）
            modified = re.sub(r'^\d+\.\s*', '', line)

            # 仅处理 relation 行
            if not modified.lstrip().startswith("[relation"):
                continue

            # 提取方括号内文本
            m = re.search(r'\[(.+?)\]\s*$', modified)
            inside = m.group(1) if m else modified.strip().lstrip('[').rstrip(']')
            parts = [p.strip() for p in inside.split('|')]

            if not parts or parts[0] != "relation":
                continue

            if len(parts) != 5:
                raise ValueError(
                    f"{line} -> Format Error. Should be [relation | SubjectEntityID | Predicate | ObjectEntityID | \"Evidence or supporting phrase\"]."
                )

            _, source_id, rel_type, target_id, evidence = parts

            # EID 严格校验
            if source_id not in entity_ids:
                raise ValueError(f"SubjectEntityID '{source_id}' not in Entities")
            if target_id not in entity_ids:
                raise ValueError(f"ObjectEntityID '{target_id}' not in Entities")

            relations.append({
                "source": source_id,
                "target": target_id,
                "relation": rel_type,
                "evidence": evidence.strip('"'),
                "source_name": entity_name_map.get(source_id, ""),
                "target_name": entity_name_map.get(target_id, ""),
            })

        return relations

    except Exception as e:
        raise e


def _parse_llm_response_for_reasoning(response_text: str, **kwargs):
    res = extract_content('ANSWER', response_text)
    if res:
        return res
    else:
        raise Exception


def reasoning(llm, question, text):
    prompt = prompt_for_reasoning(question, text)
    res = llm.call_llm(prompt, post_process=_parse_llm_response_for_reasoning)
    return res


def _parse_llm_response_for_reasoning_simple(response_text: str, **kwargs):
    res = response_text.split("Answer:")[-1].strip()
    if res:
        return res
    else:
        raise Exception

def reasoning_simple(llm, question, text):
    system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""

    prompt = prompt_for_read_and_reasoning(question, text)
    res = llm.call_llm(prompt, system=system_prompt, post_process=_parse_llm_response_for_reasoning_simple)
    return res



def extract_entities_relations_from_doc(llm, text):
    # 抽取实体
    stage_1_prompt = prompt_for_extract_entities(text)
    entity_list = []
    for i in range(3):
        try:
            entities = llm.call_llm(stage_1_prompt, post_process=_parse_entities_from_response, cache_token=i)
            entity_list.extend(entities)
        except Exception as e:
            traceback.print_exc()
    entity_list = list({entity['mention']: entity for entity in entity_list}.values())
    for i, ent in enumerate(entity_list):
        ent['id'] = f"E{i + 1}"

    relations = []
    if len(entity_list) > 0:
        # 抽取关系
        entities_block = "\n".join(
            [f'[entity | {ent["id"]} | {ent["type"]} | "{ent["mention"]}" | {ent["description"]}]' for ent in
             entity_list])
        stage_2_prompt = prompt_for_extract_relations(text, entities_block)
        try:
            relations = llm.call_llm(stage_2_prompt, post_process=_parse_relations_from_response, entities=entity_list)
        except Exception as e:
            traceback.print_exc()
    return entity_list, relations


def _parse_llm_response_extract_evidence_entities(response_text: str, **kwargs):
    res = extract_content('ENTITY', response_text)
    pattern = r'(.+?):\s*([\d.]+)'
    matches = re.findall(pattern, res)
    entities = [(name, float(score)) for name, score in matches]
    return entities


def extract_evidence_entities(llm, question, text):
    prompt = prompt_for_evidence_entities(question, text)
    entities = llm.call_llm(prompt, post_process=_parse_llm_response_extract_evidence_entities)
    return entities
