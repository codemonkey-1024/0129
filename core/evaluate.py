import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from core.evaluate_tools.hotpot_evaluate_v1 import eval_linear_rag_metric
logger = logging.getLogger(__name__)
import re
import string
from core.utils.my_llm import LLMClient


def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

class Evaluator:
    def __init__(self, llm_model: LLMClient, predictions):
        self.llm_model = llm_model

        self.prediction_results = predictions
    
    def calculate_llm_accuracy(self,pre_answer,gold_ans):
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
        Generated answer: {pre_answer}
        Gold answer: {gold_ans}

        The generated answer should be considered correct if it:
        1. Contains the key information from the gold answer
        2. Is factually accurate and consistent with the gold answer
        3. Does not contain any contradicting information

        Respond with ONLY 'correct' or 'incorrect'.
        Response:
        """
        response = self.llm_model.call_llm(user_prompt)
        if "incorrect" not in response.strip().lower():
            return 1.0
        else:
            return 0.0

    def calculate_contain(self,pre_answers,gold_ans):
        if pre_answers is None or pre_answers == "" or (isinstance(pre_answers, str) and pre_answers.strip() == ""):
            return 0            
        if gold_ans is None or gold_ans == "" or (isinstance(gold_ans, str) and gold_ans.strip() == ""):
            return 0
        s1 = normalize_answer(pre_answers)
        s2 = normalize_answer(gold_ans)
        if s2 in s1:
            return 1
        else:
            return 0
    def evaluate_sig_sample(self,idx,prediction):
        pre_answer = prediction["pred_answer"]
        gold_ans = prediction["gold_answer"]
        # llm_acc = 0.0
        llm_acc = self.calculate_llm_accuracy(pre_answer, gold_ans)
        contain_acc = self.calculate_contain(pre_answer, gold_ans)
        return idx, llm_acc, contain_acc

    def evaluate(self,max_workers):
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx 
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            pbar = tqdm(total=len(futures), desc="Evaluating samples", unit="sample")
            for future in as_completed(futures):
                idx, llm_acc, contain_acc = future.result()
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc

                metrics = eval_linear_rag_metric(self.prediction_results)

                total_llm_score += llm_acc
                total_contain_score += contain_acc
                completed += 1
                current_llm_acc = total_llm_score / completed
                current_contain_acc = total_contain_score / completed

                metrics['LLM_Acc'] = current_llm_acc
                metrics['Contain_Acc'] = current_contain_acc
                pbar.set_postfix(metrics)
                pbar.update(1)
            pbar.close()

        llm_accuracy = sum(llm_scores) / len(llm_scores)
        contain_accuracy = sum(contain_scores) / len(contain_scores)

        logger.info(f"Evaluation Results:")
        logger.info(f"  LLM Accuracy: {llm_accuracy:.4f} ({sum(llm_scores)}/{len(llm_scores)})")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.4f} ({sum(contain_scores)}/{len(contain_scores)})")
        return metrics