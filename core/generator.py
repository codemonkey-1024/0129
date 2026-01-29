import traceback
from typing import List, Dict, Any
from config import RunningConfig
from core.llm_functions import reasoning, reasoning_simple
import tiktoken


class Generator:
    def __init__(self, cfg: RunningConfig):
        self.llm = cfg.llm_model
        self.encoder = tiktoken.encoding_for_model('gpt-4')

    def reasoning(self, query: str, docs: List[Dict[str, Any]], max_context_token: int = 2048) -> str:
        token_left = max_context_token
        try:
            context = ""
            texts = []
            for i, doc in enumerate(docs):
                text = f"{i+1}. \n{doc.get('title', '')} \n {doc.get('context', '')}"
                if token_left > 0:
                    texts.append(text)
                    token_left -= len(self.encoder.encode(text))

            context = "\n---------------------\n".join(texts)
            context = self.encoder.decode(self.encoder.encode(context)[:max_context_token])
            answer = reasoning(self.llm, query, context)
            return answer
        except Exception as e:
            traceback.format_exc()
            return ""


    def reasoning_with_path(self, query: str, path: List, max_context_token: int = 2048) -> str:
        try:
            context = ""
            texts = []
            for i, path in enumerate(path):
                text = f"{path.format_string}"
                if (max_context_token - len(self.encoder.encode(text))) > 0:
                    texts.append(text)
                    max_context_token -= len(self.encoder.encode(text))

            context = "\n---------------------\n".join(texts)
            answer = reasoning(self.llm, query, context)
            return answer
        except Exception as e:
            traceback.format_exc()
            return ""
