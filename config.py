from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from core.utils.my_llm import LLMClient, ClientConfig
from llm_config import LLM_CONFIG
from redis import Redis
from core.utils.embeddings import CustomEmbedding
from loguru import logger
import sys

class Config:
    working_dir: str = "./output"
    pretrained_model_name: str = "distilbert-base-uncased"
    embedding_model = CustomEmbedding(
        api_key="Replace your API KEY",
        base_url="Replace your API KEY",
        model_name='Qwen/Qwen3-Embedding-0.6B')

    # cache
    redis_cfg = REDIS_CONFIG = {
        "host": "127.0.0.1",
        "port": 6579,
        "decode_responses": True,
    }
    llm_model: LLMClient = LLMClient(endpoints=LLM_CONFIG['endpoints'],
                                     client_cfg=ClientConfig(cache=Redis(**redis_cfg)))
    enable_tqdm: bool = False,
    merge_doc: bool = False,
    merge_doc_size: int = 4

@dataclass
class RunningConfig(Config):
    dataset_name: str = "wikiQA"
    checkpoint: str = "./checkpoints/model.pt"
    explore_sources: List[str] = field(default_factory=lambda: ["EE", "EP"])
    rank_mode: str = "scorer" # deprecated
    max_cand: int = 32
    init_top_k_entities: int = 10 # deprecated
    beam_width: int = 2
    max_depth: int = 6
    top_k_docs: int = 100
    max_context_token: int = 1024*5
    batch_size: int = 32 # deprecated
    max_workers: int = 5
    device: str = "cuda"
    do_reasoning: bool = True
    use_cache: bool = False
    ablation: bool = False

    sample: bool = False
    sample_ratio: float = 1.0

    do_dilute_on_EE: bool = False
    dilute_ratio_on_EE: float = 1.0

    do_dilute_on_EP: bool = False
    dilute_ratio_on_EP: float = 0.7

    eval_step: int = 1
    noise_lambda: float = 0.05


@dataclass
class TrainingConfig(RunningConfig):
    train_checkpoint: str = ""
    explore_sources: List[str] = field(default_factory=lambda: ["EE", "EP"])
    samples_per_start: int = 1
    max_cand: int = 24
    beam_width: int = 2
    epochs: int = 40
    max_depth: int = 4
    top_k_docs: int = 30
    questions_per_update: int = 24
    device: str = "cuda:2"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    FT_layer: int = 2  # deprecated
    train_datasets: List[str] = field(default_factory=lambda: [f"batch_1"])
    eval_datasets: List[str] = field(default_factory=lambda: ["wikiQA_exp"])


time_statistic = defaultdict(list)



# Config Logger
logger.remove()
logger.add(
    sys.stdout,
    level="ERROR",  # "WARNING",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level:<8}</level> | "
           "<cyan>{name:<15}</cyan>:<cyan>{function:<15}</cyan>:<cyan>{line:<4}</cyan> - "
           "<level>{message}</level>"
)
