# -*- coding: utf-8 -*-

from __future__ import annotations
import random
from collections import defaultdict

import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterator

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Project Modules
from config import TrainingConfig, RunningConfig
from core.ragindex import RagIndex
from core.graph_diffusion_retriever import CoHITSRetriever
from kg_rl.path_scorer import DynamicPathScorer, ScorerAdapter
from core.utils.tools import read_jsonl
from kg_rl.env import GraphEnv, rollout
from kg_rl.policy import Policy
from kg_rl.reward_funcs import make_traj_reward_fn
from kg_rl.grpo_trainer import GRPOTrainer
from kg_rl.eval_utils import eval_on_one_dataset, eval_one_epoch



random.seed(42)
# =========================================================================
# 1. 数据环境与加载 (Data Runtime & Loading)
# =========================================================================

@dataclass
class DatasetRuntime:
    """封装单个数据集的运行时组件：环境、数据、奖励函数"""
    name: str
    env: GraphEnv
    questions: List[Dict]
    reward_fn: Any
    prefix_score_fn: Any

    cfg: Any = None
    idx: Any = None



def load_dataset_runtime(name: str, device: str, max_samples=None, mode="training", cfg_input=None) -> Optional[DatasetRuntime]:
    """加载单个数据集资源"""
    try:
        if cfg_input is not None:
            cfg = cfg_input
        elif mode == "training":
            cfg = TrainingConfig(dataset_name=name)
        else:
            cfg = RunningConfig(dataset_name=name)
        # 索引与检索器初始化
        idx = RagIndex(cfg)
        idx.index(cfg.dataset_name)


        # 加载问题集
        q_path = Path(f"output/{name}/questions.jsonl")
        if not q_path.exists():
            print(f"[Warn] Dataset file missing: {q_path}, preprocess question first")
            raise Exception

        questions = read_jsonl(q_path)
        if max_samples is not None and max_samples < len(questions):
            questions = random.sample(questions, max_samples)
        q_map = {q["question"]: q for q in questions}

        # 构建 RL 组件
        env = GraphEnv(cfg=cfg, idx=idx, max_depth=cfg.max_depth, avoid_cycles=True)
        # 注意：这里传入 top_k_docs 用于奖励计算
        reward_fn = make_traj_reward_fn(idx, q_map, top_k_docs=cfg.top_k_docs)
        prefix_fn = None

        print(f"[Info] Loaded {name}: {len(questions)} samples.")
        return DatasetRuntime(
            name=name,
            env=env,
            questions=questions,
            reward_fn=reward_fn,
            prefix_score_fn=prefix_fn,
            cfg=cfg,
            idx=idx
        )

    except Exception as e:
        print(f"[Error] Failed to load {name}: {e}")
        return None


def load_runtimes(names: List[str], device: torch.device, max_samples = None) -> List[DatasetRuntime]:
    """批量加载数据集"""
    runtimes = []
    for name in names:
        rt = load_dataset_runtime(name, device, max_samples)
        if rt: runtimes.append(rt)
    return runtimes

# =========================================================================
# 2. 训练辅助工具 (Helpers)
# =========================================================================
def get_interleaved_batches(runtimes: List[DatasetRuntime], batch_size: int, max_width: int):
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

def update_training_schedule(trainer: GRPOTrainer, epoch: int):
    """课程学习调度器：动态调整 Shaping 权重和 Entropy"""
    if epoch <= 10:
        trainer.config.temperature = 1.3
    elif epoch <= 15:
        trainer.config.temperature = 1.0
    else:
        trainer.config.temperature = 0.7

    # 2. Entropy 线性衰减 (前10个epoch衰减完毕)
    init_ent, end_ent, decay_steps = 0.03, 0.02, 5
    progress = min(epoch / decay_steps, 1.0)
    curr_ent = init_ent - (init_ent - end_ent) * progress
    trainer.config.entropy_coef = curr_ent




# =========================================================================
# 3. 核心流程 (Train & Eval)
# =========================================================================

def init_components(cfg: TrainingConfig, ref_rt: DatasetRuntime):
    """初始化 Policy, Trainer 和 Scorer"""
    print(f"Init Model: {cfg.pretrained_model_name} (LoRA={cfg.use_lora})")

    base_scorer = DynamicPathScorer(cfg.pretrained_model_name, cfg.use_lora)
    if cfg.train_checkpoint:
        print(f"Loading checkpoint: {cfg.train_checkpoint}")
        base_scorer.load_state_dict(torch.load(cfg.train_checkpoint, map_location='cpu'))
    base_scorer.to(cfg.device)

    policy = Policy(ScorerAdapter(base_scorer).to(cfg.device), temperature=0.9)
    trainer = GRPOTrainer(cfg=cfg, policy=policy, reward_fn=ref_rt.reward_fn)

    return policy, trainer, base_scorer


def run_training_epoch(trainer: GRPOTrainer, runtimes: List[DatasetRuntime], cfg: TrainingConfig,
                       writer: SummaryWriter, epoch: int, global_step: int):
    """训练一个 Epoch"""
    logs_buffer = []



    # 混合 Batch 迭代
    batch_iter = get_interleaved_batches(runtimes, cfg.questions_per_update, cfg.beam_width)
    total_est = sum(len(rt.questions) for rt in runtimes) // cfg.questions_per_update

    pbar = tqdm(batch_iter, total=total_est, desc=f"Ep {epoch} Train")

    for rt, queries, start_nodes in pbar:
        # 动态切换上下文 (Reward Fn & Prefix Fn)
        trainer.reward_fn = rt.reward_fn
        trainer.prefix_score_fn = rt.prefix_score_fn

        logs = trainer.train_step(
            env=rt.env,
            start_nodes=start_nodes,
            queries=queries,
            samples_per_start=cfg.samples_per_start
        )
        logs_buffer.append(logs)

        # 写入 TensorBoard
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"train/{k}", v, global_step)

        pbar.set_postfix(
            ds=rt.name,
            loss=f"{logs.get('loss/total', 0):.2f}"
        )
        global_step += 1
    pbar.close()
    return logs_buffer, global_step


def run_evaluation(policy: Policy, runtimes: List[DatasetRuntime], cfg: TrainingConfig) -> tuple:
    """在所有评估数据集上运行测试"""
    if not runtimes: return 0.0

    print(f"Evaluating {len(runtimes)} datasets...")
    recalls = []
    rank_scores = []

    for rt in runtimes:
        eval_qs = rt.questions
        original_qs = rt.questions
        rt.questions = eval_qs

        recall, ndcg = eval_one_epoch(cfg, policy, rt)
        recalls.append(recall)
        rank_scores.append(ndcg)
        print(f"-> {rt.name}: recall: {recall:.4f} | rank score: {ndcg:.4f}")

        rt.questions = original_qs  # 恢复
        torch.cuda.empty_cache()

    return sum(recalls) / len(recalls) if recalls else 0.0, sum(rank_scores) / len(rank_scores) if rank_scores else 0.0


# =========================================================================
# 4. 主程序入口 (Main)
# =========================================================================

def main():
    cfg = TrainingConfig()
    print(f"Device: {cfg.device}")

    # 1. 准备数据
    train_rts = load_runtimes(cfg.train_datasets, cfg.device)
    eval_rts = load_runtimes(cfg.eval_datasets, cfg.device)

    if not train_rts:
        print("No training datasets found. Exiting.")

    # 创建 Dev 集
    dev_rts = [] #load_runtimes([cfg.train_datasets[0]], cfg.device, max_samples=100)


    # 2. 初始化模型
    policy, trainer, base_scorer = init_components(cfg, eval_rts[0])

    # 3. 准备日志与保存路径
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/log_{run_id}')
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)


    best_scores = defaultdict(float)
    for rt in eval_rts:
        best_scores[rt.name] = -1.0
    best_score = -1.0
    global_step = 0

    # 4. 训练主循环
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n{'=' * 10} Epoch {epoch} / {cfg.epochs} {'=' * 10}")

        # 4.1 调度更新
        update_training_schedule(trainer, epoch)

        # # 开始之前先验证一下
        # if eval_rts:
        #     eval_score = run_evaluation(policy, eval_rts, cfg, max_samples=200)
        #     writer.add_scalar("eval/score", eval_score, epoch)


        # 4.2 训练阶段
        [random.shuffle(rt.questions) for rt in train_rts]

        for rt in train_rts:
            logs, global_step = run_training_epoch(trainer, [rt], cfg, writer, epoch, global_step)

            if logs:
                avg_loss = np.mean([l.get('loss/total', 0.0) for l in logs])
                avg_reward = np.mean([l.get('reward/total_mean', 0) for l in logs])
                print(f"\n[Train Summary] Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f}")

            # 4.3 保存最新模型
            torch.save(base_scorer.state_dict(), ckpt_dir / "scorer_last.pt")

            # 4.5 测试阶段 (Eval)
            if eval_rts:
                for rt in eval_rts:
                    recall_score, rank_scores = run_evaluation(policy, [rt], cfg)

                    if rank_scores > best_scores[rt.name]:
                        print(f"New Best Model on {rt.name}: {recall_score:.4f}, rank_score: {rank_scores}")
                        best_scores[rt.name] = rank_scores
                        torch.save(base_scorer.state_dict(), ckpt_dir / f"scorer_best_on_{rt.name}.pt")

            # 清理显存
            torch.cuda.empty_cache()

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
