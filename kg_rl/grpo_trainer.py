# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import AdamW

from config import TrainingConfig, time_statistic
from core.schema import GraphPath
from .env import GraphEnv, rollout, Trajectory
from .policy import Policy
from .reward_funcs import BatchedRewardFn

# =========================================================================
# 1. 配置类：GRPO 专用配置
# =========================================================================
@dataclass
class GRPOConfig:
    """GRPO (Group Relative Policy Optimization) 超参数配置"""
    # 优化相关
    lr: float = 5e-5  # 纯策略梯度通常需要更小的 LR
    weight_decay: float = 1e-3
    max_grad_norm: float = 1.0

    # 采样相关
    group_size: int = 8  # [关键] 每个 Query 采样的路径数量 (G)
    rollout_workers: int = 8

    # PPO / GRPO 核心
    clip_ratio: float = 0.2  # PPO Clip 范围
    ppo_epochs: int = 2  # 每批数据训练几轮
    batch_size: int = 128  # Update 时的 Batch Size (注意显存)
    grad_accum_steps: int = 2 # 梯度累计步骤
    entropy_coef: float = 0.01  # 鼓励探索
    temperature: float = 1.2


    # 奖励相关
    # 注意：GRPO 对奖励的绝对值不敏感，但对相对大小敏感
    reward_scale: float = 1.0
    beta_kl: float = 0.0  # (可选) 如果有 Reference Model，这里设为 0.01~0.1


# =========================================================================
# 2. 数据结构
# =========================================================================
class MiniBatch(NamedTuple):
    """一个训练 Mini-Batch 所需的数据 (去掉了 Value 相关字段)"""
    states: List[StepState]
    actions: torch.Tensor  # [B]
    old_logps: torch.Tensor  # [B]
    advantages: torch.Tensor  # [B] (经过 Group Norm 后的优势)
    candidates: List[List[int]]  # 候选动作


# =========================================================================
# 3. Trainer 主类 (GRPO)
# =========================================================================
class GRPOTrainer:
    def __init__(
            self,
            cfg: TrainingConfig,
            policy: Policy,
            reward_fn: BatchedRewardFn,
            config: GRPOConfig = GRPOConfig(),
    ):
        self.config = config
        self.device = next(policy.parameters()).device

        self.policy = policy
        self.reward_fn = reward_fn

        print(f"[Info] Initializing GRPO Trainer with Group Size = {config.group_size}")
        print("[Info] No Critic/ValueNet will be loaded.")

        # 优化器：只优化 Policy
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

    # -------------------------------------------------------------------------
    # 核心接口
    # -------------------------------------------------------------------------
    def train_step(
            self,
            env: GraphEnv,
            start_nodes: List[Dict],
            queries: List[str],
            warm_starts: Optional[List[GraphPath]] = None,
            samples_per_start: int = 1
    ) -> Dict[str, Any]:
        """执行一次 GRPO 迭代"""

        # 记录开始时间，用于计算总耗时
        global_start_time = time.time()

        # 强制使用配置中的 group_size
        actual_samples = self.config.group_size

        # 1. 采样 (Rollout) - 生成 G 条路径/Query
        self.policy.eval()


        s = time.time()
        with torch.no_grad():
            trajs = rollout(
                env=env,
                policy=self.policy,
                start_nodes=start_nodes,
                queries=queries,
                warm_starts=warm_starts,
                greedy=False,
                use_grad=False,
                samples_per_start=actual_samples,
                temperature=self.config.temperature
            )

        e = time.time()
        time_statistic['rollout'].append(e - s)

        if not trajs:
            return {"loss": 0.0, "reward/mean": 0.0}

        # 2. 计算奖励 (Raw Rewards)
        # 注意：这里计算的是每条路径的绝对分数
        s = time.time()
        metrics = self._compute_raw_rewards(trajs)
        e = time.time()
        time_statistic['_compute_raw_rewards'].append(e - s)

        # 3. 计算组优势 (Group Relative Advantage)
        # 这是 GRPO 替代 GAE 的核心步骤
        samples = self._compute_group_advantages(trajs)


        # 4. 策略更新 (Policy Optimization)
        s = time.time()
        train_stats = self._update_parameters(env, samples)
        e = time.time()
        time_statistic['_update_parameters'].append(e - s)

        # 记录总的采样时间
        end_time = time.time()
        # print(f"total time: {end_time - global_start_time}")
        # for k, v in time_statistic.items():
        #     print(f"{k}: {sum(v)}")

        for k, v in time_statistic.items():
            time_statistic[k].clear()
        return {**metrics, **train_stats}

    # -------------------------------------------------------------------------
    # 内部组件 1: 奖励计算
    # -------------------------------------------------------------------------
    def _compute_raw_rewards(self, trajs: List[Trajectory]) -> Dict[str, float]:
        """计算基础奖励，不涉及 Baseline"""
        for tr in trajs:
            final_gp = tr.final_path if tr.final_path else GraphPath(tr.start_node)
            path_r, r_details = self.reward_fn(tr.query, [final_gp])
            tr.reward = path_r

        # 统计指标
        rewards = [t.reward for t in trajs]
        return {
            "reward/total_mean": np.mean(rewards) if rewards else 0.0,
            "reward/std": np.std(rewards) if rewards else 0.0,
            "traj/length": np.mean([len(t.actions) for t in trajs]) if trajs else 0.0,
        }

    # -------------------------------------------------------------------------
    # 内部组件 2: GRPO 优势计算 (核心)
    # -------------------------------------------------------------------------
    def _compute_group_advantages(self, trajs: List[Trajectory]) -> List[Dict]:
        """
        计算 Group Relative Advantage:
        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
        """
        flat_samples = []

        # 1. 按 Query 分组
        # 1. 按 Query 分组
        groups = defaultdict(list)
        for tr in trajs:
            groups[(tr.query, tr.start_node['id'])].append(tr)

        # 2. 组内归一化
        for (query, start_node_id), group_trajs in groups.items():
            if not group_trajs: continue

            # 提取该组所有奖励
            # 提示：如果你的 reward_fn 返回的都是 0，这里会出问题。
            # 务必使用 Soft Reward 或 Shaping 让 rewards 有差异。
            rewards = np.array([t.reward for t in group_trajs], dtype=np.float32)

            # [GRPO Magic] 计算组内均值和标准差
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-8

            # 计算优势
            # 好的路径优势为正，差的路径优势为负
            advantages = (rewards - mean_r) / std_r

            # 3. 展开为训练样本
            for i, tr in enumerate(group_trajs):
                T = len(tr.actions)
                if T == 0: continue

                # 整条路径共享同一个最终优势 (Sparse Reward 场景常用)
                # 如果你有 Step-level reward，可以在这里做累加
                path_adv = advantages[i]

                for t in range(T):
                    flat_samples.append({
                        "state": tr.states[t],
                        "action": int(tr.actions[t]),
                        "logp_old": float(tr.log_probs[t].item()),
                        "adv": float(path_adv),
                        "candidates": tr.states[t].candidates if hasattr(tr.states[t], 'candidates') else None
                    })

        return flat_samples

    # -------------------------------------------------------------------------
    # 内部组件 3: 参数更新 (带梯度累积优化版)
    # -------------------------------------------------------------------------
    def _update_parameters(self, env: GraphEnv, samples: List[Dict]) -> Dict[str, float]:
        """
        使用 PPO/GRPO 算法更新策略网络参数。
        集成了梯度累积 (Gradient Accumulation) 以降低显存需求。
        """
        # 切换到训练模式（启用 Dropout 等）
        self.policy.train()

        # 1. [数据准备] 将 List[Dict] 转换为 Tensor，移动到 GPU
        # advs 是已经在采样阶段计算好的优势函数（可能已经过组内归一化）
        advs = torch.tensor([s["adv"] for s in samples], device=self.device)
        old_logps = torch.tensor([s["logp_old"] for s in samples], device=self.device)

        dataset_size = len(samples)
        indices = np.arange(dataset_size)
        stats = defaultdict(list)

        # [配置读取]
        # physical_bs: 物理 Batch Size，决定显存占用
        physical_bs = self.config.batch_size
        # accum_steps: 累积步数，决定等效 Batch Size
        accum_steps = getattr(self.config, "grad_accum_steps", 1)  # 默认 1 即不累积

        # PPO 通常会在同一批数据上训练多个 Epoch
        for _ in range(self.config.ppo_epochs):
            # 打乱数据索引，保证训练的随机性
            np.random.shuffle(indices)

            # [关键步骤 0] Epoch 开始前确保梯度清零
            self.optimizer.zero_grad()

            # 计算总共有多少个 micro-batch
            mini_batches = list(range(0, dataset_size, physical_bs))
            total_mini_batches = len(mini_batches)

            for i, start in enumerate(mini_batches):
                end = min(start + physical_bs, dataset_size)
                batch_idx = indices[start:end]

                # -----------------------------------------------------------
                # [步骤 1] 构造 Micro-Batch (物理小批次)
                # -----------------------------------------------------------
                # 这里的 batch_size 比较小，可以轻松塞进显存
                batch_states = [samples[k]["state"] for k in batch_idx]

                # 获取或重新计算候选动作 (取决于是否有缓存)
                batch_candidates = [
                    samples[k].get("candidates") or env.candidates(samples[k]["state"])
                    for k in batch_idx
                ]

                batch = MiniBatch(
                    states=batch_states,
                    actions=torch.tensor([samples[k]["action"] for k in batch_idx], device=self.device),
                    old_logps=old_logps[batch_idx],
                    advantages=advs[batch_idx],
                    candidates=batch_candidates
                )

                # -----------------------------------------------------------
                # [步骤 2] 前向传播 & Loss 计算
                # -----------------------------------------------------------
                # 这里会进行 Policy 的 Forward，消耗显存的主要步骤
                loss_dict = self._compute_loss(batch)

                # -----------------------------------------------------------
                # [步骤 3] Loss 缩放 (Loss Scaling) -- 梯度累积的核心
                # -----------------------------------------------------------
                # 因为 backward() 是将梯度累加(add)到 .grad 属性中。
                # 如果我们累积 4 次，梯度的总和就是 4 个 batch 的和。
                # 为了维持数学上的“平均”定义，我们需要将 Loss 除以累积步数。
                loss_scaled = loss_dict["loss"] / accum_steps

                # -----------------------------------------------------------
                # [步骤 4] 反向传播 (Backward)
                # -----------------------------------------------------------
                # 此时只计算梯度并累加，不进行参数更新 (optimizer.step)
                loss_scaled.backward()

                # -----------------------------------------------------------
                # [步骤 5] 梯度累积判断 & 参数更新
                # -----------------------------------------------------------
                # 判断条件：
                # 1. 达到了累积步数 ((i+1) % accum_steps == 0)
                # 2. 或者这是数据集的最后一个 batch (防止最后几个样本被丢弃)
                is_update_step = ((i + 1) % accum_steps == 0) or ((i + 1) == total_mini_batches)

                if is_update_step:
                    # A. 梯度裁剪 (Gradient Clipping)
                    # 对累积好的梯度进行裁剪，防止梯度爆炸
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm
                    )

                    # B. 参数更新 (Optimizer Step)
                    # 使用累积的梯度更新权重
                    self.optimizer.step()

                    # C. 梯度清零 (Zero Grad)
                    # 更新完后，清空梯度，为下一轮累积做准备
                    self.optimizer.zero_grad()

                    # D. 记录梯度范数 (仅在更新步骤记录)
                    stats["train/grad_norm"].append(grad_norm.item())

                # -----------------------------------------------------------
                # [步骤 6] 日志记录
                # -----------------------------------------------------------
                # 注意：记录的是原始 loss (loss_dict["loss"])，而不是 scaled loss
                # 这样在 Tensorboard 上看到的数值才直观
                stats["loss/total"].append(loss_dict["loss"].item())
                stats["loss/policy"].append(loss_dict["policy_loss"].item())
                stats["entropy"].append(loss_dict["entropy"].item())
                stats["ppo/clip_frac"].append(loss_dict["clip_frac"].item())

        # 返回所有步数的平均值用于打印
        return {k: np.mean(v) for k, v in stats.items()}


    def _compute_loss(self, batch: MiniBatch) -> Dict[str, torch.Tensor]:
        """计算 Policy Loss (PPO Clipped Objective)"""

        # 1. Forward Policy
        paths = [s.path for s in batch.states]
        queries = [s.query for s in batch.states]

        # policy forward: [B, Action_Space]
        logits_list = self.policy(queries, batch.candidates)

        log_probs_new = []
        entropies = []

        for logits, act in zip(logits_list, batch.actions):
            dist = Categorical(logits=logits)
            log_probs_new.append(dist.log_prob(act))
            entropies.append(dist.entropy())

        log_probs_new = torch.stack(log_probs_new)
        entropy = torch.stack(entropies).mean()

        # 2. PPO Loss
        ratio = torch.exp(log_probs_new - batch.old_logps)
        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * batch.advantages

        # PPO 目标是最大化，loss 是最小化负值
        policy_loss = -torch.min(surr1, surr2).mean()

        # 3. Total Loss
        # GRPO 中没有 Value Loss
        total_loss = policy_loss - self.config.entropy_coef * entropy

        # 监控指标
        clip_frac = (torch.abs(ratio - 1.0) > self.config.clip_ratio).float().mean()

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "clip_frac": clip_frac
        }
