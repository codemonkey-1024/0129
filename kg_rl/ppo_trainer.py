# kg_rl/ppo_trainer.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple, NamedTuple

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import AdamW

from config import TrainingConfig
from core.schema import GraphPath
from .env import GraphEnv, rollout
from .path_scorer import DynamicPathScorer, ScorerAdapter
from .policy import Policy
from .reward_funcs import BatchedRewardFn
from .value_net import ValueNet
from .schema import *


# =========================================================================
# 1. 配置类：增加 Value Clip 参数
# =========================================================================
@dataclass
class PPOConfig:
    """PPO 训练超参数配置"""
    # 优化相关
    lr: float = 1e-4
    weight_decay: float = 1e-3  # 稍微增加一点正则化
    max_grad_norm: float = 0.5  # PPO通常梯度裁剪更激进

    # PPO 核心
    clip_ratio: float = 0.2
    clip_range_vf: Optional[float] = 0.2  # [新增] Value Function 的截断范围，通常设为 clip_ratio
    ppo_epochs: int = 2  # 增加 Epoch 数以充分利用样本
    batch_size: int = 128
    entropy_coef: float = 0.01

    # GAE & 价值函数
    gamma: float = 0.99
    gae_lambda: float = 0.99
    value_coef: float = 0.5
    value_hidden_dim: int = 512

    # 奖励重塑 (Reward Shaping)
    lambda_shaping: float = 1.0
    reward_scale: float = 1.0  # 建议由归一化处理，这里设为 1.0
    alpha_query: float = 0.8  # 最终主要看 Global Reward
    query_baseline_momentum: float = 0.3

    # 采样
    rollout_workers: int = 8
    depth_norm_scale: float = 5.0  # 深度归一化分母


# =========================================================================
# 2. 数据结构
# =========================================================================
class MiniBatch(NamedTuple):
    """一个训练 Mini-Batch 所需的数据"""
    states: List[StepState]
    actions: torch.Tensor  # [B]
    old_logps: torch.Tensor  # [B]
    old_values: torch.Tensor  # [B] (新增：用于 Value Clip)
    returns: torch.Tensor  # [B] (Target Value)
    advantages: torch.Tensor  # [B]
    candidates: List[List[int]]  # 候选动作


# =========================================================================
# 3. Trainer 主类
# =========================================================================
class PPOTrainer:
    def __init__(
            self,
            cfg: TrainingConfig,
            policy: Policy,
            reward_fn: BatchedRewardFn,
            config: PPOConfig = PPOConfig(),
            prefix_score_fn: Optional[callable] = None,
    ):
        self.config = config
        self.device = next(policy.parameters()).device

        self.policy = policy
        self.reward_fn = reward_fn
        self.prefix_score_fn = prefix_score_fn

        print("[Info] Initializing Independent Critic (Twin Scorer)...")

        # 1. 实例化基础 Scorer
        critic_base = DynamicPathScorer(model_name=cfg.pretrained_model_name, use_lora=cfg.use_lora)

        # 2. [关键] 加载预训练权重
        # Critic 必须和 Policy 站在同一起跑线上，否则它在初期就是个“瞎子”
        if cfg.train_checkpoint:
            print(f"[Info] Loading Critic checkpoint from {cfg.train_checkpoint}")
            critic_base.load_state_dict(torch.load(cfg.train_checkpoint, map_location='cpu'))

        critic_base.to(self.device)

        # 3. 包装成 Adapter (方便处理 GraphPath)
        self.value_net = ScorerAdapter(critic_base).to(self.device)

        # 优化器
        policy_params = list(policy.parameters())
        critic_params = list(self.value_net.parameters())

        self.optimizer = AdamW([
            {'params': policy_params, 'lr': config.lr},
            {'params': critic_params, 'lr': config.lr}  # 这里你可以根据需要微调，比如 config.lr * 0.5
        ], weight_decay=config.weight_decay)

        self.query_baseline: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # 核心接口
    # -------------------------------------------------------------------------
    def train_step(
            self,
            env: GraphEnv,
            start_nodes: List[Dict],
            queries: List[str],
            warm_starts: Optional[List[GraphPath]] = None,
            samples_per_start: int = 1,
    ) -> Dict[str, Any]:
        """执行一次 PPO 迭代"""

        # 1. 采样 (Rollout) - 无梯度
        self.policy.eval()
        with torch.no_grad():
            trajs = self._collect_trajectories(env, start_nodes, queries, warm_starts, samples_per_start)

        if not trajs:
            return {"loss": 0.0, "reward/mean": 0.0}

        # 2. 计算奖励 (Reward Engineering)
        metrics = self._compute_rewards_and_metrics(trajs)

        # 3. 计算优势估计 (GAE) & 预计算 Value
        # 这里需要计算一次 Value 用于 GAE，注意：此时不需要梯度
        samples = self._compute_gae(trajs)

        # 4. PPO 更新循环 (Optimization) - 有梯度
        train_stats = self._update_parameters(env, samples)

        return {**metrics, **train_stats}

    # -------------------------------------------------------------------------
    # 内部组件 1: 数据收集与奖励
    # -------------------------------------------------------------------------
    def _collect_trajectories(self, env, start_nodes, queries, warm_starts, samples_per_start):
        return rollout(
            env=env,
            policy=self.policy,
            start_nodes=start_nodes,
            queries=queries,
            warm_starts=warm_starts,
            greedy=False,
            use_grad=False,
            samples_per_start=samples_per_start,
            num_workers=self.config.rollout_workers,
        )

    def _compute_rewards_and_metrics(self, trajs: List[Trajectory]) -> Dict[str, float]:
        """计算奖励并生成详细监控指标"""
        # A. 稀疏奖励计算 (按 Query 分组)
        groups = defaultdict(lambda: {"start_nodes": [], "items": []})
        for tr in trajs:
            final_gp = tr.final_path if tr.final_path else GraphPath(tr.start_node)
            groups[tr.query]["start_nodes"].append(tr.start_node)
            groups[tr.query]["items"].append({"traj": tr, "graph_path": final_gp})

        for q, pack in groups.items():
            _, reward_details = self.reward_fn(q, pack["start_nodes"], pack["items"])
            for item in pack["items"]:
                tr = item["traj"]
                r_q = float(item.get("R_q", 0.0))
                # 兼容旧逻辑，如果有 path_reward 则混合
                r_path = float(item.get("path_reward", r_q))
                raw_reward = self.config.alpha_query * r_q + (1.0 - self.config.alpha_query) * r_path
                sharpened_reward = raw_reward
                tr.reward = sharpened_reward

                tr.reward_details = reward_details

        # B. 稠密奖励 (Shaping)
        self._compute_prefix_deltas(trajs)

        # C. 统计指标 (用于 TensorBoard)
        rewards = [t.reward for t in trajs]
        metrics = {
            "reward/total_mean": np.mean(rewards) if rewards else 0.0,
            "traj/length": np.mean([len(t.actions) for t in trajs]) if trajs else 0.0
        }

        # 聚合详细奖励分量
        details_agg = defaultdict(list)
        for tr in trajs:
            if tr.reward_details:
                for k, v in tr.reward_details.items():
                    details_agg[k].append(v)
            if tr.prefix_deltas:
                details_agg["shaping_sum"].append(sum(tr.prefix_deltas))

        for k, v_list in details_agg.items():
            metrics[f"reward_comp/{k}"] = np.mean(v_list)

        return metrics

    def _compute_prefix_deltas(self, trajs: List[Trajectory]):
        if self.prefix_score_fn is None:
            for tr in trajs: tr.prefix_deltas = [0.0] * len(tr.actions)
            return

        for tr in trajs:
            scores = []
            for t, st in enumerate(tr.states):
                prefix_probs = tr.chosen_step_probs[:t]
                s, _ = self.prefix_score_fn(tr.query, st.path, prefix_probs)
                scores.append(s)

            deltas = []
            for t in range(len(tr.actions)):
                # Reward = Potential(St+1) - Potential(St)
                next_s = scores[t + 1] if t + 1 < len(scores) else 0.0  # 终止状态势能通常视为0或保持
                # 注意：如果是最后一步，Potential 是否归零取决于具体设计，
                # 这里假设最后一步没有额外的势能奖励，或者势能融合进了 Final Reward
                if t + 1 < len(scores):
                    deltas.append(scores[t + 1] - scores[t])
                else:
                    deltas.append(0.0)
            tr.prefix_deltas = deltas

    # -------------------------------------------------------------------------
    # 内部组件 2: GAE 计算
    # -------------------------------------------------------------------------
    def _compute_gae(self, trajs: List[Trajectory]) -> List[Dict]:
        """Generalized Advantage Estimation"""
        flat_samples = []

        for tr in trajs:
            T = len(tr.actions)
            if T == 0 or tr.reward is None: continue

            # 1. 奖励重塑: r_t = shaping_weight * delta_t
            # 最后一步加上中心化的 Final Reward
            baseline = self._update_query_baseline(tr.query, float(tr.reward))
            centered_final_reward = float(tr.reward) # - baseline

            step_rewards = [self.config.lambda_shaping * d for d in tr.prefix_deltas]
            # 确保长度一致
            if len(step_rewards) < T: step_rewards.extend([0.0] * (T - len(step_rewards)))
            step_rewards[T - 1] += centered_final_reward

            # 2. 预估 Value (用于 GAE 计算，无梯度)
            # 注意: 这里 detach_encoder=False 但 with_grad=False，纯推理
            values = self._predict_values(tr.states, with_grad=False).cpu().numpy().flatten()

            # 3. 反向递归计算 GAE
            advs = np.zeros(T, dtype=np.float32)
            returns = np.zeros(T, dtype=np.float32)  # Return = Advantage + Value
            last_gae = 0.0
            next_val = 0.0

            for t in reversed(range(T)):
                # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
                delta = step_rewards[t] + self.config.gamma * next_val - values[t]

                # A_t = delta + (gamma * lambda) * A_{t+1}
                last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae

                advs[t] = last_gae
                returns[t] = last_gae + values[t]

                next_val = values[t]

            # 4. 收集样本
            for t in range(T):
                flat_samples.append({
                    "state": tr.states[t],
                    "action": int(tr.actions[t]),
                    "logp_old": float(tr.log_probs[t].item()),
                    "value_old": float(values[t]),  # 记录旧 Value 用于 Clip
                    "adv": float(advs[t]),
                    "ret": float(returns[t]),
                    "query": tr.query
                })

        return flat_samples

    # -------------------------------------------------------------------------
    # 内部组件 3: 参数更新 (Optimization)
    # -------------------------------------------------------------------------
    def _update_parameters(self, env: GraphEnv, samples: List[Dict]) -> Dict[str, float]:
        self.policy.train()

        # 转换为 Tensor
        advs = torch.tensor([s["adv"] for s in samples], device=self.device)
        returns = torch.tensor([s["ret"] for s in samples], device=self.device)
        old_logps = torch.tensor([s["logp_old"] for s in samples], device=self.device)
        old_values = torch.tensor([s["value_old"] for s in samples], device=self.device)

        # 优势归一化 (Global Normalization)
        std_adv = advs.std()
        if std_adv < 1e-5:
            advs = torch.zeros_like(advs)
        else:
            advs = (advs - advs.mean()) / (std_adv + 1e-8)

        # Dataset 设置
        dataset_size = len(samples)
        indices = np.arange(dataset_size)

        # 统计
        stats = defaultdict(list)

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.config.batch_size):
                end = min(start + self.config.batch_size, dataset_size)
                batch_idx = indices[start:end]

                batch = MiniBatch(
                    states=[samples[i]["state"] for i in batch_idx],
                    actions=torch.tensor([samples[i]["action"] for i in batch_idx], device=self.device),
                    old_logps=old_logps[batch_idx],
                    old_values=old_values[batch_idx],  # 传入旧 Value
                    returns=returns[batch_idx],
                    advantages=advs[batch_idx],
                    candidates=[env.candidates(samples[i]["state"]) for i in batch_idx]
                )

                # 计算 Loss
                loss_dict = self._compute_loss(batch)

                # 反向传播
                self.optimizer.zero_grad()
                loss_dict["loss"].backward()

                # 梯度裁剪
                grad_norm_check = torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.config.max_grad_norm
                )

                if torch.isfinite(grad_norm_check):
                    self.optimizer.step()
                else:
                    print(f"[Warn] Skipped update step due to Inf/NaN gradients: {grad_norm_check}")

                # 记录详细日志
                stats["loss/total"].append(loss_dict["loss"].item())
                stats["loss/policy"].append(loss_dict["policy_loss"].item())
                stats["loss/value"].append(loss_dict["value_loss"].item())
                stats["entropy"].append(loss_dict["entropy"].item())
                stats["ppo/approx_kl"].append(loss_dict["approx_kl"].item())
                stats["ppo/clip_frac"].append(loss_dict["clip_frac"].item())
                stats["train/grad_norm"].append(grad_norm_check.item() if grad_norm_check.item() < 10 else 0.0)

                # Explained Variance
                y_true = batch.returns
                y_pred = loss_dict["values_pred"]
                var_y = torch.var(y_true)
                exp_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
                stats["val/explained_var"].append(exp_var.item())

        return {k: np.mean(v) for k, v in stats.items()}

    def _evaluate_actions(self, batch: MiniBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        辅助函数：重新计算当前 Policy 下的 Values, LogProbs, Entropy
        """
        paths = [s.path for s in batch.states]
        queries = [s.query for s in batch.states]

        # 1. 重新计算 Value (with_grad=True, 这样 Value Loss 才能更新网络)
        # 这里的 detach_encoder 应该根据策略决定，通常 PPO 中 Value 和 Policy 共享 Encoder 时不 detach
        values_pred = self._predict_values(batch.states, with_grad=True, detach_encoder=False)
        values_pred = values_pred.view(-1)

        # 2. 重新计算 Policy Logits
        logits_list = self.policy(paths, queries, batch.candidates)

        # 3. 计算 LogProb 和 Entropy
        log_probs = []
        entropies = []

        for logits, act in zip(logits_list, batch.actions):
            dist = Categorical(logits=logits)
            log_probs.append(dist.log_prob(act))
            entropies.append(dist.entropy())

        return values_pred, torch.stack(log_probs), torch.stack(entropies).mean()

    def _compute_loss(self, batch: MiniBatch) -> Dict[str, torch.Tensor]:
        """计算 PPO 综合 Loss"""

        # 1. 获取当前网络的预测值
        values_pred, log_probs_new, entropy = self._evaluate_actions(batch)

        # 2. Policy Loss (Clipped Surrogate)
        ratio = torch.exp(log_probs_new - batch.old_logps)
        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * batch.advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 3. Value Loss (Clipped) - [重要优化]
        # 使用 Clipped Value Loss 可以防止 Critic 更新过猛
        v_pred = values_pred
        v_target = batch.returns

        # 标准 MSE
        loss_v_unclipped = (v_pred - v_target) ** 2

        if self.config.clip_range_vf is not None:
            # 限制 Value 预测值的变化范围
            v_pred_clipped = batch.old_values + torch.clamp(
                v_pred - batch.old_values,
                -self.config.clip_range_vf,
                self.config.clip_range_vf
            )
            loss_v_clipped = (v_pred_clipped - v_target) ** 2
            # 取两者中较大的 Loss (Pessimistic bound)
            value_loss = 0.5 * torch.max(loss_v_unclipped, loss_v_clipped).mean()
        else:
            value_loss = 0.5 * loss_v_unclipped.mean()

        # 4. Total Loss
        total_loss = policy_loss + \
                     self.config.value_coef * value_loss - \
                     self.config.entropy_coef * entropy

        # 5. 监控指标
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            clip_frac = (torch.abs(ratio - 1.0) > self.config.clip_ratio).float().mean()

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "values_pred": values_pred
        }

    # -------------------------------------------------------------------------
    # 辅助工具: Value Prediction
    # -------------------------------------------------------------------------
    def _predict_values(
            self,
            states: List[StepState],
            with_grad: bool,
            detach_encoder: bool = False
    ) -> torch.Tensor:
        """
        使用 BERT Critic 计算状态价值 V(s)。
        """
        if not states:
            return torch.tensor([], device=self.device)

        # 1. 提取路径和 Query
        paths = [s.path for s in states]
        queries = [s.query for s in states]

        # 2. 准备上下文 (Grad vs No Grad)
        ctx = torch.enable_grad if with_grad else torch.no_grad

        with ctx():
            # score_paths 会返回 [B, 1] 的 Tensor
            # ScorerAdapter 内部会自动处理 Tokenize, Forward 等所有脏活
            values = self.value_net.score_paths(paths, queries)

            # 展平为 [B]
            return values.view(-1)
    def _update_query_baseline(self, query: str, reward: float) -> float:
        """Query-level Baseline (Moving Average)"""
        old_b = self.query_baseline.get(query, reward)
        new_b = (1.0 - self.config.query_baseline_momentum) * old_b + \
                self.config.query_baseline_momentum * reward
        self.query_baseline[query] = new_b
        return new_b