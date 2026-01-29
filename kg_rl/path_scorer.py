import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Sequence, List, Tuple, Optional
from loguru import logger
from core.schema import GraphPath

# === PEFT 检查保持不变 ===
try:
    from peft import LoraConfig, get_peft_model, TaskType

    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False


# ==========================================
# 1. 基础组件 (Text Encoder)
# ==========================================

class TextEncoderBase(nn.Module):
    """
    负责将所有文本映射为向量的基础编码器。
    集成自动适配 LoRA target_modules 的功能。
    """

    def __init__(self, model_name: str, use_lora: bool, lora_config: dict = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        base_model = AutoModel.from_pretrained(model_name)

        if use_lora and _HAS_PEFT:
            lora_config = lora_config or {}

            # --- [修复核心] 自动探测目标模块 ---
            # 优先使用配置中的设置，如果没有，则自动推断
            target_modules = lora_config.get("target_modules", None)
            if target_modules is None:
                target_modules = self._get_auto_target_modules(base_model)

            logger.info(f"LoRA detected model type: {base_model.config.model_type}")
            logger.info(f"LoRA applying to modules: {target_modules}")

            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("alpha", 16),
                lora_dropout=lora_config.get("dropout", 0.1),
                target_modules=target_modules  # 传入自动适配后的模块列表
            )
            self.bert = get_peft_model(base_model, peft_config)
        else:
            self.bert = base_model
            self._freeze_layers()

        self.hidden_size = self.bert.config.hidden_size
        self.null_embedding = nn.Parameter(torch.randn(1, self.hidden_size) * 1e-3)



        # 能够有效降低显存占用
        self.bert.config.use_cache = False  # 推荐关掉 cache，避免冲突
        self.bert.gradient_checkpointing_enable()
        self.bert.enable_input_require_grads()

    def _get_auto_target_modules(self, model: nn.Module) -> List[str]:
        """
        根据模型架构自动返回 LoRA 需要注入的层名称。
        保留了原本的适配逻辑。
        """
        # 获取模型类型字符串 (如 'distilbert', 'bert', 'roberta')
        model_type = getattr(getattr(model, "config", None), "model_type", "").lower()

        if model_type == 'distilbert':
            # DistilBERT 使用 q_lin, k_lin, v_lin
            return ["q_lin", "k_lin", "v_lin"]

        if model_type in {"bert", "roberta", "albert", "electra", "deberta", "deberta-v2"}:
            # BERT 家族通常使用 query, key, value
            return ["query", "key", "value"]

        if model_type in {"llama", "mistral", "qwen", "qwen2"}:
            # Llama 家族通常使用 q_proj, k_proj, v_proj
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

        if model_type == 't5':
            return ["q", "k", "v"]

        # 默认回退
        logger.warning(f"Unknown model type '{model_type}', falling back to ['query', 'key', 'value']")
        return ["query", "key", "value"]

    def _freeze_layers(self):
        # 简单的冻结逻辑
        for param in self.bert.parameters():
            param.requires_grad = False
        # 如果需要解冻最后几层，可以在这里添加逻辑

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, texts: Sequence[Optional[str]]) -> torch.Tensor:
        """
        输入: 文本列表，允许包含 None 或 ""
        输出: [Batch, Hidden]
        """
        if not texts:
            return torch.empty(0, self.hidden_size, device=self.device)

        batch_size = len(texts)
        device = self.device

        # 1. 预处理：标记哪些是有效的非空文本
        # text.strip() check ensures strings with only spaces are also treated as empty
        valid_mask = []
        valid_texts_original = []

        for t in texts:
            is_valid = t is not None and isinstance(t, str) and len(t.strip()) > 0
            valid_mask.append(is_valid)
            if is_valid:
                valid_texts_original.append(t)

        # 转换为 Tensor Mask
        valid_mask_tensor = torch.tensor(valid_mask, device=device, dtype=torch.bool)

        # 2. 准备输出容器，默认用 Null Embedding 填充
        # expand 扩展维度 [1, H] -> [B, H]
        output_emb = self.null_embedding.expand(batch_size, -1).clone()

        # 如果全是空的，直接返回 null embedding 即可
        if not valid_texts_original:
            return output_emb

        # 3. 对非空文本进行去重 (优化 BERT 计算量)
        unique_texts = list(set(valid_texts_original))
        text2idx = {t: i for i, t in enumerate(unique_texts)}

        # 4. 运行 BERT (只针对 Unique Non-Empty Texts)
        inputs = self.tokenizer(
            unique_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        bert_outputs = self.bert(**inputs)

        # 通常取 [CLS] (index 0)
        unique_embs = bert_outputs.last_hidden_state[:, 0, :]  # [U, H]

        # 5. 映射回原始顺序 (Map Back)
        # 获取所有有效文本在 unique_embs 中的索引
        indices_in_unique = [text2idx[t] for t in valid_texts_original]
        indices_tensor = torch.tensor(indices_in_unique, device=device)

        # Gather 对应的向量 [V, H] (V = number of valid texts)
        mapped_valid_embs = unique_embs.index_select(0, indices_tensor)

        # 6. 填入最终 Tensor
        # 利用 mask 将计算好的向量填入对应位置
        # 注意：这里会产生梯度连接，反向传播可以正常流向 BERT
        # 未被填充的位置保留了 null_embedding，其梯度流向 self.null_embedding
        output_emb[valid_mask_tensor] = mapped_valid_embs

        return output_emb

# ==========================================
# 2. 核心模块：Quadruple 融合与门控
# ==========================================

class TypeAwareFusionLayer(nn.Module):
    """
    升级版融合层：
    能够区分显式关系（KB Edge）和隐式关系（Text Co-occurrence）。
    针对 Co-occurrence，它会尝试从 Context 中挖掘出 'Latent Relation'。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. 显式结构特征处理 (h, r, t)
        self.struct_proj = nn.Linear(hidden_size * 3, hidden_size)

        # 2. 隐式关系挖掘器 (Latent Relation Miner)
        # 输入: h, t, context -> 输出: inferred_relation
        # 逻辑: 给定头尾实体，在上下文中寻找连接它们的语义
        self.latent_rel_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # [h; t; c]
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)  # 生成一个“推断出的关系向量”
        )

        # 3. 自适应门控 (Adaptive Gate)
        # 决定当前这一步是更像 "KB推理" 还是 "阅读理解"
        # 0 -> 全靠 KB 结构, 1 -> 全靠文本挖掘
        self.mode_gate = nn.Sequential(
            nn.Linear(hidden_size * 4, 1),  # 看 h, r, t, c 全貌
            nn.Sigmoid()
        )

        # 4. Query-Guided Attention (保留之前的优点)
        # 即使是共现，也要看跟 Query 有没有关系
        self.ctx_query_gate = nn.Linear(hidden_size * 2, 1)

        self.out_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, h_emb, r_emb, t_emb, c_emb, q_emb_expanded):
        """
        r_emb: 对于共现边，这里输入的是 "co-occurrence" 的静态向量
        c_emb: 文档内容的向量表示
        """

        # --- A. 显式路径特征 (Structural Path) ---
        # 传统的 KB 推理：h + r + t
        struct_feat = self.struct_proj(torch.cat([h_emb, r_emb, t_emb], dim=-1))

        # --- B. 隐式关系挖掘 (Latent Relation Extraction) ---
        # 假设：真正的关系隐藏在 h, t 和 context 的交互中
        # 例如：h=Google, t=YouTube, c="Google acquired YouTube in 2006..."
        # latent_rel_feat 应该捕捉到 "acquisition" 的语义
        latent_input = torch.cat([h_emb, t_emb, c_emb], dim=-1)
        latent_rel_feat = self.latent_rel_proj(latent_input)

        # --- C. 自适应融合 (Adaptive Fusion) ---
        # 计算门控值 alpha
        # 拼接所有信息让模型自己判断：如果 r_emb 是 "co-cur" (泛化向量)，模型会倾向于提高 alpha
        gate_input = torch.cat([h_emb, r_emb, t_emb, c_emb], dim=-1)
        alpha = self.mode_gate(gate_input)  # shape: [N, 1]

        # 动态混合：
        # 如果是明确关系 (born_in)，alpha 趋近 0，使用 struct_feat
        # 如果是共现关系 (co-cur)，alpha 趋近 1，使用 latent_rel_feat 替代原有的结构特征
        fused_step = (1 - alpha) * struct_feat + alpha * latent_rel_feat

        # --- D. Query 过滤 (Query Relevance) ---
        # 这一步依然非常关键：就算挖掘出了关系，如果跟 Query 无关也要丢弃
        # 例如 Query 问 "时间"，Context 说的是 "地点"，则过滤掉
        q_ctx_input = torch.cat([fused_step, q_emb_expanded], dim=-1)
        relevance = torch.sigmoid(self.ctx_query_gate(q_ctx_input))

        final_step = self.out_layer_norm(fused_step * relevance)

        return final_step

# ==========================================
# 3. 序列建模：Transformer Path Encoder
# ==========================================

class PathTransformer(nn.Module):
    """
    替代 GRU，使用 Transformer Encoder 建模路径中的长距离依赖。
    """

    def __init__(self, hidden_size: int, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 可学习的位置编码 (假设路径最大长度 20)
        self.pos_emb = nn.Parameter(torch.randn(1, 20, hidden_size))

    def forward(self, step_embs, mask=None):
        """
        step_embs: [B, L, H]
        mask: [B, L] (True for padding, False for valid in PyTorch Transformer convention usually,
                      but let's use src_key_padding_mask logic: True is masked/padded)
        """
        B, L, H = step_embs.shape

        # 添加位置编码
        pos = self.pos_emb[:, :L, :]
        x = step_embs + pos

        # Transformer Forward
        # mask logic: True values are ignored
        out = self.transformer(x, src_key_padding_mask=~mask)  # 注意传入 mask 的逻辑反转
        return out


# ==========================================
# 4. 主模型：Dynamic Path Scorer
# ==========================================

class DynamicPathScorer(nn.Module):
    def __init__(
            self,
            model_name: str = "distilbert-base-uncased",
            use_lora: bool = True
    ):
        super().__init__()

        # 1. 共享文本编码器
        self.text_encoder = TextEncoderBase(model_name, use_lora)
        self.hidden_size = self.text_encoder.hidden_size

        # 2. 四元组融合层
        self.quad_fusion = TypeAwareFusionLayer(self.hidden_size)

        # 3. 路径序列建模
        self.path_encoder = PathTransformer(self.hidden_size)

        # 4. 最终评分 Cross-Attention
        # Query 关注 路径的所有步
        self.attn_score = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            batch_first=True
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )



    @property
    def device(self):
        return self.text_encoder.device

    def load_pretrained_mrp(self, path="checkpoints/mrp_pretrained"):
        # 加载 TextEncoder (包含 LoRA 或 BERT 权重)
        self.text_encoder.load_state_dict(torch.load(f"{path}/text_encoder.pt"))

        # 加载 Fusion Layer (主要是 latent_rel_proj 的参数)
        # 注意：mode_gate 和 ctx_query_gate 在预训练中没用到，它们会保持随机初始化
        # 这是正常的，因为这两个 gate 需要在 Path Ranking 任务中学习
        self.quad_fusion.load_state_dict(torch.load(f"{path}/fusion_layer.pt"), strict=False)
        print("Loaded MRP pretrained weights!")

    def _flatten_paths(self, paths_batch: List[List[Tuple[str, str, str, str]]]):
        """
        Flatten paths for efficient batch encoding.
        Returns flattened lists and reconstruction indices.
        """
        flat_h, flat_r, flat_t, flat_c = [], [], [], []
        path_lens = []

        for path in paths_batch:
            # [新增] 防御性编程：如果 Adapter 漏掉了空路径，这里强制补一个 Dummy
            # 防止 Transformer 接收到长度为 0 的序列导致 NaN 或报错
            if not path:
                path = [("Unknown", "[NULL]", "Unknown", "")]

            path_lens.append(len(path))
            for h, r, t, c in path:
                flat_h.append(h if h else "") # 确保不为 None
                flat_r.append(r if r else "")
                flat_t.append(t if t else "")
                flat_c.append(c if c else "")

        return (flat_h, flat_r, flat_t, flat_c), path_lens

    def encode_path(self, paths: List[List[Tuple[str, str, str, str]]], queries: List[str]):
        """
        paths: List of Paths. Each Path is List of (h, r, t, context_str)
        queries: List of query strings (one per path)
        """
        batch_size = len(paths)

        # --- A. Encode Query ---
        q_emb = self.text_encoder(queries)  # [B, H]

        # --- B. Encode Path Components (Flattened) ---
        (flat_h, flat_r, flat_t, flat_c), path_lens = self._flatten_paths(paths)
        max_len = max(path_lens)

        # 批量编码所有组件
        # 注意：这里会发生大量的文本编码，实际生产中应当有 Cache 机制
        all_texts = flat_h + flat_r + flat_t + flat_c
        # 为避免重复编码，text_encoder 内部已做去重
        # 这里为了演示逻辑清晰，我们假设内部处理好了

        h_emb = self.text_encoder(flat_h)
        r_emb = self.text_encoder(flat_r)
        t_emb = self.text_encoder(flat_t)
        c_emb = self.text_encoder(flat_c)

        # --- C. Query-Guided Quadruple Fusion ---
        # 我们需要把 Query 扩展到和 Flattened Steps 一样的维度
        # queries [B, H] -> need [Total_Steps, H]
        q_emb_expanded_list = []
        for i, q_vec in enumerate(q_emb):
            # 重复 path_lens[i] 次
            q_emb_expanded_list.append(q_vec.unsqueeze(0).repeat(path_lens[i], 1))
        q_emb_expanded = torch.cat(q_emb_expanded_list, dim=0)

        # 融合：得到每一跳的向量 [Total_Steps, H]
        step_vecs = self.quad_fusion(h_emb, r_emb, t_emb, c_emb, q_emb_expanded)

        # --- D. Reconstruct Batch Sequence for Transformer ---
        # [Total_Steps, H] -> [B, Max_Len, H]
        padded_steps = torch.zeros(batch_size, max_len, self.hidden_size, device=self.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)  # 1 for valid

        cursor = 0
        for i, length in enumerate(path_lens):
            end = cursor + length
            padded_steps[i, :length, :] = step_vecs[cursor:end]
            mask[i, :length] = True
            cursor = end

        # --- E. Path Modeling (Transformer) ---
        # [B, L, H]
        path_context = self.path_encoder(padded_steps, mask=mask)

        # --- F. Reasoning & Scoring ---
        # Query: [B, 1, H]
        # Key/Value: Path Context [B, L, H]
        # Attention: Query looks at the path to find evidence

        q_in = q_emb.unsqueeze(1)
        # key_padding_mask: True for padding (ignored)
        attn_out, _ = self.attn_score(q_in, path_context, path_context, key_padding_mask=~mask)
        attn_out = attn_out.squeeze(1)  # [B, H]

        # Concatenate Query + Attended Path Info
        final_feat = torch.cat([q_emb, attn_out], dim=-1)  # [B, 2H]

        return final_feat


    def forward(self, paths: List[List[Tuple[str, str, str, str]]], queries: List[str]):
        final_feat = self.encode_path(paths, queries)
        score = self.final_mlp(final_feat)  # [B, 1]
        return score

    def score(self, paths: List[List[Tuple[str, str, str, str]]], queries: List[str]):
        final_feat = self.encode_path(paths, queries)
        score = self.final_mlp(final_feat)  # [B, 1]
        return final_feat, score



class ScorerAdapter(nn.Module):
    """
    适配器：负责把 GraphPath 转成 PathScorer 所需的 (h, r, t, doc_id) 四元组列表。
    """

    def __init__(self, base_scorer: DynamicPathScorer, max_batch_size: int = 512):
        super().__init__()
        self.base = base_scorer
        self.max_batch_size = max_batch_size

    @staticmethod
    def _graphpath_to_quads(gp: GraphPath) -> List[Tuple[str, str, str, str]]:
        # [修改] 处理初始状态 (只有节点，没有边的情况)
        if not gp.relations:
            # 如果路径为空（通常是 Start Node），构造一个虚拟的“初始步”
            # 这样 ValueNet 就能基于起始节点的信息预测初始价值 V(s0)
            if gp.nodes:
                node = gp.nodes[0]
                # 提取节点文本信息
                mention = node.get("mention", node.get("id", "unk"))
                desc = node.get("description", "")
                # 拼接文本
                node_text = f"{mention}, {desc}" if mention else "Unknown"

                # 返回四元组: (Head=StartNode, Rel=[START], Tail=StartNode, Context=Initial State)
                return [(node_text, "[START]", node_text, "Initial State")]
            else:
                # 极罕见情况：连 start node 都没有
                return [("Unknown", "[NULL]", "Unknown", "")]
        quads: List[Tuple[str, str, str, str]] = []
        for rel in gp.relations:
            # Head
            h_mention = rel["begin"].get("mention", rel["begin"].get("id"))
            h_desc = rel["begin"].get("description", "")
            h = f"{h_mention}, {h_desc}" if h_mention else "Unknown"  # 增加默认值保护

            # Relation
            r = rel.get("r", "rel")  # 默认值

            # Tail
            t_mention = rel["end"].get("mention", rel["end"].get("id"))
            t_desc = rel["end"].get("description", "")
            t = f"{t_mention}, {t_desc}" if t_mention else "Unknown"  # 增加默认值保护

            # Context / Doc
            if r == 'co-cur' and "doc" in rel:
                doc_title = rel["doc"].get('title', "")
                doc_ctx = rel["doc"].get('context', "")
                # 只有当确实有内容时才拼接
                if doc_title or doc_ctx:
                    doc = None #f"{doc_title}\n{doc_ctx}"
                else:
                    doc = None
            else:
                doc = None #rel.get('evidence')

            quads.append((h, r, t, doc))
        return quads

    def score_paths(
            self,
            paths: List[GraphPath],
            queries: List[str]
    ):
        """
        输入一批 GraphPath 和 query，输出每条路径的打分：
          - return_full=False: 只返回最后一跳 score [B, 1]
          - return_full=True:  返回 (full_path_score, last_triple_score)
        """
        # if len(paths) > 512:
        #     logger.warning(f"Warning: paths over large: {len(paths)}")

        assert len(paths) == len(queries), "paths 和 queries 长度必须一致"

        quads = [self._graphpath_to_quads(p) for p in paths]

        all_full_scores = []

        for start in range(0, len(quads), self.max_batch_size):
            end = start + self.max_batch_size
            quad_batch = quads[start:end]
            query_batch = queries[start:end]

            full_scores = self.base(
                quad_batch,
                query_batch,
            )
            all_full_scores.append(full_scores)
        full_scores = torch.cat(all_full_scores, dim=0)
        return full_scores


    def encode_paths(
            self,
            paths: List[GraphPath],
            queries: List[str]
    ):
        """
        输入一批 GraphPath 和 query，输出每条路径的打分：
          - return_full=False: 只返回最后一跳 score [B, 1]
          - return_full=True:  返回 (full_path_score, last_triple_score)
        """
        if len(paths) > 512:
            logger.warning(f"Warning: paths over large: {len(paths)}")

        assert len(paths) == len(queries), "paths 和 queries 长度必须一致"

        quads = [self._graphpath_to_quads(p) for p in paths]

        path_embed_list = []

        for start in range(0, len(quads), self.max_batch_size):
            end = start + self.max_batch_size
            quad_batch = quads[start:end]
            query_batch = queries[start:end]

            path_embeds = self.base.encode_path(
                quad_batch,
                query_batch,
            )
            path_embed_list.append(path_embeds)
        return torch.cat(path_embed_list, dim=0)




# ==========================================
# 5. 使用示例
# ==========================================
if __name__ == '__main__':
    # 模拟数据：路径现在是四元组 (Head, Relation, Tail, Context)
    # Context 可以是一段证明文本，如果没有则为空字符串
    paths = [
        [
            ("Obama", "born_in", "Hawaii", "Barack Obama was born in Honolulu, Hawaii."),
            ("Hawaii", "located_in", "USA", "")
        ],
        [
            ("Paris", "capital_of", "France", "Paris is the capital and most populous city of France."),
            ("France", "part_of", "Europe", "")
        ]
    ]
    queries = [
        "Where was Obama born?",
        "Is Paris in Asia?"
    ]

    print("Initializing QG-PathNet...")
    model = DynamicPathScorer(use_lora=True)

    # 模拟微调：开启梯度
    model.train()

    print("Forward Pass...")
    scores = model(paths, queries)
    print("Path Scores:", scores)
    print("Shape:", scores.shape)