# kg_rl/value_net.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import torch
import torch.nn as nn

# kg_rl/value_net.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """带残差连接的 MLP 块"""

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # GELU 通常比 ReLU 表现更好
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual Connection


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()

        # 1. 输入映射层 (Input Projection)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. 深层残差网络 (Deep Residual Body)
        # num_layers 控制深度，建议 2-4 层
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])

        # 3. 输出头 (Output Head)
        self.output_head = nn.Linear(hidden_dim, 1)

        # 初始化优化 (Optional but recommended)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 降低增益，特别是对于价值网络
            if m == self.output_head:
                nn.init.orthogonal_(m.weight, gain=0.01)  # 输出层小增益
            else:
                nn.init.orthogonal_(m.weight, gain=0.5)  # 隐藏层也用较小增益

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x).view(-1)
