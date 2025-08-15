# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import math
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):
    def __init__(
            self,
            activation: str,
            cross_attention: nn.Module,
            d_model: int,
            dim_feedforward: int,
            dropout: float,
            pos_enc_at_attn: bool,
            pos_enc_at_cross_attn_keys: bool,
            pos_enc_at_cross_attn_queries: bool,
            self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        query = key = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2, _ = self.self_attn(
            query=query,  # 使用正确的参数名
            key=key,  # 使用正确的参数名
            value=tgt2  # 使用正确的参数名
        )
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            **kwds
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
            self,
            tgt,
            memory,
            pos: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
import torch
from torch import nn


# 2. 创建位置编码
def create_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2])
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

    if d_model % 2 != 0:
        pe[:, -1] = torch.sin(position * div_term[d_model // 2])

    return pe

#
# # 1. 准备输入数据
# B, N1, N2, C = 8, 128, 224, 224  # C=224是特征维度
# x = torch.randn(B, N1, N2, C)  # 形状: (8, 3, 224, 224)
# y = torch.randn(B, N2, N2, C)  # 形状: (8, 224, 224, 224)
#
# # 2. 重塑输入为 (sequence_length, batch_size, feature_dim)
# # 对于x: 将N1*N2作为序列长度 (3*224=672)
# x_reshaped = x.reshape(B, N1 * N2, C).transpose(0, 1)  # (672, 8, 224)
# # 对于y: 将N2*N2作为序列长度 (224*224=50176)
# y_reshaped = y.reshape(B, N2 * N2, C).transpose(0, 1)  # (50176, 8, 224)
#
# # 3. 创建位置编码
# x_pos = create_positional_encoding(N1 * N2, C).unsqueeze(1).repeat(1, B, 1)  # (672, 8, 224)
# y_pos = create_positional_encoding(N2 * N2, C).unsqueeze(1).repeat(1, B, 1)  # (50176, 8, 224)
#
# # 4. 初始化注意力模块
# self_attn = nn.MultiheadAttention(embed_dim=C, num_heads=8)  # 增加头数处理高维特征
# cross_attn = nn.MultiheadAttention(embed_dim=C, num_heads=8)
#
# attention_layer = MemoryAttentionLayer(
#     activation="relu",
#     cross_attention=cross_attn,
#     d_model=C,
#     dim_feedforward=512,  # 增大前馈网络维度
#     dropout=0.1,
#     pos_enc_at_attn=True,
#     pos_enc_at_cross_attn_keys=True,
#     pos_enc_at_cross_attn_queries=True,
#     self_attention=self_attn
# )
#
# memory_attention = MemoryAttention(
#     d_model=C,
#     pos_enc_at_input=True,
#     layer=attention_layer,
#     num_layers=2,
#     batch_first=False
# )
#
# # 5. 应用注意力 (注意内存使用)
# with torch.no_grad():  # 初始测试时关闭梯度计算节省内存
#     output = memory_attention(
#         curr=x_reshaped,
#         memory=y_reshaped,
#         curr_pos=x_pos,
#         memory_pos=y_pos,
#         num_obj_ptr_tokens=0
#     )
#
# # 6. 恢复形状
# output = output.transpose(0, 1).reshape(B, N1, N2, C)  # (8, 3, 224, 224)

