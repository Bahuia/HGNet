# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/30
# @Author  : Yongrui Chen
# @File    : gnn.py
# @Software: PyCharm
"""

import sys
import torch
import math
from torch import nn
from torch.nn import functional as F
from models.attention import MultiHeadAttention

sys.path.append("..")

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Block(nn.Module):
    
    def __init__(self, d_h, dropout_p):
        super().__init__()
        self.attn = MultiHeadAttention(d_h, d_h, d_h, h=4, dropout_p=dropout_p)
        self.l1 = nn.Linear(d_h, d_h*4)
        self.l2 = nn.Linear(d_h *4, d_h)
        self.ln_1 = nn.LayerNorm(d_h)
        self.ln_2 = nn.LayerNorm(d_h)
        self.drop = nn.Dropout(dropout_p)
        self.act = gelu

    def forward(self, q, k, m):
        q = self.attn(q, k, mask=m).squeeze(1)
        t = self.ln_1(q)
        q = self.drop(self.l2(self.act(self.l1(t))))
        q = self.ln_1(q + t)
        return q

class GraphTransformer(nn.Module):

    def __init__(self, n_blocks, hidden_size, dropout=0.1):
        super().__init__()

        self.N = n_blocks
        self.d_h = hidden_size
        self.n_head = 4
        self.g_in = nn.Parameter(torch.Tensor(1, hidden_size))
        # self.gat = nn.ModuleList([Block(hidden_size, dropout) for _ in range(n_blocks)])
        encoder_tf_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=self.n_head)
        self.gat = nn.TransformerEncoder(encoder_tf_layer, num_layers=n_blocks)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, t, v_in, e_in, adj):

        bs = v_in.size(0)
        v_len = v_in.size(1)
        g_in = self.g_in.unsqueeze(0).expand(bs, 1, self.d_h)
        vgraph = torch.cat([v_in, g_in, e_in], 1).transpose(0, 1)

        adj_mask = []
        for i in range(0, len(adj)):
            adj_mask.append((adj[i]==0).unsqueeze(0).repeat(self.n_head, 1, 1))
        adj_mask = torch.cat(adj_mask, dim=0)

        g_enc = self.gat(vgraph, mask=adj_mask).transpose(0, 1)

        v_out = g_enc[:,:v_len, :]
        e_out = g_enc[:, v_len + 1:, :]
        g_out = g_enc[:, v_len]
        return v_out, e_out, g_out