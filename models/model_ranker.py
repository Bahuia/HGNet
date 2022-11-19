# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : model_ranker.py
# @Software: PyCharm
"""

import sys
import torch
import torch.nn as nn
sys.path.append("..")
from utils.embedding import Embeddings
from models.rnn import LSTM
from utils.utils import max_pooling_by_lens, max_pooling_by_mask, length_array_to_mask_tensor


class RankingModel(nn.Module):

    def __init__(self, vocab, args):

        super(RankingModel, self).__init__()
        self.args = args

        wo_vocab = vocab
        self.word_embedding = Embeddings(args.d_emb_wo, wo_vocab)

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        self.wo_lstm = LSTM(d_input=args.d_emb_wo, d_h=args.d_h_wo // 2,
                                n_layers=args.n_lstm_layers, birnn=args.birnn, dropout=args.dropout)

        self.dropout = nn.Dropout(args.dropout)

        self.cos = nn.CosineSimilarity(eps=1e-6)

    def forward(self, batch):

        q, q_lens, ment_f, \
        pos_r, pos_r_lens, \
        neg_r, neg_r_lens, neg_r_blens = batch

        bs = q.size(0)

        q_encodings, _ = self.encode(q, q_lens, self.word_embedding, self.wo_lstm)
        q_encodings = self.dropout(q_encodings)
        q_e_mask = length_array_to_mask_tensor(q_lens, value=ment_f, mask_symbol=1)
        if self.args.cuda:
            q_e_mask = q_e_mask.to(self.args.gpu)

        # q_h: Size(bs, d_h_wo)
        q_h = max_pooling_by_mask(q_encodings, q_e_mask)

        pos_r_encodings, _ = self.encode(pos_r, pos_r_lens, self.word_embedding, self.wo_lstm)
        pos_r_encodings = self.dropout(pos_r_encodings)
        # pos_r_h: Size(bs, d_h_wo)
        pos_r_h = max_pooling_by_lens(pos_r_encodings, pos_r_lens)

        neg_r_encodings, _ = self.encode(neg_r, neg_r_lens, self.word_embedding, self.wo_lstm)
        neg_r_encodings = self.dropout(neg_r_encodings)
        # neg_r_h: Size(total_neg_r_number, d_h_wo)
        neg_r_h = max_pooling_by_lens(neg_r_encodings, neg_r_lens)
        neg_r_h = neg_r_h.split([self.args.ns for _ in range(bs)])

        pos_score = self.cos(q_h, pos_r_h)
        pos_scores = [pos_score[i].unsqueeze(0).expand(self.args.ns, -1)
                      for i in range(bs)]

        neg_scores = []
        for i in range(bs):
            neg_score = self.cos(q_h[i].unsqueeze(0).expand(self.args.ns, -1), neg_r_h[i])
            neg_scores.append(neg_score.unsqueeze(-1))

        return pos_scores, neg_scores


    def ranking(self, sample):
        q, q_lens, ment_f, \
        pos_r, pos_r_lens, \
        neg_r, neg_r_lens, neg_r_blens = sample

        bs = q.size(0)
        assert bs == 1

        q_encodings, _ = self.encode(q, q_lens, self.word_embedding, self.wo_lstm)
        q_encodings = self.dropout(q_encodings)
        q_e_mask = length_array_to_mask_tensor(q_lens, value=ment_f, mask_symbol=1)
        if self.args.cuda:
            q_e_mask = q_e_mask.to(self.args.gpu)

        # q_h: Size(bs, d_h_wo)
        q_h = max_pooling_by_mask(q_encodings, q_e_mask)

        neg_r_encodings, _ = self.encode(neg_r, neg_r_lens, self.word_embedding, self.wo_lstm)
        neg_r_encodings = self.dropout(neg_r_encodings)
        # neg_r_h: Size(total_neg_r_number, d_h_wo)
        neg_r_h = max_pooling_by_lens(neg_r_encodings, neg_r_lens)

        ns = neg_r_h.size(0)

        scores = self.cos(q_h.expand(ns, -1), neg_r_h)
        return scores


    def encode(self, src_seq, src_lens, embedding, lstm):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """
        src_embed = embedding(src_seq)
        src_encodings, final_states = lstm(src_embed, src_lens)

        return src_encodings, final_states