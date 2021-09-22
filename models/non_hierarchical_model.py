# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : non_hierarchical_model.py
# @Software: PyCharm
"""

import sys
import copy
import pickle
import numpy as np
from operator import itemgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertConfig, BertTokenizer

sys.path.append("..")
from utils.embedding import Embeddings
from models.rnn import LSTM
from models.gnn import GraphTransformer
from models.attention import dot_prod_attention
from models.pointer_net import PointerNet
from utils.utils import identity, mk_graph_for_gnn, length_array_to_mask_tensor, step_to_av_step, \
    max_pooling_by_lens, instance_tensor_to_pool_without_class, pad_tensor_1d, get_inv_edge
from rules.grammar import AbstractQueryGraph, V_CLASS_IDS, E_CLASS_IDS
from utils.query_interface import KB_query_with_timeout

V_CLASS_NUM = len(V_CLASS_IDS)
E_CLASS_NUM = len(E_CLASS_IDS)


class NonHierarchicalModel(nn.Module):

    def __init__(self, args):

        super(NonHierarchicalModel, self).__init__()
        self.args = args

        self.dataset = args.dataset

        if args.dataset == "lcq":
            self.kb = "dbpedia"
        else:
            self.kb = "freebase"

        if not self.args.use_bert:
            self.tokenizer = pickle.load(open(args.wo_vocab, 'rb'))
            self.pad = self.tokenizer.lookup(self.tokenizer.pad_token)

            self.word_embedding = Embeddings(args.d_emb, self.tokenizer)
            
            self.mention_feature_embedding = nn.Embedding(2, args.d_emb)

            self.vertex_matching_feature_embedding = nn.Embedding(2, args.d_f)
            self.vertex_matching_feature_linear = nn.Linear(args.d_h + args.d_f, args.d_h, bias=False)

            self.edge_matching_feature_embedding = nn.Embedding(2, args.d_f)
            self.edge_matching_feature_linear = nn.Linear(args.d_h + args.d_f, args.d_h, bias=False)

            # lstm encoder for question
            self.encoder_lstm = LSTM(d_input=args.d_emb, d_h=args.d_h // 2,
                                     n_layers=args.n_lstm_layers, birnn=args.birnn, dropout=args.dropout)
            self.d_h_tmp = args.d_h
        else:
            self.config_bert = BertConfig.from_pretrained(args.bert_mode)
            self.encoder_bert = BertModel.from_pretrained(args.bert_mode)
            self.tokenizer_bert = BertTokenizer.from_pretrained(args.bert_mode)
            self.pad = self.tokenizer_bert.vocab["[PAD]"]
            self.d_h_tmp = self.config_bert.hidden_size

        self.vertex_class_embedding = nn.Embedding(V_CLASS_NUM, args.d_h)       # embeddings for vertex class
        self.edge_class_embedding = nn.Embedding(E_CLASS_NUM, args.d_h)         # embeddings for edge class

        self.vertex_embedding = nn.Embedding(100, args.d_h)             # vertex ID
        self.edge_embedding = nn.Embedding(100, args.d_h)               # edge ID

        self.vertex_segment_embedding = nn.Embedding(100, args.d_h)     # segment of vertex (subquery ID of vertex)
        self.edge_segment_embedding = nn.Embedding(100, args.d_h)       # segment of edge (subquery ID of edge)

        self.segment_switch_embedding = nn.Embedding(2, args.d_h)       # 0 and 1 denote False and True,
                                                                        # whether switch the segment (subquery)

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        self.decoder_cell_init = nn.Linear(self.d_h_tmp, args.d_h)

        # decoder for AQG generation
        self.decoder_lstm = nn.LSTMCell(args.d_h, args.d_h)
        self.enc_att_linear = nn.Linear(self.d_h_tmp, args.d_h)
        self.dec_input_linear = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        # encoder for AQG at each time step
        self.graph_encoder = GraphTransformer(n_blocks=args.n_gnn_blocks,
                                              hidden_size=args.d_h, dropout=args.dropout)
        # active function
        self.read_out_active = torch.tanh if args.readout == 'non_linear' else identity

        self.query_vec_to_av_vec = nn.Linear(args.d_h, args.d_h,
                                             bias=args.readout == 'non_linear')
        self.query_vec_to_ae_vec = nn.Linear(args.d_h, args.d_h,
                                             bias=args.readout == 'non_linear')
        self.query_vec_to_seg_vec = nn.Linear(args.d_h, args.d_h,
                                             bias=args.readout == 'non_linear')

        # pointer network for "add vertex" operation
        self.av_pointer_net = PointerNet(args.d_h, args.d_h, attention_type=args.att_type)

        # pointer network for "add edge" operation
        self.ae_pointer_net = PointerNet(args.d_h, args.d_h, attention_type=args.att_type)

        # pointer network for "select vertex" operation
        self.sv_pointer_net = PointerNet(args.d_h, args.d_h, attention_type=args.att_type)

        # classifier for predicting whether switch segment
        self.seg_readout_b = nn.Parameter(torch.FloatTensor(2).zero_())
        self.seg_readout = lambda q: F.linear(self.read_out_active(self.query_vec_to_ae_vec(q)),
                                              self.segment_switch_embedding.weight, self.seg_readout_b)

        # pointer networks for predicting to copy which vertex and which edge
        self.vertex_copy_pointer_net = PointerNet(args.d_h, args.d_h, attention_type=args.att_type)
        self.edge_copy_pointer_net = PointerNet(args.d_h, args.d_h, attention_type=args.att_type)
        # placeholder encoding that denotes "do not copy vertex" and "do not copy edge"
        self.not_v_copy_encoding = nn.Parameter(self.new_tensor(args.d_h))
        self.not_e_copy_encoding = nn.Parameter(self.new_tensor(args.d_h))

        # pointer networks for vertex linking and edge linking
        self.vertex_link_pointer_net =  PointerNet(args.d_h,
                                                   self.d_h_tmp,
                                                   attention_type=args.att_type)
        self.edge_link_pointer_net = PointerNet(args.d_h,
                                                self.d_h_tmp,
                                                attention_type=args.att_type)

    def forward(self, batch):
        # q:                    (bs, max_q_len)
        # q_lens:               (bs)
        # ment_f:               (bs, max_q_len)
        # match_f:              (total_e_num)
        # v_instance_tensor:    (total_v_num, max_v_len)
        # v_instance_lens:      (total_v_num)
        # v_instance_classes:   LIST, (total_v_num)   the class label of each vertex
        # v_instance_s_ids:     LIST, (total_v_num)   the sample id that each vertex belongs to
        # e_instance_tensor:    (total_e_num, max_e_len)
        # e_instance_lens:      (total_e_num)
        # e_instance_classes:   LIST, (total_e_num)   the class label of each edge
        # e_instance_s_ids:     LIST, (total_e_num)   the sample id that each edge belongs to
        q, q_lens, ment_f, match_f, \
        v_instance_tensor, v_instance_lens, v_instance_s_ids, v_instance_names, \
        e_instance_tensor, e_instance_lens, e_instance_s_ids, e_instance_names, \
        gold_aqgs, gold_graphs, gold_aqg_obj_labels, gold_obj_labels, \
        gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
        gold_v_copy_labels, gold_e_copy_labels, gold_segment_switch_labels, \
        v_st_idx, e_st_idx, v_instance_name2id, e_instance_name2id, \
        data = batch

        # for i, x in enumerate(e_instance_names[0]):
        #     print(i, e_instance_tensor[i], x)
        # exit()

        # v_instance_encodings:     (total_v_num, max_v_len, d_h)
        # v_instance_vec:           (total_v_num, d_h)
        if not self.args.use_bert:
            v_instance_encodings, _ = self.encode(v_instance_tensor, v_instance_lens)
        else:
            v_instance_encodings, _ = self.bert_encode(v_instance_tensor, v_instance_lens)
        v_instance_encodings = self.dropout(v_instance_encodings)
        v_instance_vec = max_pooling_by_lens(v_instance_encodings, v_instance_lens)

        if self.args.use_matching_feature:
            # TODO: vertex matching feature
            pass

        # e_instance_encodings:     (total_e_num, max_e_len, d_h)
        # e_instance_vec:           (total_e_num, d_h)
        if not self.args.use_bert:
            e_instance_encodings, _ = self.encode(e_instance_tensor, e_instance_lens)
        else:
            e_instance_encodings, _ = self.bert_encode(e_instance_tensor, e_instance_lens)
        e_instance_encodings = self.dropout(e_instance_encodings)
        e_instance_vec = max_pooling_by_lens(e_instance_encodings, e_instance_lens)

        if self.args.use_matching_feature:
            # match_f_embed:        (total_v_num, max_v_len, d_h)
            match_f_embed = self.matching_feature_embedding(match_f)
            e_instance_vec = self.matching_feature_linear(torch.cat([e_instance_vec, match_f_embed], dim=-1))

        v_instance_pool = instance_tensor_to_pool_without_class(v_instance_vec, v_instance_s_ids)
        v_instance_pool, v_instance_pool_lens = pad_tensor_1d(v_instance_pool, 0)
        e_instance_pool = instance_tensor_to_pool_without_class(e_instance_vec, e_instance_s_ids)
        e_instance_pool, e_instance_pool_lens = pad_tensor_1d(e_instance_pool, 0)

        # print(v_instance_pool_lens)
        # print([len(x) for x in v_instance_names])
        # print(e_instance_pool_lens)
        # print([len(x) for x in e_instance_names])
        # print()
        #
        # for s_id, labels in enumerate(gold_obj_labels):
        #     print(data[s_id]["id"])
        #     for i, x in enumerate(labels):
        #         if i == 0 or i % 3 == 1:
        #             print("av", v_instance_names[s_id][x])
        #         elif i % 3 == 0:
        #             print("ae", e_instance_names[s_id][x])
        #         else:
        #             print("sv", x)
        #     print("---------------------------------")
        # exit()

        v_instance_pool_mask = length_array_to_mask_tensor(v_instance_pool_lens)
        e_instance_pool_mask = length_array_to_mask_tensor(e_instance_pool_lens)
        if self.args.cuda:
            v_instance_pool_mask = v_instance_pool_mask.to(self.args.gpu)
            e_instance_pool_mask = e_instance_pool_mask.to(self.args.gpu)

        # for i, x in enumerate(e_instance_names[2]):
        #     print(i, x)
        # print(gold_obj_labels[2])
        # print(data[2]["id"])
        # print(data[2]["sparql"])
        # print(e_instance_pool_lens)
        # exit()

        # encoding question
        # q_encodings:      (bs, max_q_len, d_h)
        # q_mask:           (bs, max_q_len)     0: True, 1: False
        # enc_h_last:       (2, bs, d_h // 2)
        # enc_cell_last:    (2, bs, d_h // 2)
        if not self.args.use_bert:
            q_encodings, (enc_h_last, enc_cell_last) = self.encode(q,
                                                                   q_lens,
                                                                   mention_feature=ment_f if self.args.use_mention_feature else None)
            enc_context = torch.cat([enc_cell_last[0], enc_cell_last[1]], -1)
        else:
            q_encodings, enc_context = self.bert_encode(q,
                                                        q_lens,
                                                        mention_feature=ment_f if self.args.use_mention_feature else None)

        q_encodings = self.dropout(q_encodings)
        q_mask = length_array_to_mask_tensor(q_lens)
        q_v_mask = length_array_to_mask_tensor(q_lens, value=ment_f, mask_symbol=0) if self.args.dataset != "cwq" else q_mask
        q_e_mask = length_array_to_mask_tensor(q_lens, value=ment_f, mask_symbol=1) if self.args.dataset != "cwq" else q_mask
        if self.args.cuda:
            q_mask = q_mask.to(self.args.gpu)
            q_v_mask = q_v_mask.to(self.args.gpu)
            q_e_mask = q_e_mask.to(self.args.gpu)


        # initialize state for decoder
        # dec_init_state:    (bs, d_h), (bs, d_h)
        dec_init_state = self.init_decoder_state(enc_context, self.decoder_cell_init)
        h_last = dec_init_state

        # zero embedding for empty AQG, (d_h)
        zero_graph_encoding = Variable(self.new_tensor(self.args.d_h).zero_())

        batch_size = len(q)
        max_op_num = max([len(x) for x in gold_graphs])     # maximum number of operations

        # scores:           loss for graph generation
        # action_probs:     probabilities for each prediction (without softmax)
        scores = [[] for _ in range(batch_size)]
        action_probs = [[] for _ in range(batch_size)]

        # switch_seg_scores:    loss for segment switch mechanism
        switch_seg_scores = [[] for _ in range(batch_size)]

        #################################### Decoding for AQG generation ###############################################
        ################################################################################################################
        for t in range(max_op_num):

            # build encodings for AQG graph, vertex, edge at last timestep
            graph_encodings = []
            vertex_encodings = []
            edge_encodings = []

            # print("+++++++++++++++++++ t:", t)
            for s_id in range(batch_size):
                assert len(gold_graphs[s_id]) == len(gold_aqg_obj_labels[s_id])

                if t < len(gold_graphs[s_id]):
                    v_tensor, v_ins_tensor, v_segment_tensor, \
                    e_tensor, e_ins_tensor, e_segment_tensor, adj = gold_graphs[s_id][t]

                    vertex_embed = v_instance_pool[s_id].index_select(0, v_ins_tensor)
                    edge_embed = e_instance_pool[s_id].index_select(0, e_ins_tensor)

                    # for v_ins_id in v_ins_tensor.data:
                    #     print(v_instance_names[s_id][v_ins_id])
                    # for e_ins_id in e_ins_tensor.data:
                    #     print(e_instance_names[s_id][e_ins_id])
                    # print("----------------------------------")

                    # print(v_instance_vec.size())
                    # print(e_instance_vec.size())
                    # print(v_instance_pool[s_id].size())
                    # print(e_instance_pool[s_id].size())
                    # print(v_tensor)
                    # print(e_tensor)
                    # print(vertex_embed.size())
                    # print(edge_embed.size())
                    # exit()

                    # use the embeddings of vertex ID, edge ID
                    if self.args.use_id_embedding:
                        v_embed = self.vertex_embedding(v_tensor)           # Index vertex embedding
                        e_embed = self.edge_embedding(e_tensor)             # Index edge embedding
                        vertex_embed = torch.add(vertex_embed, v_embed)
                        edge_embed = torch.add(edge_embed, e_embed)

                    # use the embeddings of segment
                    if self.args.use_segment_embedding:
                        v_segment_embed = self.vertex_segment_embedding(v_segment_tensor)     # Segment vertex embedding
                        e_segment_embed = self.edge_segment_embedding(e_segment_tensor)       # Segment vertex embedding
                        vertex_embed = torch.add(vertex_embed, v_segment_embed)
                        edge_embed = torch.add(edge_embed, e_segment_embed)

                    # use gnn encoding the AQG
                    vertex_encoding, edge_encoding, \
                    graph_encoding = self.encode_graph(vertex_embed, edge_embed, adj)

                else:
                    # padding zero encodings
                    graph_encoding = zero_graph_encoding
                    zero_vertex_encoding = Variable(self.new_tensor((t - 2) // 3 + 2, self.args.d_h).zero_())
                    zero_edge_encoding = Variable(self.new_tensor((t - 1) // 3 * 2, self.args.d_h).zero_())
                    vertex_encoding = zero_vertex_encoding
                    edge_encoding = zero_edge_encoding

                graph_encodings.append(graph_encoding)
                vertex_encodings.append(vertex_encoding)
                edge_encodings.append(edge_encoding)

            # graph_encodings:      (bs, d_h)
            # vertex_encodings:     (bs, (t - 2) // 3 + 2, d_h)
            graph_encodings = torch.stack(graph_encodings)
            vertex_encodings = torch.stack(vertex_encodings)

            # since the number of edges in each graph may not be the same (due to copy edge), padding is required.
            # edge_encodings:       (bs, (t - 1) // 3 * 2, d_h)
            edge_encodings, edge_nums = pad_tensor_1d(edge_encodings, 0)
            if self.args.cuda:
                edge_encodings = edge_encodings.to(self.args.gpu)

            if self.args.use_graph_encoder:
                # dec_input_embeds:     (bs, d_h)
                dec_input_embeds = graph_encodings
            else:
                dec_input_embeds = h_last[0]

            # one step decoding
            # h_t:      (bs, d_h)
            # cell_t:   (bs, d_h)
            (h_t, cell_t), ctx, att = self.decode_step(self.decoder_lstm,
                                                       self.enc_att_linear,
                                                       self.dec_input_linear,
                                                       h_last, q_encodings, dec_input_embeds,
                                                       src_token_mask=q_e_mask if self.args.use_mention_feature else q_mask,
                                                       return_att_weight=True)

            # calculate probabilities for each operation
            if t == 0 or t % 3 == 1:
                v_instance_pool_mask_tmp = v_instance_pool_mask.clone()
                for b_id in range(batch_size):
                    ans_o_id = -1
                    for o_id in [i for i in range(v_instance_pool_lens[b_id])]:
                        if v_instance_names[b_id][o_id][1] == "ans":
                            ans_o_id = o_id
                    assert ans_o_id != -1
                    if t == 0:
                        for _i in range(v_instance_pool_lens[b_id]):
                            if _i == ans_o_id:
                                v_instance_pool_mask_tmp[b_id][_i] = 0
                            else:
                                v_instance_pool_mask_tmp[b_id][_i] = 1
                    else:
                        for _i in range(v_instance_pool_lens[b_id]):
                            if _i == ans_o_id:
                                v_instance_pool_mask_tmp[b_id][_i] = 1
                            else:
                                v_instance_pool_mask_tmp[b_id][_i] = 0

                action_prob = self.av_pointer_net(src_encodings=v_instance_pool, query_vec=h_t.unsqueeze(0),
                                                  src_token_mask=v_instance_pool_mask_tmp)
            elif t % 3 == 0:
                action_prob = self.ae_pointer_net(src_encodings=e_instance_pool, query_vec=h_t.unsqueeze(0),
                                                  src_token_mask=e_instance_pool_mask)
            else:
                # Cannot select the newly added point
                sv_mask = torch.cat([self.new_long_tensor(vertex_encodings.size(0), vertex_encodings.size(1) - 1).fill_(1),
                                     self.new_long_tensor(vertex_encodings.size(0), 1).zero_()], -1)
                action_prob = self.sv_pointer_net(src_encodings=vertex_encodings, query_vec=h_t.unsqueeze(0),
                                                  src_token_mask=sv_mask==0)

            # recording probs for AQG generation
            for s_id in range(batch_size):
                if t < len(gold_obj_labels[s_id]):
                    action_probs[s_id].append(action_prob[s_id])

            action_prob = F.softmax(action_prob, dim=-1)
            # save softmax prob for calculate loss
            for s_id in range(batch_size):
                if t < len(gold_obj_labels[s_id]):
                    act_prob_t_i = action_prob[s_id, gold_obj_labels[s_id][t]]
                    scores[s_id].append(act_prob_t_i)

            # calculate loss for switching segment
            if self.args.dataset == "cwq":
                if t == 0 or t % 3 == 1:
                    switch_seg_prob = F.softmax(self.seg_readout(h_t), dim=-1)
                    for s_id in range(batch_size):
                        if t < len(gold_aqg_obj_labels[s_id]) and gold_aqg_obj_labels[s_id][t] != V_CLASS_IDS["end"]:
                            # get time step of add vertex operation
                            t_av = step_to_av_step(t)
                            seg_prob_t_i = switch_seg_prob[s_id, gold_segment_switch_labels[s_id][t_av]]
                            switch_seg_scores[s_id].append(seg_prob_t_i)

            h_last = (h_t, cell_t)

        # AQG generation loss
        score = torch.stack([torch.stack(score_i, dim=0).log().sum() for score_i in scores], dim=0)

        # Only CWQ has sub-query
        if self.args.dataset == "cwq":
            # AQG generation loss + segment switch loss
            switch_seg_score = torch.stack([torch.stack(score_i, dim=0).log().sum() for score_i in switch_seg_scores], dim=0)
            score = torch.add(score, switch_seg_score)

        return -score, action_probs

    def generate(self, sample, beam_size=5, sparql_cache=None):

        # q:                    (bs, max_q_len)
        # q_lens:               (bs)
        # ment_f:               (bs, max_q_len)
        # match_f:              (total_e_num)
        # v_instance_tensor:    (total_v_num, max_v_len)
        # v_instance_lens:      (total_v_num)
        # v_instance_classes:   LIST, (total_v_num)   the class label of each vertex
        # v_instance_s_ids:     LIST, (total_v_num)   the sample id that each vertex belongs to
        # e_instance_tensor:    (total_e_num, max_e_len)
        # e_instance_lens:      (total_e_num)
        # e_instance_classes:   LIST, (total_e_num)   the class label of each edge
        # e_instance_s_ids:     LIST, (total_e_num)   the sample id that each edge belongs to
        q, q_lens, ment_f, match_f, \
        v_instance_tensor, v_instance_lens, v_instance_s_ids, v_instance_names, \
        e_instance_tensor, e_instance_lens, e_instance_s_ids, e_instance_names, \
        gold_aqgs, gold_graphs, gold_aqg_obj_labels, gold_obj_labels, \
        gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
        gold_v_copy_labels, gold_e_copy_labels, gold_segment_switch_labels, \
        v_st_idx, e_st_idx, v_instance_name2id, e_instance_name2id, \
        data = sample

        # for i, x in enumerate(v_instance_names[0]):
        #     print(i, v_instance_tensor[i], x)
        # for i, x in enumerate(e_instance_names[0]):
        #     print(i, e_instance_tensor[i], x)

        data = data[0]
        v_instance_names = v_instance_names[0]
        e_instance_names = e_instance_names[0]
        v_instance_name2id = v_instance_name2id[0]
        e_instance_name2id = e_instance_name2id[0]

        v_st_idx = [[st, v_class] for v_class, st in v_st_idx[0].items()] + [[len(v_instance_names), -1]]
        e_st_idx = [[st, e_class] for e_class, st in e_st_idx[0].items()] + [[len(e_instance_names), -1]]
        v_st_idx.sort(key=lambda x: x[0])
        e_st_idx.sort(key=lambda x: x[0])

        # v_instance_encodings:     (total_v_num, max_v_len, d_h)
        # v_instance_vec:           (total_v_num, d_h)
        if not self.args.use_bert:
            v_instance_encodings, _ = self.encode(v_instance_tensor, v_instance_lens)
        else:
            v_instance_encodings, _ = self.bert_encode(v_instance_tensor, v_instance_lens)
        v_instance_vec = max_pooling_by_lens(v_instance_encodings, v_instance_lens)

        if self.args.use_matching_feature:
            # TODO: vertex matching feature
            pass

        # e_instance_encodings:     (total_e_num, max_e_len, d_h)
        # e_instance_vec:           (total_e_num, d_h)
        if not self.args.use_bert:
            e_instance_encodings, _ = self.encode(e_instance_tensor, e_instance_lens)
        else:
            e_instance_encodings, _ = self.bert_encode(e_instance_tensor, e_instance_lens)
        e_instance_vec = max_pooling_by_lens(e_instance_encodings, e_instance_lens)

        if self.args.use_matching_feature:
            # match_f_embed:        (total_v_num, max_v_len, d_h)
            match_f_embed = self.matching_feature_embedding(match_f)
            e_instance_vec = self.matching_feature_linear(torch.cat([e_instance_vec, match_f_embed], dim=-1))

        v_instance_pool = instance_tensor_to_pool_without_class(v_instance_vec, v_instance_s_ids)
        v_instance_pool, v_instance_pool_lens = pad_tensor_1d(v_instance_pool, 0)
        e_instance_pool = instance_tensor_to_pool_without_class(e_instance_vec, e_instance_s_ids)
        e_instance_pool, e_instance_pool_lens = pad_tensor_1d(e_instance_pool, 0)

        v_instance_pool_mask = length_array_to_mask_tensor(v_instance_pool_lens)
        e_instance_pool_mask = length_array_to_mask_tensor(e_instance_pool_lens)
        if self.args.cuda:
            v_instance_pool_mask = v_instance_pool_mask.to(self.args.gpu)
            e_instance_pool_mask = e_instance_pool_mask.to(self.args.gpu)

        if self.args.dataset == "lcq":
            for i, (e_class, e_name, e_true_name) in enumerate(e_instance_names):
                if e_name in ["MAX", "MIN", "=", "!=", ">", "<", ">=", "<=", "during", "overlap", "ASC", "DESC"]:
                    e_instance_pool_mask[0][i] = 1
        else:
            for i, (e_class, e_name, e_true_name) in enumerate(e_instance_names):
                if e_name in ["ASK"]:
                    e_instance_pool_mask[0][i] = 1

        # for i, x in enumerate(v_instance_names):
        #     print(i, x)
        # print()

        # encoding question
        # q_encodings:      (bs, max_q_len, d_h)
        # q_mask:           (bs, max_q_len)     0: True, 1: False
        # enc_h_last:       (2, bs, d_h // 2)
        # enc_cell_last:    (bs, d_h)
        if not self.args.use_bert:
            q_encodings, (enc_h_last, enc_cell_last) = self.encode(q,
                                                                   q_lens,
                                                                   mention_feature=ment_f if self.args.use_mention_feature else None)
            enc_context = torch.cat([enc_cell_last[0], enc_cell_last[1]], -1)
        else:
            q_encodings, enc_context = self.bert_encode(q,
                                                        q_lens,
                                                        mention_feature=ment_f if self.args.use_mention_feature else None)
        q_encodings = self.dropout(q_encodings)
        q_mask = length_array_to_mask_tensor(q_lens)
        q_v_mask = length_array_to_mask_tensor(q_lens, value=ment_f, mask_symbol=0) if self.args.dataset != "cwq" else q_mask
        q_e_mask = length_array_to_mask_tensor(q_lens, value=ment_f, mask_symbol=1) if self.args.dataset != "cwq" else q_mask
        if self.args.cuda:
            q_mask = q_mask.to(self.args.gpu)
            q_v_mask = q_v_mask.to(self.args.gpu)
            q_e_mask = q_e_mask.to(self.args.gpu)

        # when testing, only handle one question at each batch
        assert len(q) == 1

        #################################### Decoding for AQG generation ###############################################
        ################################################################################################################
        # initialize state for decoder
        # dec_init_state:    (bs, d_h), (bs, d_h)
        dec_init_state = self.init_decoder_state(enc_context, self.decoder_cell_init)
        h_last = dec_init_state

        t = 0
        # initialize one empty AQG
        aqg = AbstractQueryGraph()
        aqg.init_state()

        beams = [aqg]                   # Initially, the beam set only consists of an empty AQG.
        completed_beams = []            # LIST, each element is (AQG, time_step, previous aqg id)

        # if the number of completed AQG is equal with beam size, BREAK
        # or if over than the predefined operation numbers, BREAK
        while len(completed_beams) < beam_size and t < self.args.max_num_op:

            # expand question encoding to match the current number of beams
            # exp_q_encodings:      (beam_num, max_q_len, d_h)
            exp_q_encodings = q_encodings.expand(len(beams), q_encodings.size(1), q_encodings.size(2))

            # build encodings for AQG graph, vertex, edge at last timestep
            graph_encodings = []
            vertex_encodings = []

            for b_id, aqg in enumerate(beams):
                # get the state of the last AQG
                vertices, v_classes, v_segments, edges, e_classes, e_segments, triples = aqg.get_state()
                v_tensor, v_class_tensor, v_segment_tensor, \
                e_tensor, e_class_tensor, e_segment_tensor, adj = mk_graph_for_gnn(vertices, v_classes, v_segments,
                                                                                   edges, e_classes, e_segments,
                                                                                   triples)

                v_ins = []
                for v in vertices:
                    v_class = aqg.get_vertex_label(v)
                    if v_class == V_CLASS_IDS["var"]:
                        v_ins.append(v_instance_name2id["var"])
                    elif v_class == V_CLASS_IDS["ans"]:
                        v_ins.append(v_instance_name2id["ans"])
                    else:
                        v_ins.append(v_instance_name2id[aqg.get_vertex_instance(v)[-1]])
                v_ins_tensor = torch.LongTensor(v_ins)

                e_ins = []
                for e in edges:
                    e_class = aqg.get_edge_label(e)
                    if e_class % 2 == 0:
                        e_ins.append(2 * e_instance_name2id[aqg.get_edge_instance(e)[-1]])
                    else:
                        e_ins.append(2 * e_instance_name2id[aqg.get_edge_instance(e)[-1]] + 1)
                e_ins_tensor = torch.LongTensor(e_ins)

                # move to GPU
                if self.args.cuda:
                    v_tensor = v_tensor.to(self.args.gpu)
                    e_tensor = e_tensor.to(self.args.gpu)
                    v_ins_tensor = v_ins_tensor.to(self.args.gpu)
                    e_ins_tensor = e_ins_tensor.to(self.args.gpu)
                    v_segment_tensor = v_segment_tensor.to(self.args.gpu)
                    e_segment_tensor = e_segment_tensor.to(self.args.gpu)
                    adj = adj.to(self.args.gpu)

                vertex_embed = v_instance_pool[0].index_select(0, v_ins_tensor)
                edge_embed = e_instance_pool[0].index_select(0, e_ins_tensor)

                # use the embeddings of vertex ID, edge ID
                if self.args.use_id_embedding:
                    v_embed = self.vertex_embedding(v_tensor)       # Index vertex embedding
                    e_embed = self.edge_embedding(e_tensor)         # Index edge embedding
                    vertex_embed = torch.add(vertex_embed, v_embed)
                    edge_embed = torch.add(edge_embed, e_embed)

                # use the embeddings of segment
                if self.args.use_segment_embedding:
                    v_segment_embed = self.vertex_segment_embedding(v_segment_tensor)   # Segment vertex embedding
                    e_segment_embed = self.edge_segment_embedding(e_segment_tensor)     # Segment edge embedding
                    vertex_embed = torch.add(vertex_embed, v_segment_embed)
                    edge_embed = torch.add(edge_embed, e_segment_embed)

                # use gnn encoding the AQG
                vertex_encoding, edge_encoding, graph_encoding = self.encode_graph(vertex_embed, edge_embed, adj)

                graph_encodings.append(graph_encoding)
                vertex_encodings.append(vertex_encoding)

            graph_encodings = torch.stack(graph_encodings)
            vertex_encodings = torch.stack(vertex_encodings)

            if self.args.use_graph_encoder:
                # dec_input_embeds:     (beam_num, d_h)
                dec_input_embeds = graph_encodings
            else:
                dec_input_embeds = h_last[0]

            # one step decoding
            # h_t:      (beam_num, d_h)
            # cell_t:   (beam_num, d_h)
            (h_t, cell_t), ctx, att = self.decode_step(self.decoder_lstm,
                                                       self.enc_att_linear,
                                                       self.dec_input_linear,
                                                       h_last, exp_q_encodings, dec_input_embeds,
                                                       src_token_mask=q_e_mask if self.args.use_mention_feature else q_mask,
                                                       return_att_weight=True)

            ####################################### add vertex operation ###############################################
            if t == 0 or t % 3 == 1:
                op = 'av'
                # action_prob:      (beam_num, V_CLASS_NUM), probs of add vertex operation
                # switch_seg_prob:  (beam_num, 2), probs of segment switching
                # v_copy_prob:      (beam_num, v_num_at_t + 1), probs of whether copy vertex
                action_prob = self.av_pointer_net(src_encodings=v_instance_pool, query_vec=h_t.unsqueeze(0),
                                                  src_token_mask=v_instance_pool_mask)
                action_prob = F.log_softmax(action_prob, dim=-1)
                switch_seg_prob = F.log_softmax(self.seg_readout(h_t), dim=-1)

                # save the possible directions for AQG expansion
                new_aqg_meta = []
                for b_id, aqg in enumerate(beams):

                    ans_o_id = -1
                    for o_id in [i for i in range(len(v_instance_pool[0]))]:
                        if v_instance_names[o_id][1] == "ans":
                            ans_o_id = o_id
                    assert ans_o_id != -1

                    if t == 0:
                        # first vertex is always in the class of "ans"
                        meta_entry = {
                            "op": op,
                            "obj": ans_o_id,
                            "obj_score": action_prob[b_id, ans_o_id],
                            "seg": 0,
                            "seg_score": None,
                            "new_aqg_score": aqg.get_score() + action_prob[b_id, ans_o_id].cpu().detach().numpy(),
                            "prev_aqg_id": b_id
                        }
                        new_aqg_meta.append(meta_entry)
                        continue

                    # enumerate add vertex object
                    o_range = [i for i in range(len(v_instance_pool[0])) if i != ans_o_id]
                    for o_id in o_range:
                        # update probability
                        new_aqg_score = aqg.get_score() + action_prob[b_id, o_id].cpu().detach().numpy()

                        # end signal for AQG generation
                        if v_instance_names[o_id] == "end":
                            meta_entry = {
                                "op": op,
                                "obj": o_id,
                                "obj_score": action_prob[b_id, o_id],
                                "seg": 0,
                                "seg_score": None,
                                "new_aqg_score": new_aqg_score,
                                "prev_aqg_id": b_id
                            }
                            new_aqg_meta.append(meta_entry)
                            continue

                        # enumerate whether switch segment
                        if self.args.dataset == "cwq" and aqg.vertex_number >= self.args.v_num_start_switch_segment:
                            seg_range = [0, 1]
                        else:
                            seg_range = [0]

                        for seg_id in seg_range:
                            if self.args.dataset == "cwq":
                                # update probability by adding segment switching probability
                                new_aqg_score_1 = new_aqg_score + switch_seg_prob[b_id, seg_id].cpu().detach().numpy()
                            else:
                                new_aqg_score_1 = new_aqg_score

                            meta_entry = {
                                "op": op,
                                "obj": o_id,
                                "obj_score": action_prob[b_id, o_id],
                                "seg": seg_id,
                                "seg_score": switch_seg_prob[b_id, seg_id],
                                "new_aqg_score": new_aqg_score_1,
                                "prev_aqg_id": b_id
                            }
                            new_aqg_meta.append(meta_entry)

            ######################################### add edge operation ###############################################
            elif t % 3 == 0:

                e_instance_pool_mask_tmp = torch.ones(h_t.size(0), e_instance_pool.size(1))
                if self.args.cuda:
                    e_instance_pool_mask_tmp = e_instance_pool_mask_tmp.to(self.args.gpu)
                for _i in range(e_instance_pool_mask_tmp.size(0)):
                    e_instance_pool_mask_tmp[_i][:] = e_instance_pool_mask[:]

                for b_id, aqg in enumerate(beams):
                    for _i, (_e_class, _e_name, _e_true_name) in enumerate(e_instance_names):
                        if e_instance_pool_mask_tmp[b_id][_i] == 1: continue
                        for _e_id, _e_ins in aqg.e_instances.items():
                            if _e_name == _e_ins[-1]:
                                e_instance_pool_mask_tmp[b_id][_i] = 1

                    if self.args.dataset == "lcq":
                        now_s_class = aqg.get_vertex_label(aqg.cur_v_slc)
                        now_o_class = aqg.get_vertex_label(aqg.cur_v_add)
                        if now_s_class == V_CLASS_IDS["type"] or now_o_class == V_CLASS_IDS["type"]:
                            for _i, (_e_class, _e_name, _e_true_name) in enumerate(e_instance_names):
                                if _e_name == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                                    e_instance_pool_mask_tmp[b_id][_i] = 0
                                else:
                                    e_instance_pool_mask_tmp[b_id][_i] = 1
                        else:
                            for _i, (_e_class, _e_name, _e_true_name) in enumerate(e_instance_names):
                                if _e_name == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                                    e_instance_pool_mask_tmp[b_id][_i] = 1

                    if self.args.use_kb_constraint:

                        all_var = True
                        for v, v_class in aqg.v_labels.items():
                            if v_class not in [V_CLASS_IDS["ans"], V_CLASS_IDS["var"]]:
                                all_var = False
                        if all_var:
                            continue

                        cnt_timeout = 0
                        for _i, (_e_class, _e_name, _e_true_name) in enumerate(e_instance_names):
                            if e_instance_pool_mask_tmp[b_id][_i] == 1:
                                continue
                            tmp_aqg = copy.deepcopy(aqg)
                            e_id = len(tmp_aqg.edges)

                            e_class = -1
                            for j in range(len(e_st_idx) - 1):
                                if _i >= e_st_idx[j][0] and _i < e_st_idx[j + 1][0]:
                                    e_class = e_st_idx[j][1]

                            assert e_class != -1
                            assert e_class % 2 == 0

                            e_true_name = e_instance_names[_i][2]
                            direction = e_true_name[-1]
                            if direction == "-":
                                e_class += 1

                            tmp_aqg.update_state("ae", [e_class, -1])
                            tmp_aqg.set_edge_instance(e_id, [_i, _e_name])
                            tmp_aqg.set_edge_instance(get_inv_edge(e_id), [_i, _e_name])

                            try:
                                tmp_queries = tmp_aqg.to_temporary_sparql_query(kb=self.kb)
                                if not tmp_queries:
                                    e_instance_pool_mask_tmp[b_id][_i] = 1
                                    continue

                                for one_query in tmp_queries:
                                    if sparql_cache is not None and data["id"] in sparql_cache and one_query in sparql_cache[data["id"]]:
                                        result = sparql_cache[data["id"]][one_query]
                                    else:
                                        result = KB_query_with_timeout(one_query, self.args.kb_endpoint)

                                    if sparql_cache is not None:
                                        if data["id"] not in sparql_cache:
                                            sparql_cache[data["id"]] = {}
                                        sparql_cache[data["id"]][one_query] = result
                                    # print(t)
                                    # print(one_query)
                                    # print(result)
                                    # print()
                                    if result == "TimeOut":
                                        result = [False]
                                        cnt_timeout += 1
                                    # print(src_mask[b_id])
                                    if not result[0]:
                                        e_instance_pool_mask_tmp[b_id][_i] = 1
                                        break
                                if cnt_timeout >= 3:
                                    for _j, (_e_class, _e_name, _e_true_name) in enumerate(e_instance_names):
                                        e_instance_pool_mask_tmp[b_id][_j] = 1
                                    break
                            except:
                                # if self.args.dataset != "lcq":
                                #     src_mask[b_id][i] = 1
                                e_instance_pool_mask_tmp[b_id][_i] = 1
                                continue

                op = "ae"
                # action_prob:      (beam_num, E_CLASS_NUM), probs of add edge operation
                # e_copy_prob:      (beam_num, e_num_at_t + 1), probs of whether copy vertex
                action_prob = self.ae_pointer_net(src_encodings=e_instance_pool, query_vec=h_t.unsqueeze(0),
                                                  src_token_mask=e_instance_pool_mask_tmp)
                action_prob = F.log_softmax(action_prob, dim=-1)

                new_aqg_meta = []
                for b_id, aqg in enumerate(beams):
                    # Enumerate add edge object
                    o_range = [i for i in range(len(e_instance_pool[0]))]
                    for o_id in o_range:
                        if e_instance_pool_mask_tmp[b_id][o_id] == 1:
                            continue
                        # update probability
                        new_aqg_score = aqg.get_score() + action_prob[b_id, o_id].cpu().detach().numpy()

                        meta_entry = {
                            "op": op,
                            "obj": o_id,
                            "obj_score": action_prob[b_id, o_id],
                            "seg": 0,
                            "seg_score": None,
                            "new_aqg_score": new_aqg_score,
                            "prev_aqg_id": b_id
                        }
                        new_aqg_meta.append(meta_entry)

            ###################################### select vertex operation #############################################
            else:
                op = "sv"
                # cannot select the newly added vertex
                sv_mask = torch.cat([self.new_long_tensor(vertex_encodings.size(0), vertex_encodings.size(1) - 1).fill_(1),
                                     self.new_long_tensor(vertex_encodings.size(0), 1).zero_()], -1)
                # action_prob:      (beam_num, v_num_at_t), probs of select vertex operation
                action_prob = self.sv_pointer_net(src_encodings=vertex_encodings, query_vec=h_t.unsqueeze(0),
                                                  src_token_mask=sv_mask==0)
                action_prob = F.log_softmax(action_prob, dim=-1)

                new_aqg_meta = []
                for b_id, aqg in enumerate(beams):
                    # enumerate vertex in current AQG to select
                    o_range = [i for i in range(vertex_encodings.size(1) - 1)]
                    for o_id in o_range:
                        o_score = action_prob[b_id, o_id]
                        # update probability
                        new_aqg_score = aqg.get_score() + o_score.cpu().detach().numpy()
                        meta_entry = {
                            "op": op,
                            "obj": o_id,
                            "obj_score": action_prob[b_id, o_id],
                            "seg": 0,
                            "seg_score": None,
                            "new_aqg_score": new_aqg_score,
                            "prev_aqg_id": b_id
                        }
                        new_aqg_meta.append(meta_entry)

            if not new_aqg_meta:
                break

            # new_aqg_scores:       (beam_num)
            new_aqg_scores = self.new_tensor([x["new_aqg_score"] for x in new_aqg_meta])
            # select top-k aqg with highest probs
            k = min(new_aqg_scores.size(0), beam_size - len(completed_beams))
            top_new_aqg_scores, meta_ids = torch.topk(new_aqg_scores, k=len(new_aqg_scores))

            live_aqg_ids = []
            new_beams = []
            cnt = 0
            for new_aqg_score, meta_id in zip(top_new_aqg_scores.cpu().detach().numpy(), meta_ids.data.cpu()):
                if cnt >= k:
                    break
                aqg_meta_entry = new_aqg_meta[meta_id]
                op = aqg_meta_entry["op"]
                obj = aqg_meta_entry["obj"]
                prev_aqg_id = aqg_meta_entry["prev_aqg_id"]
                prev_aqg = beams[prev_aqg_id]

                # build new AQG
                new_aqg = copy.deepcopy(prev_aqg)
                new_aqg.update_score(new_aqg_score)

                if op == "av" and v_instance_names[obj][1] == "end":
                    # generation is end

                    # if self.args.dataset == "lcq":
                    #     if new_aqg.check_final_structure(data["instance_pool"], self.args.dataset):
                    #         completed_beams.append([new_aqg, prev_aqg_id, t])
                    # else:
                    #     completed_beams.append([new_aqg, prev_aqg_id, t])
                    completed_beams.append([new_aqg, prev_aqg_id, t])
                else:
                    # update AQG state
                    if op == "av":
                        v_class = -1
                        for j in range(len(v_st_idx) - 1):
                            if obj >= v_st_idx[j][0] and obj < v_st_idx[j + 1][0]:
                                v_class = v_st_idx[j][1]
                        assert v_class != -1
                        v_id = len(new_aqg.vertices)
                        switch_segment = aqg_meta_entry["seg"]
                        new_aqg.update_state("av", [v_class, -1, switch_segment])
                        new_aqg.set_vertex_instance(v_id, [obj, v_instance_names[obj][1]])
                    elif op == "ae":
                        e_class = -1
                        for j in range(len(e_st_idx) - 1):
                            if obj >= e_st_idx[j][0] and obj < e_st_idx[j + 1][0]:
                                e_class = e_st_idx[j][1]

                        assert e_class != -1
                        assert e_class % 2 == 0

                        e_true_name = e_instance_names[obj][2]
                        direction = e_true_name[-1]
                        if direction == "-":
                            e_class += 1
                        e_id = len(new_aqg.edges)
                        new_aqg.update_state("ae", [e_class, -1])
                        new_aqg.set_edge_instance(e_id, [obj, e_instance_names[obj][1]])
                        new_aqg.set_edge_instance(get_inv_edge(e_id), [obj, e_instance_names[obj][1]])
                    else:
                        new_aqg.update_state("sv", obj)

                    if self.args.dataset == "lcq" and not new_aqg.check_temporary_structure(self.args.dataset):
                        continue

                    new_beams.append(new_aqg)
                    live_aqg_ids.append(prev_aqg_id)
                cnt += 1

            if not live_aqg_ids:
                break

            # print()
            # print("######################################################################################")
            # print("Timestep: {}, Operation: {} ".format(t, op))
            # for _aqg in new_beams:
            #     _aqg.show_state()

            h_last = (h_t[live_aqg_ids], cell_t[live_aqg_ids])
            beams = new_beams

            t += 1

        # sort by total probability
        completed_beams.sort(key=lambda x: -x[0].get_score())
        completed_beams = [x[0] for x in completed_beams]
        # do not complete any AQG
        if len(completed_beams) == 0:
            return []

        # print()
        # print("===============================================================================================")
        # print("Complete Beams:")
        # for x, _, _ in completed_beams:
        #     x.show_state()

        return completed_beams

    def encode_graph(self, vertex_tensor, edge_tensor, adj):
        """
        encode a graph by the adjacency matrix
        @param vertex_tensor:       (v_num, d_h)
        @param edge_tensor:         (e_num, d_h)
        @param adj:                 (v_num + e_num + 1, v_num + e_num + 1)
        @return:                    (v_num, d_h), (e_num, d_h), (d_h)
        """
        return self.graph_encoder(vertex_tensor, edge_tensor, adj)

    def encode(self, src_seq, src_lens, init_states=None, mention_feature=None, return_embedding=False):
        """
        encode the source sequence
        @param src_seq:             (bs, max_seq_len)
        @param src_lens:            (bs)
        @param return_embedding:    whether return word embeddings
        @return:
            src_encodings:          (bs, max_seq_len, d_h)
            last_state, last_cell:  (bs, d_h), (bs, d_h)
        """
        src_embed = self.word_embedding(src_seq)

        if mention_feature is not None:
            ment_f_embed = self.mention_feature_embedding(mention_feature)
            src_embed = torch.add(src_embed, ment_f_embed)

        src_encodings, final_states = self.encoder_lstm(src_embed, src_lens, init_states=init_states)

        if return_embedding:
            return src_encodings, final_states, src_embed
        return src_encodings, final_states

    def bert_encode(self, src_seq, src_lens, mention_feature=None):
        """
        encode the source sequence
        @param src_seq:             (bs, max_seq_len)
        @param src_lens:            (bs)
        @param return_embedding:    whether return word embeddings
        @return:
            src_encodings:          (bs, max_seq_len, d_h)
            last_state, last_cell:  (bs, d_h), (bs, d_h)
        """
        # BERT encoding
        if mention_feature is None:
            mention_feature = torch.zeros_like(src_seq)
            if src_seq.cuda:
                mention_feature = mention_feature.to(src_seq.device)

        src_mask = length_array_to_mask_tensor(src_lens, reverse=False)
        if src_seq.cuda:
            src_mask = src_mask.to(src_seq.device)
        src_encodings, src_pooling = self.encoder_bert(input_ids=src_seq,
                                                       token_type_ids=mention_feature,
                                                       attention_mask=src_mask)
        return src_encodings, src_pooling

    def decode_step(self, decoder_lstm, enc_att_linear, dec_input_linear, h_last,
                    src_encodings, dec_input_encodings, src_token_mask=None, return_att_weight=False):
        """
        one decoding step
        @param h_last:                  (bs, d_h)
        @param src_encodings:           (bs, max_seq_len, d_h)
        @param dec_input_encodings:     (bs, d_h)
        @param src_token_mask:          (bs, max_seq_len)
        @return:
        """
        src_encodings_linear = enc_att_linear(src_encodings)

        # context_t:        (bs, d_h)
        # alpha_t:          (bs, max_seq_len), attention weights
        context_t, alpha_t = dot_prod_attention(dec_input_encodings, src_encodings,
                                                src_encodings_linear, mask=src_token_mask)

        dec_input = torch.tanh(dec_input_linear(torch.cat([dec_input_encodings, context_t], 1)))
        dec_input = self.dropout(dec_input)

        h_t, cell_t = decoder_lstm(dec_input, h_last)

        if return_att_weight:
            return (h_t, cell_t), context_t, alpha_t
        else:
            return (h_t, cell_t), context_t

    def init_decoder_state(self, enc_last_cell, decoder_cell_init):
        """
        @param enc_last_cell:   (bs, d_h)
        """
        h_0 = decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)
        return h_0, Variable(self.new_tensor(h_0.size()).zero_())


def mask_av_action_prob(dataset, action_prob, t, aqg, data):
    bs = action_prob.size(0)
    assert action_prob.size(1) == len(V_CLASS_IDS)
    if t == 0:
        mask = np.ones((bs, len(V_CLASS_IDS)), dtype=np.uint8)
        for i in range(bs):
            mask[i][V_CLASS_IDS["ans"]] = 0
        mask = torch.ByteTensor(mask)
    else:
        mask = np.zeros((bs, len(V_CLASS_IDS)), dtype=np.uint8)
        for i in range(bs):
            mask[i][V_CLASS_IDS["ans"]] = 1
            for o_id in [V_CLASS_IDS["type"], V_CLASS_IDS["ent"]]:
                if o_id not in data[i]["instance_pool"]["vertex"]:
                    mask[i][o_id] = 1
            if dataset == "lcq":
                mask[i][V_CLASS_IDS["val"]] = 1
                # if aqg[i] is not None:
                #     cnt_var = 0
                #     for v, v_class in aqg[i].v_labels.items():
                #         if v_class == V_CLASS_IDS["var"]:
                #             cnt_var += 1
                #     if cnt_var > 0:
                #         mask[i][V_CLASS_IDS["var"]] = 1
        mask = torch.ByteTensor(mask)
    if action_prob.is_cuda:
        mask = mask.to(action_prob.device)
    action_prob.masked_fill_(mask.bool(), -float('inf'))
    return action_prob

def mask_ae_action_prob(dataset, action_prob, aqg, data):
    bs = action_prob.size(0)
    assert action_prob.size(1) == len(E_CLASS_IDS)
    mask = np.zeros((bs, len(E_CLASS_IDS)), dtype=np.uint8)
    for i in range(bs):

        for o_id in [E_CLASS_IDS["agg+"], E_CLASS_IDS["cmp+"], E_CLASS_IDS["ord+"], E_CLASS_IDS["rel+"]]:
            if o_id not in data[i]["instance_pool"]["edge"]:
                mask[i][o_id] = 1
                mask[i][o_id + 1] = 1

        if dataset == "lcq":
            mask[i][E_CLASS_IDS["cmp+"]] = 1
            mask[i][E_CLASS_IDS["cmp+"] + 1] = 1
            mask[i][E_CLASS_IDS["ord+"]] = 1
            mask[i][E_CLASS_IDS["ord+"] + 1] = 1

    mask = torch.ByteTensor(mask)
    if action_prob.is_cuda:
        mask = mask.to(action_prob.device)
    action_prob.masked_fill_(mask.bool(), -float('inf'))
    return action_prob