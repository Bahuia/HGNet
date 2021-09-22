# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/30
# @Author  : Yongrui Chen
# @File    : model.py
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
from utils.utils import identity, mk_graph_for_gnn, length_array_to_mask_tensor, step_to_av_step, step_to_ae_step, \
    max_pooling_by_lens, max_pooling_by_mask, instance_tensor_to_pool, pad_tensor_1d, get_inv_edge, \
    text_to_tensor_1d, tokenize_word_sentence_bert, tokenize_word_sentence, is_relation, cal_literal_matching_score
from rules.grammar import AbstractQueryGraph, V_CLASS_IDS, E_CLASS_IDS, cal_edge_matching_total_score, get_relation_true_name, get_relation_last_name
from utils.query_interface import KB_query, KB_query_with_timeout

V_CLASS_NUM = len(V_CLASS_IDS)
E_CLASS_NUM = len(E_CLASS_IDS)


class AQGNet(nn.Module):

    def __init__(self, args):

        super(AQGNet, self).__init__()
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

        self.aqg_decoder_cell_init = nn.Linear(self.d_h_tmp, args.d_h)
        self.vertex_decoder_cell_init = nn.Linear(self.d_h_tmp, args.d_h)
        self.edge_decoder_cell_init = nn.Linear(self.d_h_tmp, args.d_h)

        # decoder for AQG generation
        self.aqg_decoder_lstm = nn.LSTMCell(args.d_h, args.d_h)
        self.aqg_enc_att_linear = nn.Linear(self.d_h_tmp, args.d_h)
        self.aqg_dec_input_linear = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)

        # decoder for predicting instance of each vertex (vertex linking)
        self.vertex_decoder_lstm = nn.LSTMCell(args.d_h, args.d_h)
        self.vertex_enc_att_linear = nn.Linear(self.d_h_tmp, args.d_h)
        self.vertex_dec_input_linear = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)
        self.vertex_dec_output_linear = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)

        # decoder for predicting instance of each edge (edge linking)
        self.edge_decoder_lstm = nn.LSTMCell(args.d_h, args.d_h)
        self.edge_enc_att_linear = nn.Linear(self.d_h_tmp, args.d_h)
        self.edge_dec_input_linear = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)
        self.edge_dec_output_linear = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)

        self.vertex_dec_act_linear = nn.Linear(self.d_h_tmp, args.d_h, bias=False)
        self.edge_dec_act_linear = nn.Linear(self.d_h_tmp, args.d_h, bias=False)

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

        self.av_readout_b = nn.Parameter(torch.FloatTensor(V_CLASS_NUM).zero_())
        self.ae_readout_b = nn.Parameter(torch.FloatTensor(E_CLASS_NUM).zero_())

        # classifiers for "add vertex" operation and "add edge" operation
        self.av_readout = lambda q: F.linear(self.read_out_active(self.query_vec_to_av_vec(q)),
                                             self.vertex_class_embedding.weight, self.av_readout_b)
        self.ae_readout = lambda q: F.linear(self.read_out_active(self.query_vec_to_ae_vec(q)),
                                             self.edge_class_embedding.weight, self.ae_readout_b)
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
        v_instance_tensor, v_instance_lens, v_instance_classes, v_instance_s_ids, v_instance_names, \
        e_instance_tensor, e_instance_lens, e_instance_classes, e_instance_s_ids, e_instance_names, \
        gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
        gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
        gold_v_copy_labels, gold_e_copy_labels, gold_segment_switch_labels, data = batch

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

        # v_instance_pool:      LIST (bs), each element is a DICT {v_class_id: (total_v_num_class, d_h)}
        # e_instance_pool:      LIST (bs), each element is a DICT {e_class_id: (total_e_num_class, d_h)}
        v_instance_pool = instance_tensor_to_pool(v_instance_vec, v_instance_classes, v_instance_s_ids)
        e_instance_pool = instance_tensor_to_pool(e_instance_vec, e_instance_classes, e_instance_s_ids)

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

        q_pooling = max_pooling_by_lens(q_encodings, q_lens)
        q_v_pooling = max_pooling_by_mask(q_encodings, q_v_mask)
        q_e_pooling = max_pooling_by_mask(q_encodings, q_e_mask)

        # initialize state for decoder
        # dec_init_state:    (bs, d_h), (bs, d_h)
        dec_init_state = self.init_decoder_state(enc_context, self.aqg_decoder_cell_init)
        h_last = dec_init_state

        # zero embedding for empty AQG, (d_h)
        zero_graph_encoding = Variable(self.new_tensor(self.args.d_h).zero_())

        batch_size = len(q)
        max_op_num = max([len(x) for x in gold_graphs])     # maximum number of operations

        # aqg_scores:           loss for AQG generation
        # aqg_action_probs:     probabilities for each prediction (without softmax)
        aqg_scores = [[] for _ in range(batch_size)]
        aqg_action_probs = [[] for _ in range(batch_size)]

        # v_copy_scores:        loss for vertex copy mechanism
        # e_copy_scores:        loss for edge copy mechanism
        # switch_seg_scores:    loss for segment switch mechanism
        v_copy_scores = [[] for _ in range(batch_size)]
        e_copy_scores = [[] for _ in range(batch_size)]
        switch_seg_scores = [[] for _ in range(batch_size)]

        graph_encodings_history = [[] for _ in range(batch_size)]
        vertex_encodings_history = [[] for _ in range(batch_size)]
        edge_encodings_history = [[] for _ in range(batch_size)]

        #################################### Decoding for AQG generation ###############################################
        ################################################################################################################
        for t in range(max_op_num):

            # build encodings for AQG graph, vertex, edge at last timestep
            graph_encodings = []
            vertex_encodings = []
            edge_encodings = []

            for s_id in range(batch_size):
                assert len(gold_graphs[s_id]) == len(gold_aqg_obj_labels[s_id])

                if t < len(gold_graphs[s_id]):
                    v_tensor, v_class_tensor, v_segment_tensor, \
                    e_tensor, e_class_tensor, e_segment_tensor, adj = gold_graphs[s_id][t]

                    vertex_embed = self.vertex_class_embedding(v_class_tensor)      # Class vertex embedding
                    edge_embed = self.edge_class_embedding(e_class_tensor)          # Class edge embedding

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

                    # save graph history
                    graph_encodings_history[s_id].append(graph_encoding)
                    vertex_encodings_history[s_id].append(vertex_encoding)
                    edge_encodings_history[s_id].append(edge_encoding)

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
            (h_t, cell_t), ctx, att = self.decode_step(self.aqg_decoder_lstm,
                                                       self.aqg_enc_att_linear,
                                                       self.aqg_dec_input_linear,
                                                       h_last, q_encodings, dec_input_embeds,
                                                       src_token_mask=q_e_mask if self.args.use_mention_feature else q_mask,
                                                       return_att_weight=True)

            gold_aqgs_at_t = [gold_aqgs[s_id][t] if t < len(gold_aqgs[s_id]) else None for s_id in range(batch_size)]

            # calculate probabilities for each operation
            if t == 0 or t % 3 == 1:
                action_prob = self.av_readout(h_t)
                if self.args.mask_aqg_prob:
                    action_prob = mask_av_action_prob(self.args.dataset, action_prob, t, gold_aqgs_at_t, data)
            elif t % 3 == 0:
                action_prob = self.ae_readout(h_t)
                if self.args.mask_aqg_prob:
                    action_prob = mask_ae_action_prob(self.args.dataset, action_prob, gold_aqgs_at_t, data)
            else:
                # Cannot select the newly added point
                sv_mask = torch.cat([self.new_long_tensor(vertex_encodings.size(0), vertex_encodings.size(1) - 1).fill_(1),
                                     self.new_long_tensor(vertex_encodings.size(0), 1).zero_()], -1)
                action_prob = self.sv_pointer_net(src_encodings=vertex_encodings, query_vec=h_t.unsqueeze(0),
                                                  src_token_mask=sv_mask==0)

            # recording probs for AQG generation
            for s_id in range(batch_size):
                if t < len(gold_aqg_obj_labels[s_id]):
                    aqg_action_probs[s_id].append(action_prob[s_id])

            action_prob = F.softmax(action_prob, dim=-1)

            # save softmax prob for calculate loss
            for s_id in range(batch_size):
                if t < len(gold_aqg_obj_labels[s_id]):
                    act_prob_t_i = action_prob[s_id, gold_aqg_obj_labels[s_id][t]]
                    aqg_scores[s_id].append(act_prob_t_i)

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

            # calculate loss for vertex copy mechanism
            if self.args.use_v_copy:
                if t == 0 or t % 3 == 1:
                    not_v_copy_encoding_tmp = self.not_v_copy_encoding.unsqueeze(0).unsqueeze(0).expand(vertex_encodings.size(0), 1,
                                                                                                   self.not_v_copy_encoding.size(-1))
                    # combine embeddings of all vertices and embeddings of "do not copy vertex"
                    src_encodings = torch.cat([vertex_encodings, not_v_copy_encoding_tmp], dim=1)
                    v_copy_prob = self.vertex_copy_pointer_net(src_encodings=src_encodings, query_vec=h_t.unsqueeze(0),
                                                          src_token_mask=None)
                    v_copy_prob = F.softmax(v_copy_prob, dim=-1)
                    for s_id in range(batch_size):
                        if t < len(gold_aqg_obj_labels[s_id]) and gold_aqg_obj_labels[s_id][t] == V_CLASS_IDS["ent"]:
                            t_av = step_to_av_step(t)
                            # if label == -1, select the last one (embeddings of "do not copy vertex")
                            label = src_encodings.size(1) - 1 if gold_v_copy_labels[s_id][t_av] == -1 else gold_v_copy_labels[s_id][t_av]
                            cp_v_prob_t_i = v_copy_prob[s_id, label]
                            v_copy_scores[s_id].append(cp_v_prob_t_i)

            # calculate loss for edge copy mechanism
            if self.args.use_e_copy:
                if t != 0 and t % 3 == 0:
                    not_e_copy_encoding_tmp = self.not_e_copy_encoding.unsqueeze(0).unsqueeze(0).expand(edge_encodings.size(0), 1,
                                                                                                   self.not_e_copy_encoding.size(-1))
                    # combine embeddings of all edges and embeddings of "do not copy edge"
                    src_encodings = torch.cat([edge_encodings, not_e_copy_encoding_tmp], dim=1)
                    e_copy_prob = self.edge_copy_pointer_net(src_encodings=src_encodings, query_vec=h_t.unsqueeze(0),
                                                          src_token_mask=None)
                    e_copy_prob = F.softmax(e_copy_prob, dim=-1)
                    for s_id in range(batch_size):
                        if t < len(gold_aqg_obj_labels[s_id]) and (gold_aqg_obj_labels[s_id][t] == E_CLASS_IDS["rel+"]
                                                               or gold_aqg_obj_labels[s_id][t] == E_CLASS_IDS["rel-"]):
                            t_ae = step_to_ae_step(t)
                            # if label == -1, select the last one (embeddings of "do not copy edge")
                            label = src_encodings.size(1) - 1 if gold_e_copy_labels[s_id][t_ae] == -1 else gold_e_copy_labels[s_id][t_ae]
                            cp_e_prob_t_i = e_copy_prob[s_id, label]
                            e_copy_scores[s_id].append(cp_e_prob_t_i)

            h_last = (h_t, cell_t)

        # AQG generation loss
        score = torch.stack([torch.stack(score_i, dim=0).log().sum() for score_i in aqg_scores], dim=0)

        # Only CWQ has sub-query
        if self.args.dataset == "cwq":
            # AQG generation loss + segment switch loss
            switch_seg_score = torch.stack([torch.stack(score_i, dim=0).log().sum() for score_i in switch_seg_scores], dim=0)
            score = torch.add(score, switch_seg_score)

        # AQG generation loss + segment switch loss + vertex copy loss
        if self.args.use_v_copy:
            v_copy_score = torch.stack([torch.stack(score_i, dim=0).log().sum()
                                        if len(score_i) > 0 else Variable(self.new_tensor([0]).squeeze(0))
                                        for score_i in v_copy_scores ], dim=0)
            score = torch.add(score, v_copy_score)

        # AQG generation loss + segment switch loss + vertex copy loss + edge copy loss
        if self.args.use_e_copy:
            e_copy_score = torch.stack([torch.stack(score_i, dim=0).log().sum()
                                        if len(score_i) > 0 else Variable(self.new_tensor([0]).squeeze(0))
                                        for score_i in e_copy_scores if len(score_i) > 0], dim=0)
            score = torch.add(score, e_copy_score)

        ##################################### Decoding for vertex linking ##############################################
        ################################################################################################################
        h_last = self.init_decoder_state(enc_context, self.vertex_decoder_cell_init)

        # encoding of the final AQG, i.e., the completed AQG
        final_graph_encodings = torch.stack([x[-1] for x in graph_encodings_history])
        final_vertex_encodings = [x[-1] for x in vertex_encodings_history]
        final_edge_encodings = [x[-1] for x in edge_encodings_history]

        # zero_act_embed:               (d_h)
        # zero_vertex_encoding :        (d_h)
        # zero_v_instance_encoding:     (d_h)
        zero_act_embed = Variable(self.new_tensor(self.d_h_tmp).zero_())
        zero_vertex_encoding = Variable(self.new_tensor(self.args.d_h).zero_())
        zero_v_instance_encoding = Variable(self.new_tensor(self.d_h_tmp).zero_())

        # v_instance_scores:           loss for vertex linking
        # v_instance_action_probs:     probabilities for each prediction (without softmax)
        v_instance_scores = [[] for _ in range(batch_size)]
        v_instance_action_probs = [[] for _ in range(batch_size)]

        for t in range(max_op_num - 1):
            if t != 0 and t % 3 != 1:
                # only handle "add vertex"
                continue
            t_av = step_to_av_step(t)

            if t_av == 0:
                # the last action is None when the first "add vertex" operation
                act_last_embeds = Variable(self.new_tensor(batch_size, self.d_h_tmp).zero_(),
                                           requires_grad=False)
            else:
                # build the encoding of the last action
                act_last_embeds = []
                for s_id in range(batch_size):

                    if t_av - 1 < len(gold_v_instance_obj_labels[s_id]) and gold_v_instance_obj_labels[s_id][t_av - 1] != -1:
                        v_last_class = gold_aqg_obj_labels[s_id][t - 1] if t == 1 else gold_aqg_obj_labels[s_id][t - 3]
                        if v_last_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"], V_CLASS_IDS["end"]]:
                            # last vertex instance encoding
                            act_last_embed = v_instance_pool[s_id][v_last_class][gold_v_instance_obj_labels[s_id][t_av - 1]]
                        else:
                            act_last_embed = zero_act_embed
                    else:
                        act_last_embed = zero_act_embed
                    act_last_embeds.append(act_last_embed)

                act_last_embeds = torch.stack(act_last_embeds)

            # dec_input_embeds:     (bs, d_h)
            dec_input_embeds = self.vertex_dec_act_linear(act_last_embeds)
            # add the final AQG embeddings for predict the vertex instance (structural information)
            if self.args.use_graph_auxiliary_vector:
                dec_input_embeds = torch.add(dec_input_embeds, final_graph_encodings)

            # add the final vertex embeddings for predict the vertex instance (vertex class information)
            if self.args.use_vertex_auxiliary_encoding:
                final_vertex_encodings_tmp = []
                for s_id in range(batch_size):
                    if t_av < gold_aqgs[s_id][-1].vertex_number:
                        v_add_at_t = gold_aqgs[s_id][-1].get_v_add_history(t_av)
                        final_vertex_encodings_tmp.append(final_vertex_encodings[s_id][v_add_at_t])
                    else:
                        final_vertex_encodings_tmp.append(zero_vertex_encoding)
                final_vertex_encodings_tmp = torch.stack(final_vertex_encodings_tmp, dim=0)
                dec_input_embeds = torch.add(dec_input_embeds, final_vertex_encodings_tmp)

            # one step decoding for vertex linking
            # h_t:      (bs, d_h)
            # cell_t:   (bs, d_h)
            (h_t, cell_t), ctx, att = self.decode_step(self.vertex_decoder_lstm,
                                                       self.vertex_enc_att_linear,
                                                       self.vertex_dec_input_linear,
                                                       h_last, q_encodings, dec_input_embeds,
                                                       src_token_mask=q_v_mask if self.args.use_mention_feature else q_mask,
                                                       return_att_weight=True)
            if self.args.context_mode == "attention":
                # query_t:  (bs, d_h)
                query_t = self.vertex_dec_output_linear(torch.cat([h_t, ctx], 1))
            else:
                if self.args.use_mention_feature:
                    # query_t:  (bs, d_h)
                    query_t = self.vertex_dec_output_linear(torch.cat([h_t, q_v_pooling], 1))
                else:
                    # query_t:  (bs, d_h)
                    query_t = self.vertex_dec_output_linear(torch.cat([h_t, q_pooling], 1))

            # candidate vertex instance encodings
            src_encodings = []
            for s_id in range(batch_size):
                if t_av < len(gold_v_instance_obj_labels[s_id]) and gold_v_instance_obj_labels[s_id][t_av] != -1:
                    v_class = gold_aqg_obj_labels[s_id][t]
                    src_encodings.append(v_instance_pool[s_id][v_class])
                else:
                    # do not need to select vertex instance for "var" or "ans" vertex
                    src_encodings.append(zero_v_instance_encoding.unsqueeze(0))

            # src_encodings:        (bs, max_cand_v_num, d_h)
            # src_lens:             (bs)
            # src_mask:             (bs, max_cand_v_num)      0: True, 1: False
            src_encodings, src_lens = pad_tensor_1d(src_encodings, 0)
            src_mask = length_array_to_mask_tensor(src_lens)
            if self.args.cuda:
                src_encodings = src_encodings.to(self.args.gpu)
                src_mask = src_mask.to(self.args.gpu)

            # action_prob:          (bs, max_cand_v_num)
            action_prob = self.vertex_link_pointer_net(src_encodings=src_encodings, query_vec=query_t.unsqueeze(0),
                                                       src_token_mask=src_mask)

            # recording probs for vertex linking
            for s_id in range(batch_size):
                if t_av < len(gold_v_instance_obj_labels[s_id]) and gold_v_instance_obj_labels[s_id][t_av] != -1:
                    v_instance_action_probs[s_id].append(action_prob[s_id])
            action_prob = F.softmax(action_prob, dim=-1)

            for s_id in range(batch_size):
                if t_av < len(gold_v_instance_obj_labels[s_id]) and gold_v_instance_obj_labels[s_id][t_av] != -1:
                    act_prob_t_i = action_prob[s_id, gold_v_instance_obj_labels[s_id][t_av]]
                    v_instance_scores[s_id].append(act_prob_t_i)
            h_last = (h_t, cell_t)

        # loss of vertex linking
        v_instance_score = torch.stack([torch.stack(score_i, dim=0).log().sum() for score_i in v_instance_scores], dim=0)

        ####################################### Decoding for edge linking ##############################################
        ################################################################################################################
        h_last = self.init_decoder_state(enc_context, self.edge_decoder_cell_init)

        # zero_edge_encoding:           (d_h)
        # zero_e_instance_encoding:     (d_h)
        zero_edge_encoding = Variable(self.new_tensor(self.args.d_h).zero_())
        zero_e_instance_encoding = Variable(self.new_tensor(self.d_h_tmp).zero_())

        # e_instance_scores:           loss for edge linking
        # e_instance_action_probs:     probabilities for each prediction (without softmax)
        e_instance_scores = [[] for _ in range(batch_size)]
        e_instance_action_probs = [[] for _ in range(batch_size)]

        for t in range(max_op_num - 1):
            if t == 0 or t % 3 != 0:
                # Only handle "add edge"
                continue
            t_ae = step_to_ae_step(t)

            if t_ae == 0:
                # the last action is None when the first "add edge" operation
                act_last_embeds = Variable(self.new_tensor(batch_size, self.d_h_tmp).zero_(),
                                           requires_grad=False)
            else:
                # build the encoding of the last action
                act_last_embeds = []
                for s_id in range(batch_size):

                    if t_ae - 1 < len(gold_e_instance_obj_labels[s_id]):
                        # the timestep of the last predicted class: t - 3
                        e_last_class = gold_aqg_obj_labels[s_id][t - 3]
                        # Unified processing of forward and reverse relations
                        e_last_class = e_last_class - 1 if e_last_class % 2 == 1 else e_last_class
                        act_last_embed = e_instance_pool[s_id][e_last_class][gold_e_instance_obj_labels[s_id][t_ae - 1]]
                    else:
                        act_last_embed = zero_act_embed
                    act_last_embeds.append(act_last_embed)
                act_last_embeds = torch.stack(act_last_embeds)

            # dec_input_embeds:     (bs, d_h)
            dec_input_embeds = self.edge_dec_act_linear(act_last_embeds)

            # add the final AQG embeddings for predict the edge instance (structural information)
            if self.args.use_graph_auxiliary_vector:
                dec_input_embeds = torch.add(dec_input_embeds, final_graph_encodings)

            # add the final vertex embeddings for predict the edge instance (edge class information)
            if self.args.use_edge_auxiliary_encoding:
                final_edge_encodings_tmp = []
                for s_id in range(batch_size):
                    if t_ae < gold_aqgs[s_id][-1].edge_number // 2:
                        e_add_at_t = gold_aqgs[s_id][-1].get_e_add_history(t_ae)
                        final_edge_encodings_tmp.append(final_edge_encodings[s_id][e_add_at_t])
                    else:
                        final_edge_encodings_tmp.append(zero_edge_encoding)
                final_edge_encodings_tmp = torch.stack(final_edge_encodings_tmp, dim=0)
                dec_input_embeds = torch.add(dec_input_embeds, final_edge_encodings_tmp)

            # one step decoding for edge linking
            # h_t:      (bs, d_h)
            # cell_t:   (bs, d_h)
            (h_t, cell_t), ctx, att = self.decode_step(self.edge_decoder_lstm,
                                                       self.edge_enc_att_linear,
                                                       self.edge_dec_input_linear,
                                                       h_last, q_encodings, dec_input_embeds,
                                                       src_token_mask=q_e_mask if self.args.use_mention_feature else q_mask,
                                                       return_att_weight=True)

            if self.args.context_mode == "attention":
                # query_t:  (bs, d_h)
                query_t = self.edge_dec_output_linear(torch.cat([h_t, ctx], 1))
            else:
                if self.args.use_mention_feature:
                    # query_t:  (bs, d_h)
                    query_t = self.edge_dec_output_linear(torch.cat([h_t, q_e_pooling], 1))
                else:
                    # query_t:  (bs, d_h)
                    query_t = self.edge_dec_output_linear(torch.cat([h_t, q_pooling], 1))

            # candidate edge instance encodings
            src_encodings = []
            for s_id in range(batch_size):
                if t_ae < len(gold_e_instance_obj_labels[s_id]):
                    e_class = gold_aqg_obj_labels[s_id][t]
                    e_class = e_class - 1 if e_class % 2 == 1 else e_class
                    src_encodings.append(e_instance_pool[s_id][e_class])
                else:
                    src_encodings.append(zero_e_instance_encoding.unsqueeze(0))

            # src_encodings:        (bs, max_cand_e_num, d_h)
            # src_lens:             (bs)
            # src_mask:             (bs, max_cand_e_num)      0: True, 1: False
            src_encodings, src_lens = pad_tensor_1d(src_encodings, 0)
            src_mask = length_array_to_mask_tensor(src_lens)
            if self.args.cuda:
                src_encodings = src_encodings.to(self.args.gpu)
                src_mask = src_mask.to(self.args.gpu)

            # action_prob:          (bs, max_cand_e_num)
            action_prob = self.edge_link_pointer_net(src_encodings=src_encodings, query_vec=query_t.unsqueeze(0),
                                                     src_token_mask=src_mask)

            # recording probs for edge linking
            for s_id in range(batch_size):
                if t_ae < len(gold_e_instance_obj_labels[s_id]):
                    e_instance_action_probs[s_id].append(action_prob[s_id])
            action_prob = F.softmax(action_prob, dim=-1)

            for s_id in range(batch_size):
                if t_ae < len(gold_e_instance_obj_labels[s_id]):
                    act_prob_t_i = action_prob[s_id, gold_e_instance_obj_labels[s_id][t_ae]]
                    e_instance_scores[s_id].append(act_prob_t_i)
            h_last = (h_t, cell_t)

        # loss of edge linking
        e_instance_score = torch.stack([torch.stack(score_i, dim=0).log().sum() for score_i in e_instance_scores],
                                       dim=0)

        # combing loss
        score = torch.add(score, v_instance_score)
        score = torch.add(score, e_instance_score)

        return -score, aqg_action_probs, v_instance_action_probs, e_instance_action_probs

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
        v_instance_tensor, v_instance_lens, v_instance_classes, v_instance_s_ids, v_instance_names, \
        e_instance_tensor, e_instance_lens, e_instance_classes, e_instance_s_ids, e_instance_names, \
        gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
        gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
        gold_v_copy_labels, gold_e_copy_labels, gold_segment_switch_labels, data = sample

        data = data[0]

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

        q_pooling = max_pooling_by_lens(q_encodings, q_lens)
        q_v_pooling = max_pooling_by_mask(q_encodings, q_v_mask)
        q_e_pooling = max_pooling_by_mask(q_encodings, q_e_mask)

        # when testing, only handle one question at each batch
        assert len(q) == 1

        #################################### Decoding for AQG generation ###############################################
        ################################################################################################################
        # initialize state for decoder
        # dec_init_state:    (bs, d_h), (bs, d_h)
        dec_init_state = self.init_decoder_state(enc_context, self.aqg_decoder_cell_init)
        h_last = dec_init_state

        graph_encodings_histories = []
        vertex_encodings_histories = []
        edge_encodings_histories = []

        t = 0
        # initialize one empty AQG
        aqg = AbstractQueryGraph()
        aqg.init_state()

        aqg_beams = [aqg]                   # Initially, the beam set only consists of an empty AQG.
        aqg_completed_beams = []            # LIST, each element is (AQG, time_step, previous aqg id)

        # if the number of completed AQG is equal with beam size, BREAK
        # or if over than the predefined operation numbers, BREAK
        while len(aqg_completed_beams) < beam_size and t < self.args.max_num_op:

            # expand question encoding to match the current number of beams
            # exp_q_encodings:      (beam_num, max_q_len, d_h)
            exp_q_encodings = q_encodings.expand(len(aqg_beams), q_encodings.size(1), q_encodings.size(2))

            # build encodings for AQG graph, vertex, edge at last timestep
            graph_encodings = []
            vertex_encodings = []
            edge_encodings = []

            # only one sample (one question)
            graph_encodings_histories.append([])
            vertex_encodings_histories.append([])
            edge_encodings_histories.append([])

            for b_id, aqg in enumerate(aqg_beams):
                # get the state of the last AQG
                vertices, v_classes, v_segments, edges, e_classes, e_segments, triples = aqg.get_state()
                v_tensor, v_class_tensor, v_segment_tensor, \
                e_tensor, e_class_tensor, e_segment_tensor, adj = mk_graph_for_gnn(vertices, v_classes, v_segments,
                                                                                   edges, e_classes, e_segments,
                                                                                   triples)
                # move to GPU
                if self.args.cuda:
                    v_tensor = v_tensor.to(self.args.gpu)
                    e_tensor = e_tensor.to(self.args.gpu)
                    v_class_tensor = v_class_tensor.to(self.args.gpu)
                    e_class_tensor = e_class_tensor.to(self.args.gpu)
                    v_segment_tensor = v_segment_tensor.to(self.args.gpu)
                    e_segment_tensor = e_segment_tensor.to(self.args.gpu)
                    adj = adj.to(self.args.gpu)

                vertex_embed = self.vertex_class_embedding(v_class_tensor)  # Class vertex embedding
                edge_embed = self.edge_class_embedding(e_class_tensor)      # Class edge embedding

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

                # save graph history
                graph_encodings_histories[-1].append(graph_encoding)
                vertex_encodings_histories[-1].append(vertex_encoding)
                edge_encodings_histories[-1].append(edge_encoding)

                graph_encodings.append(graph_encoding)
                vertex_encodings.append(vertex_encoding)
                edge_encodings.append(edge_encoding)

            graph_encodings = torch.stack(graph_encodings)
            vertex_encodings = torch.stack(vertex_encodings)
            edge_encodings, edge_nums = pad_tensor_1d(edge_encodings, 0)
            if self.args.cuda:
                edge_encodings = edge_encodings.to(self.args.gpu)

            if self.args.use_graph_encoder:
                # dec_input_embeds:     (beam_num, d_h)
                dec_input_embeds = graph_encodings
            else:
                dec_input_embeds = h_last[0]

            # one step decoding
            # h_t:      (beam_num, d_h)
            # cell_t:   (beam_num, d_h)
            (h_t, cell_t), ctx, att = self.decode_step(self.aqg_decoder_lstm,
                                                       self.aqg_enc_att_linear,
                                                       self.aqg_dec_input_linear,
                                                       h_last, exp_q_encodings, dec_input_embeds,
                                                       src_token_mask=q_e_mask if self.args.use_mention_feature else q_mask,
                                                       return_att_weight=True)

            ####################################### add vertex operation ###############################################
            if t == 0 or t % 3 == 1:
                op = 'av'
                # action_prob:      (beam_num, V_CLASS_NUM), probs of add vertex operation
                # switch_seg_prob:  (beam_num, 2), probs of segment switching
                # v_copy_prob:      (beam_num, v_num_at_t + 1), probs of whether copy vertex
                action_prob = self.av_readout(h_t)
                if self.args.mask_aqg_prob:
                    action_prob = mask_av_action_prob(self.args.dataset, action_prob, t,
                                                      [aqg for _ in range(action_prob.size(0))],
                                                      [data for _ in range(action_prob.size(0))])
                action_prob = F.log_softmax(action_prob, dim=-1)

                switch_seg_prob = F.log_softmax(self.seg_readout(h_t), dim=-1)
                not_v_copy_encoding_tmp = self.not_v_copy_encoding.unsqueeze(0).unsqueeze(0).expand(vertex_encodings.size(0),
                                                                                               1,
                                                                                               self.not_v_copy_encoding.size(-1))
                # combing the encodings of all vertices and the encoding of "do not copy vertex"
                src_encodings = torch.cat([vertex_encodings, not_v_copy_encoding_tmp], dim=1)
                v_copy_prob = self.vertex_copy_pointer_net(src_encodings=src_encodings, query_vec=h_t.unsqueeze(0),
                                                      src_token_mask=None)
                v_copy_prob = F.log_softmax(v_copy_prob, dim=-1)

                # save the possible directions for AQG expansion
                new_aqg_meta = []
                for b_id, aqg in enumerate(aqg_beams):
                    if t == 0:
                        # first vertex is always in the class of "ans"
                        meta_entry = {
                            "op": op,
                            "obj": V_CLASS_IDS["ans"],
                            "obj_score": action_prob[b_id, V_CLASS_IDS["ans"]],
                            "seg": 0,
                            "seg_score": None,
                            "cp_v": -1,
                            "cp_v_score": None,
                            "cp_e": -1,
                            "cp_e_score": None,
                            "new_aqg_score": aqg.get_score() + action_prob[b_id, V_CLASS_IDS["ans"]].cpu().detach().numpy(),
                            "prev_aqg_id": b_id
                        }
                        new_aqg_meta.append(meta_entry)
                        continue

                    # enumerate add vertex object
                    o_range = [i for i in range(len(V_CLASS_IDS))]
                    for o_id in o_range:
                        # Except for the first vertex, there will be no "ans" vertex.
                        if o_id == V_CLASS_IDS["ans"]:
                            continue
                        # The class o_id must have an instance.
                        if o_id not in [V_CLASS_IDS["end"], V_CLASS_IDS["var"]] and o_id not in data["instance_pool"]["vertex"]:
                            continue
                        # LC-QuAD dataset does not have "val" vertices.
                        if o_id == V_CLASS_IDS["val"] and self.args.dataset == "lcq":
                            continue

                        # update probability
                        new_aqg_score = aqg.get_score() + action_prob[b_id, o_id].cpu().detach().numpy()

                        # end signal for AQG generation
                        if o_id == V_CLASS_IDS["end"]:
                            meta_entry = {
                                "op": op,
                                "obj": o_id,
                                "obj_score": action_prob[b_id, o_id],
                                "seg": 0,
                                "seg_score": None,
                                "cp_v": -1,
                                "cp_v_score": None,
                                "cp_e": -1,
                                "cp_e_score": None,
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

                            # enumerate the copied vertex, only when the class is entity
                            if self.args.use_v_copy and o_id == V_CLASS_IDS["ent"]:
                                v_copy_range = [j for j in range(src_encodings.size(1))]
                            else:
                                # only the last encoding, denoting "do not copy vertex"
                                v_copy_range = [src_encodings.size(1) - 1]

                            for cp_v_id in v_copy_range:
                                if cp_v_id != src_encodings.size(1) - 1 and aqg.get_vertex_label(cp_v_id) != V_CLASS_IDS["ent"]:
                                    continue

                                # update probability by adding vertex copying probability
                                if self.args.use_v_copy:
                                    new_aqg_score_2 = new_aqg_score_1 + v_copy_prob[b_id, cp_v_id].cpu().detach().numpy()
                                else:
                                    new_aqg_score_2 = new_aqg_score_1

                                meta_entry = {
                                    "op": op,
                                    "obj": o_id,
                                    "obj_score": action_prob[b_id, o_id],
                                    "seg": seg_id,
                                    "seg_score": switch_seg_prob[b_id, seg_id],
                                    "cp_v": cp_v_id if cp_v_id != src_encodings.size(1) - 1 else -1,
                                    "cp_v_score": v_copy_prob[b_id, cp_v_id],
                                    "cp_e": -1,
                                    "cp_e_score": None,
                                    "new_aqg_score": new_aqg_score_2,
                                    "prev_aqg_id": b_id
                                }
                                new_aqg_meta.append(meta_entry)

            ######################################### add edge operation ###############################################
            elif t % 3 == 0:
                op = "ae"
                # action_prob:      (beam_num, E_CLASS_NUM), probs of add edge operation
                # e_copy_prob:      (beam_num, e_num_at_t + 1), probs of whether copy vertex
                action_prob = self.ae_readout(h_t)
                if self.args.mask_aqg_prob:
                    action_prob = mask_ae_action_prob(self.args.dataset, action_prob,
                                                      [aqg for _ in range(action_prob.size(0))],
                                                      [data for _ in range(action_prob.size(0))])
                action_prob = F.log_softmax(action_prob, dim=-1)
                not_e_copy_encoding_tmp = self.not_e_copy_encoding.unsqueeze(0).unsqueeze(0).expand(edge_encodings.size(0),
                                                                                                    1,
                                                                                                    self.not_e_copy_encoding.size(-1))
                # combing the encodings of all edges and the encoding of "do not copy edge"
                src_encodings = torch.cat([edge_encodings, not_e_copy_encoding_tmp], dim=1)
                e_copy_prob = self.edge_copy_pointer_net(src_encodings=src_encodings, query_vec=h_t.unsqueeze(0),
                                                      src_token_mask=None)
                e_copy_prob = F.log_softmax(e_copy_prob, dim=-1)

                new_aqg_meta = []
                for b_id, aqg in enumerate(aqg_beams):
                    # Enumerate add edge object
                    o_range = [i for i in range(len(E_CLASS_IDS))]
                    for o_id in o_range:
                        # if "+" direction class does not have instances, skip
                        if o_id % 2 == 0 and o_id not in data["instance_pool"]["edge"] and o_id + 1 not in data["instance_pool"]["edge"]:
                            continue
                        # if "-" direction class does not have instances, skip
                        if o_id % 2 == 1 and o_id not in data["instance_pool"]["edge"] and o_id - 1 not in data["instance_pool"]["edge"]:
                            continue

                        # LC-QuAD dataset does not have "cmp" and "ord" edges.
                        if self.args.dataset == "lcq":
                            if o_id in [E_CLASS_IDS["cmp+"], E_CLASS_IDS["cmp-"], E_CLASS_IDS["ord+"], E_CLASS_IDS["ord-"]]:
                                continue

                        # update probability
                        new_aqg_score = aqg.get_score() + action_prob[b_id, o_id].cpu().detach().numpy()

                        # enumerate the copy edge
                        if self.args.use_e_copy and (o_id == E_CLASS_IDS["rel+"] or o_id == E_CLASS_IDS["rel-"]):
                            e_copy_range = [j for j in range(src_encodings.size(1) - 1) if j < edge_nums[b_id]] + [src_encodings.size(1) - 1]
                        else:
                            # only the last encoding, denoting "do not copy edge"
                            e_copy_range = [src_encodings.size(1) - 1]

                        for cp_e_id in e_copy_range:
                            # update probability by adding edge copying probability
                            if self.args.use_e_copy:
                                new_aqg_score_1 = new_aqg_score + e_copy_prob[b_id, cp_e_id].cpu().detach().numpy()
                            else:
                                new_aqg_score_1 = new_aqg_score

                            meta_entry = {
                                "op": op,
                                "obj": o_id,
                                "obj_score": action_prob[b_id, o_id],
                                "seg": 0,
                                "seg_score": None,
                                "cp_v": -1,
                                "cp_v_score": None,
                                "cp_e": cp_e_id if cp_e_id != src_encodings.size(1) - 1 else -1,
                                "cp_e_score": e_copy_prob[b_id, cp_e_id],
                                "new_aqg_score": new_aqg_score_1,
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
                for b_id, aqg in enumerate(aqg_beams):
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
                            "cp_v": -1,
                            "cp_v_score": None,
                            "cp_e": -1,
                            "cp_e_score": None,
                            "new_aqg_score": new_aqg_score,
                            "prev_aqg_id": b_id
                        }
                        new_aqg_meta.append(meta_entry)

            if not new_aqg_meta:
                break

            # new_aqg_scores:       (beam_num)
            new_aqg_scores = self.new_tensor([x["new_aqg_score"] for x in new_aqg_meta])
            # select top-k aqg with highest probs
            k = min(new_aqg_scores.size(0), beam_size - len(aqg_completed_beams))
            top_new_aqg_scores, meta_ids = torch.topk(new_aqg_scores, k=len(new_aqg_scores))

            live_aqg_ids = []
            new_aqg_beams = []
            cnt = 0
            for new_aqg_score, meta_id in zip(top_new_aqg_scores.cpu().detach().numpy(), meta_ids.data.cpu()):
                if cnt >= k:
                    break
                aqg_meta_entry = new_aqg_meta[meta_id]
                op = aqg_meta_entry["op"]
                obj = aqg_meta_entry["obj"]
                prev_aqg_id = aqg_meta_entry["prev_aqg_id"]
                prev_aqg = aqg_beams[prev_aqg_id]

                # build new AQG
                new_aqg = copy.deepcopy(prev_aqg)
                new_aqg.update_score(new_aqg_score)

                if op == "av" and obj == V_CLASS_IDS["end"]:
                    # generation is end

                    if self.args.dataset == "lcq":
                        if new_aqg.check_final_structure(data["instance_pool"], self.args.dataset):
                            aqg_completed_beams.append([new_aqg, prev_aqg_id, t])
                    else:
                        aqg_completed_beams.append([new_aqg, prev_aqg_id, t])
                else:
                    # update AQG state
                    if op == "av":
                        v_copy = aqg_meta_entry["cp_v"]
                        switch_segment = aqg_meta_entry["seg"]
                        new_aqg.update_state("av", [obj, v_copy, switch_segment])
                    elif op == "ae":
                        e_copy = aqg_meta_entry["cp_e"]
                        new_aqg.update_state("ae", [obj, e_copy])
                    else:
                        new_aqg.update_state("sv", obj)

                    if self.args.dataset == "lcq" and not new_aqg.check_temporary_structure(self.args.dataset):
                        continue

                    new_aqg_beams.append(new_aqg)
                    live_aqg_ids.append(prev_aqg_id)
                cnt += 1

            if not live_aqg_ids:
                break

            # print()
            # print("######################################################################################")
            # print("Timestep: {}, Operation: {} ".format(t, op))
            # for _aqg in new_aqg_beams:
            #     _aqg.show_state()

            h_last = (h_t[live_aqg_ids], cell_t[live_aqg_ids])
            aqg_beams = new_aqg_beams

            t += 1

        # sort by total probability
        aqg_completed_beams.sort(key=lambda x: -x[0].get_score())
        # do not complete any AQG
        if len(aqg_completed_beams) == 0:
            return []

        # print()
        # print("===============================================================================================")
        # print("Complete Beams:")
        # for x, _, _ in aqg_completed_beams:
        #     x.show_state()

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

        # v_instance_pool:      LIST (bs), each element is a DICT {v_class_id: (total_v_num_class, d_h)}
        v_instance_pool = instance_tensor_to_pool(v_instance_vec, v_instance_classes, v_instance_s_ids)[0]

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

        # e_instance_pool:      LIST (bs), each element is a DICT {e_class_id: (total_e_num_class, d_h)}
        e_instance_pool = instance_tensor_to_pool(e_instance_vec, e_instance_classes, e_instance_s_ids)[0]

        if self.args.dataset in ["lcq", "wsp"] and self.args.use_matching_score:
            # Record each edge instance whether matching the question
            e_instance_literal_score = {}
            for i, (e_class, e_name, e_true_name) in enumerate(e_instance_names[0]):
                if e_class in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
                    if self.dataset == "lcq":
                        e_toks = tokenize_word_sentence(e_true_name)
                    else:
                        e_toks = tokenize_word_sentence(get_relation_last_name(e_name, kb="freebase"))
                    match_mode, match_len = cal_literal_matching_score(self.args.dataset,
                                                                       data["question_toks"],
                                                                       e_toks)
                    e_instance_literal_score[e_name] = match_len
                    # print(e_name, match_mode, match_len)
                else:
                    e_instance_literal_score[e_name] = 0
        else:
            e_instance_literal_score = {e_name: 0 for i, (e_class, e_name, e_true_nam) in enumerate(e_instance_names[0])}

        for pred_aqg, pred_aqg_id, final_t in aqg_completed_beams:
            v_h_last = self.init_decoder_state(enc_context, self.vertex_decoder_cell_init)

            # print("====================  CURRENT AQG:")
            # pred_aqg.show_state()

            # Final AQG encodings
            final_graph_encodings = graph_encodings_histories[final_t][pred_aqg_id]
            final_vertex_encodings = vertex_encodings_histories[final_t][pred_aqg_id]
            final_edge_encodings = edge_encodings_histories[final_t][pred_aqg_id]

            v_completed_beams = self.generate_vertex_instance(beam_size, pred_aqg, final_graph_encodings,
                                                              final_vertex_encodings,
                                                              q_encodings, q_pooling, q_v_pooling, q_mask, q_v_mask,
                                                              v_instance_pool, v_h_last, data)

            for v_pred_aqg in v_completed_beams:

                # print("====================  CURRENT V AQG:")
                # v_pred_aqg.show_state()

                e_h_last = self.init_decoder_state(enc_context, self.edge_decoder_cell_init)

                e_completed_beams = self.generate_edge_instance(beam_size, v_pred_aqg, final_graph_encodings,
                                                                final_edge_encodings,
                                                                q_encodings, q_pooling, q_e_pooling, q_mask, q_e_mask,
                                                                e_instance_pool, e_instance_literal_score,
                                                                e_h_last, data,
                                                                sparql_cache=sparql_cache)
                if e_completed_beams:
                    return e_completed_beams

        return []

    #################################### Decoding for vertex linking ###############################################
    ################################################################################################################
    def generate_vertex_instance(self, beam_size, pred_aqg, final_graph_encodings, final_vertex_encodings,
                                 q_encodings, q_pooling, q_v_pooling, q_mask, q_v_mask, v_instance_pool, h_last, data):

        zero_act_embed = Variable(self.new_tensor(self.d_h_tmp).zero_())
        zero_v_instance_encoding = Variable(self.new_tensor(self.d_h_tmp).zero_())

        t = 1
        beams = [pred_aqg]  # Initially, the beam set only consists of the predicted AQG without any instance
        completed_beams = []  # LIST, each element is AQG

        pred_aqg_obj_labels = [x for x in pred_aqg.pred_obj_labels]
        pred_v_copy_labels = [x for x in pred_aqg.pred_v_copy_labels]

        while len(completed_beams) < beam_size and t < len(pred_aqg_obj_labels) - 1:

            # expand question encoding to match the current number of beams
            # exp_q_encodings:      (beam_num, max_q_len, d_h)
            exp_q_encodings = q_encodings.expand(len(beams), q_encodings.size(1), q_encodings.size(2))
            exp_q_pooling = q_pooling.expand(len(beams), q_pooling.size(1))
            exp_q_v_pooling = q_v_pooling.expand(len(beams), q_v_pooling.size(1))

            if t % 3 != 1:
                # only handle "add vertex"
                t += 1
                continue
            t_av = step_to_av_step(t)

            if t == 1:
                # the last action is None when the first "add vertex" operation
                act_last_embeds = Variable(self.new_tensor(len(beams), self.d_h_tmp).zero_(),
                                           requires_grad=False)
            else:
                # build the encoding of the last action
                act_last_embeds = []
                for b_id, aqg in enumerate(beams):
                    if aqg.pred_v_instance_labels:
                        v_last_class, act_last_id = aqg.pred_v_instance_labels[-1]
                        if v_last_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"], V_CLASS_IDS["end"]]:
                            # last vertex instance encoding
                            act_last_embed = v_instance_pool[v_last_class][act_last_id]
                        else:
                            act_last_embed = zero_act_embed
                    else:
                        act_last_embed = zero_act_embed

                    act_last_embeds.append(act_last_embed)
                act_last_embeds = torch.stack(act_last_embeds)

            # dec_input_embeds:     (beam_num, d_h)
            dec_input_embeds = self.vertex_dec_act_linear(act_last_embeds)

            # add the final AQG embeddings for predict the vertex instance (structural information)
            if self.args.use_graph_auxiliary_vector:
                dec_input_embeds = torch.add(dec_input_embeds, final_graph_encodings)

            # add the final vertex embeddings for predict the vertex instance (vertex class information)
            if self.args.use_vertex_auxiliary_encoding:
                final_vertex_encodings_tmp = []
                for b_id, aqg in enumerate(beams):
                    v_add_at_t = aqg.get_v_add_history(t_av)
                    final_vertex_encodings_tmp.append(final_vertex_encodings[v_add_at_t])
                final_vertex_encodings_tmp = torch.stack(final_vertex_encodings_tmp, dim=0)
                dec_input_embeds = torch.add(dec_input_embeds, final_vertex_encodings_tmp)

            # one step decoding for vertex linking
            # h_t:      (beam_num, d_h)
            # cell_t:   (beam_num, d_h)
            (h_t, cell_t), ctx, att = self.decode_step(self.vertex_decoder_lstm,
                                                       self.vertex_enc_att_linear,
                                                       self.vertex_dec_input_linear,
                                                       h_last, exp_q_encodings, dec_input_embeds,
                                                       src_token_mask=q_v_mask if self.args.use_mention_feature else q_mask,
                                                       return_att_weight=True)

            if self.args.context_mode == "attention":
                # query_t:  (beam_num, d_h)
                query_t = self.vertex_dec_output_linear(torch.cat([h_t, ctx], 1))
            else:
                if self.args.use_mention_feature:
                    # query_t:  (beam_num, d_h)
                    query_t = self.vertex_dec_output_linear(torch.cat([h_t, exp_q_v_pooling], 1))
                else:
                    # query_t:  (beam_num, d_h)
                    query_t = self.vertex_dec_output_linear(torch.cat([h_t, exp_q_pooling], 1))

            # candidate vertex instance encodings
            src_encodings = []
            for b_id, aqg in enumerate(beams):
                pred_v_class = pred_aqg_obj_labels[t]
                if pred_v_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"], V_CLASS_IDS["end"]]:
                    src_encodings.append(v_instance_pool[pred_v_class])
                else:
                    # do not need to select vertex instance for "var" or "ans" vertex
                    src_encodings.append(zero_v_instance_encoding.unsqueeze(0))

            # src_encodings:        (beam_num, max_cand_v_num, d_h)
            # src_lens:             (beam_num)
            # src_mask:             (beam_num, max_cand_v_num)      0: True, 1: False
            src_encodings, src_lens = pad_tensor_1d(src_encodings, 0)
            src_mask = length_array_to_mask_tensor(src_lens)
            for b_id, aqg in enumerate(beams):
                pred_v_class = pred_aqg_obj_labels[t]
                pred_v_copy = pred_v_copy_labels[t_av]
                if pred_v_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"], V_CLASS_IDS["end"]]:

                    if self.args.use_v_copy:
                        if pred_v_class == V_CLASS_IDS["ent"] and pred_v_copy != -1:
                            for i in range(len(v_instance_pool[pred_v_class])):
                                # if use copy mechanism, mask all other instance except for the predicted vertex
                                try:
                                    if i != aqg.get_vertex_instance(pred_v_copy)[0]:
                                        src_mask[b_id][i] = 1
                                except:
                                    print(data["id"])
                                    print(t)
                                    print(t_av)
                                    print(pred_v_class)
                                    print(pred_v_copy)
                                    print(pred_aqg_obj_labels)
                                    print(pred_v_copy_labels)
                                    aqg.show_state()
                                    exit()
                        else:
                            # mask vertex instances that have been used
                            for i in range(len(v_instance_pool[pred_v_class])):
                                for _v_class, _o_id in aqg.pred_v_instance_labels:
                                    if _v_class == pred_v_class and _o_id == i:
                                        src_mask[b_id][i] = 1
                                        break
            if self.args.cuda:
                src_encodings = src_encodings.to(self.args.gpu)
                src_mask = src_mask.to(self.args.gpu)

            # action_prob:          (beam_num, max_cand_v_num)
            action_prob = self.vertex_link_pointer_net(src_encodings=src_encodings, query_vec=query_t.unsqueeze(0),
                                                       src_token_mask=src_mask)
            action_prob = F.log_softmax(action_prob, dim=-1)

            new_aqg_meta = []  # save the possible directions for AQG expansion
            live_aqg_ids = []
            new_beams = []

            for b_id, aqg in enumerate(beams):
                pred_v_class = pred_aqg_obj_labels[t]

                if pred_v_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"], V_CLASS_IDS["end"]]:
                    # enumerate the vertex instance
                    for o_id in range(len(v_instance_pool[pred_v_class])):
                        if src_mask[b_id][o_id] == 1:
                            continue

                        # update probability
                        new_v_score = aqg.get_v_score() + action_prob[b_id, o_id].cpu().detach().numpy()
                        # get the index of added vertex at timestep t.
                        v_id = aqg.get_v_add_history(t_av)
                        meta_entry = {
                            "vertex": v_id,
                            "obj": o_id,
                            "obj_score": action_prob[b_id, o_id],
                            "new_v_score": new_v_score,
                            "prev_aqg_id": b_id
                        }
                        new_aqg_meta.append(meta_entry)
                else:
                    new_beams.append(aqg)
                    live_aqg_ids.append(b_id)

            if not new_aqg_meta and not new_beams:
                break

            # new_v_scores:       (beam_num)
            new_v_scores = self.new_tensor([x['new_v_score'] for x in new_aqg_meta])
            # select top-k beams with highest probs
            k = min(new_v_scores.size(0), beam_size - len(completed_beams))
            top_new_v_scores, meta_ids = torch.topk(new_v_scores, k=k)

            for _, meta_id in zip(top_new_v_scores.data.cpu(), meta_ids.data.cpu()):
                aqg_meta_entry = new_aqg_meta[meta_id]
                v = aqg_meta_entry["vertex"]
                obj = aqg_meta_entry["obj"]
                new_v_score = aqg_meta_entry["new_v_score"]
                prev_aqg_id = aqg_meta_entry['prev_aqg_id']
                prev_aqg = beams[prev_aqg_id]

                # build new AQG
                new_aqg = copy.deepcopy(prev_aqg)
                new_aqg.update_v_score(new_v_score)

                v_class = pred_aqg_obj_labels[t]
                v_instance_name = data["instance_pool"]["vertex"][v_class][obj]
                # set the predicted instance to vertex v
                new_aqg.set_vertex_instance(v, [obj, v_instance_name[0]])
                new_aqg.pred_v_instance_labels.append([v_class, obj])

                new_beams.append(new_aqg)
                live_aqg_ids.append(prev_aqg_id)

            # print()
            # print("######################################################################################")
            # print("Timestep: {}, Operation: {} ".format(t, "av"))
            # for _aqg in new_beams:
            #     _aqg.show_state()

            if not live_aqg_ids:
                break

            h_last = (h_t[live_aqg_ids], cell_t[live_aqg_ids])
            beams = new_beams
            t += 1

        for aqg in beams:
            validity = True
            for v in aqg.vertices:
                if aqg.v_labels[v] in [V_CLASS_IDS["ans"], V_CLASS_IDS["var"]]:
                    continue
                if v not in aqg.v_instances:
                    validity = False
                    break
            if validity:
                completed_beams.append(aqg)
        completed_beams.sort(key=lambda x: -x.get_v_score())

        # print()
        # print("===============================================================================================")
        # print("Complete Vertex Beams:")
        # for x in completed_beams:
        #     x.show_state()
        # exit()

        return completed_beams

    ###################################### Decoding for edge linking ###############################################
    ################################################################################################################
    def generate_edge_instance(self, beam_size, pred_aqg, final_graph_encodings, final_edge_encodings,
                                 q_encodings, q_pooling, q_e_pooling, q_mask, q_e_mask, e_instance_pool,
                               e_instance_literal_score, h_last, data, sparql_cache=None):

        t = 3
        beams = [pred_aqg]      # Initially, the beam set only consists of the predicted AQG without any edge instance
        completed_beams = []    # LIST, each element is an AQG
        pred_aqg_obj_labels = [x for x in pred_aqg.pred_obj_labels]
        pred_e_copy_labels = [x for x in pred_aqg.pred_e_copy_labels]

        while len(completed_beams) < beam_size and t < len(pred_aqg_obj_labels):

            # expand question encoding to match the current number of beams
            # exp_q_encodings:      (beam_num, max_q_len, d_h)
            exp_q_encodings = q_encodings.expand(len(beams), q_encodings.size(1), q_encodings.size(2))
            exp_q_pooling = q_pooling.expand(len(beams), q_pooling.size(1))
            exp_q_e_pooling = q_e_pooling.expand(len(beams), q_e_pooling.size(1))

            if t % 3 != 0:
                t += 1
                # only handle "add edge"
                continue
            t_ae = step_to_ae_step(t)

            if t_ae == 0:
                # the last action is None when the first "add edge" operation
                act_last_embeds = Variable(self.new_tensor(len(beams), self.d_h_tmp).zero_(),
                                           requires_grad=False)
            else:
                # build the encoding of the last action
                act_last_embeds = []
                for b_id, aqg in enumerate(beams):
                    e_last_class, act_last_id = aqg.pred_e_instance_labels[-1]
                    # only use the class of even id, denoting "+" direction
                    e_last_class = e_last_class - 1 if e_last_class % 2 == 1 else e_last_class
                    # last edge instance encoding
                    act_last_embed = e_instance_pool[e_last_class][act_last_id]
                    act_last_embeds.append(act_last_embed)

                act_last_embeds = torch.stack(act_last_embeds)

            # dec_input_embeds:     (beam_num, d_h)
            dec_input_embeds = self.edge_dec_act_linear(act_last_embeds)

            # add the final AQG embeddings for predict the edge instance (structural information)
            if self.args.use_graph_auxiliary_vector:
                dec_input_embeds = torch.add(dec_input_embeds, final_graph_encodings)

            # add the final vertex embeddings for predict the vertex instance (edge class information)
            if self.args.use_edge_auxiliary_encoding:
                final_edge_encodings_tmp = []
                for b_id, aqg in enumerate(beams):
                    e_add_at_t = aqg.get_e_add_history(t_ae)
                    final_edge_encodings_tmp.append(final_edge_encodings[e_add_at_t])
                final_edge_encodings_tmp = torch.stack(final_edge_encodings_tmp, dim=0)
                dec_input_embeds = torch.add(dec_input_embeds, final_edge_encodings_tmp)

            # one step decoding for edge linking
            # h_t:      (beam_num, d_h)
            # cell_t:   (beam_num, d_h)
            (h_t, cell_t), ctx, att = self.decode_step(self.edge_decoder_lstm,
                                                       self.edge_enc_att_linear,
                                                       self.edge_dec_input_linear,
                                                       h_last, exp_q_encodings, dec_input_embeds,
                                                       src_token_mask=q_e_mask if self.args.use_mention_feature else q_mask,
                                                       return_att_weight=True)

            if self.args.context_mode == "attention":
                # query_t:  (beam_num, d_h)
                query_t = self.edge_dec_output_linear(torch.cat([h_t, ctx], 1))
            else:
                if self.args.use_mention_feature:
                    # query_t:  (beam_num, d_h)
                    query_t = self.edge_dec_output_linear(torch.cat([h_t, exp_q_e_pooling], 1))
                else:
                    # query_t:  (beam_num, d_h)
                    query_t = self.edge_dec_output_linear(torch.cat([h_t, exp_q_pooling], 1))

            # candidate edge instance encodings
            src_encodings = []
            for b_id, aqg in enumerate(beams):
                pred_e_class = pred_aqg_obj_labels[t]
                pred_e_class = pred_e_class - 1 if pred_e_class % 2 == 1 else pred_e_class
                src_encodings.append(e_instance_pool[pred_e_class])

            # src_encodings:        (beam_num, max_cand_e_num, d_h)
            # src_lens:             (beam_num)
            # src_mask:             (beam_num, max_cand_e_num)      0: True, 1: False
            src_encodings, src_lens = pad_tensor_1d(src_encodings, 0)
            src_mask = length_array_to_mask_tensor(src_lens)

            for b_id, aqg in enumerate(beams):
                pred_e_class = pred_aqg_obj_labels[t]
                pred_e_class = pred_e_class - 1 if pred_e_class % 2 == 1 else pred_e_class
                pred_e_copy = pred_e_copy_labels[t_ae]
                e_id = aqg.get_e_add_history(t_ae)

                if self.args.use_e_copy:
                    if pred_e_copy != -1:
                        for i in range(len(e_instance_pool[pred_e_class])):
                            # if use copy mechanism, mask all other instance except for the predicted edge
                            if i != aqg.get_edge_instance(pred_e_copy)[0]:
                                src_mask[b_id][i] = 1
                    else:
                        # mask edge instances that have been used
                        for i in range(len(e_instance_pool[pred_e_class])):
                            for _e_class, _o_id in aqg.pred_e_instance_labels:
                                if _e_class == pred_e_class and _o_id == i:
                                    src_mask[b_id][i] = 1
                                    break

                # mask edge instances using KB constraint
                if pred_e_class in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
                    # make constraint for <var, rdf:type, type>
                    if self.args.dataset == "lcq":
                        now_s, now_o = -1, -1
                        for j, triple in enumerate(aqg.triples):
                            s, o, p = triple
                            if p == e_id:
                                now_s, now_o = s, o
                                break
                        assert now_s != -1 and now_o != -1
                        now_s_class = aqg.get_vertex_label(now_s)
                        now_o_class = aqg.get_vertex_label(now_o)
                        if now_s_class == V_CLASS_IDS["type"] or now_o_class == V_CLASS_IDS["type"]:
                            for i, (e_instance_name, _) in enumerate(data["instance_pool"]["edge"][pred_e_class]):
                                if e_instance_name == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                                    src_mask[b_id][i] = 0
                                else:
                                    src_mask[b_id][i] = 1
                        else:
                            for i, (e_instance_name, _) in enumerate(data["instance_pool"]["edge"][pred_e_class]):
                                if e_instance_name == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                                    src_mask[b_id][i] = 1
                                else:
                                    src_mask[b_id][i] = 0

                    # make constraint for <var1, rel, var2> <var2, cmp, date>
                    if self.args.dataset in ["wsp", "cwq"]:
                        date_vars = set()
                        for j, triple in enumerate(aqg.triples):
                            s, o, p = triple
                            _p_class = aqg.get_edge_label(p)
                            if _p_class not in [E_CLASS_IDS["cmp+"], E_CLASS_IDS["cmp-"]]:
                                continue
                            _s_class = aqg.get_vertex_label(s)
                            _o_class = aqg.get_vertex_label(o)
                            if _s_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] and _o_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]]:
                                date_vars.add(s)
                                date_vars.add(o)
                            else:
                                if _s_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]]:
                                    _o_name = aqg.get_vertex_instance(o)[-1]
                                    if "xsd:dateTime" in _o_name:
                                        date_vars.add(s)
                                if _o_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]]:
                                    _s_name = aqg.get_vertex_instance(s)[-1]
                                    if "xsd:dateTime" in _s_name:
                                        date_vars.add(o)

                        now_s, now_o = -1, -1
                        for j, triple in enumerate(aqg.triples):
                            s, o, p = triple
                            if p == e_id:
                                now_s, now_o = s, o
                                break

                        assert now_s != -1 and now_o != -1
                        if now_s in date_vars or now_o in date_vars:
                            for i, (e_instance_name, _) in enumerate(data["instance_pool"]["edge"][pred_e_class]):
                                if e_instance_name.split(".")[-1] in ["from", "to", "from$$$to", "start_date",
                                                                      "end_date", "start_date$$$end_date"]:
                                    src_mask[b_id][i] = 0
                                else:
                                    src_mask[b_id][i] = 1
                        else:
                            for i, (e_instance_name, _) in enumerate(data["instance_pool"]["edge"][pred_e_class]):
                                if e_instance_name.split(".")[-1] in ["from", "to", "from$$$to", "start_date",
                                                                      "end_date", "start_date$$$end_date"]:
                                    src_mask[b_id][i] = 1
                                else:
                                    src_mask[b_id][i] = 0

                    if self.args.use_kb_constraint:

                        cnt_timeout = 0
                        for i, (e_instance_name, _) in enumerate(data["instance_pool"]["edge"][pred_e_class]):
                            if src_mask[b_id][i] == 1:
                                continue
                            tmp_aqg = copy.deepcopy(aqg)
                            tmp_aqg.set_edge_instance(e_id, [i, e_instance_name])
                            tmp_aqg.set_edge_instance(get_inv_edge(e_id), [i, e_instance_name])
                            try:
                                tmp_queries = tmp_aqg.to_temporary_sparql_query(kb=self.kb)
                                if not tmp_queries:
                                    src_mask[b_id][i] = 1
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
                                    # print(t_ae)
                                    # print(one_query)
                                    # print(result)
                                    # print()
                                    if result == "TimeOut":
                                        result = [False]
                                        cnt_timeout += 1
                                    # print(src_mask[b_id])
                                    if not result[0]:
                                        src_mask[b_id][i] = 1
                                        break
                                if cnt_timeout >= 3:
                                    for j in range(i, len(data["instance_pool"]["edge"][pred_e_class])):
                                        src_mask[b_id][j] = 1
                                    break
                            except:
                                # if self.args.dataset != "lcq":
                                #     src_mask[b_id][i] = 1
                                src_mask[b_id][i] = 1
                                continue

                if pred_e_class in [E_CLASS_IDS["agg+"], E_CLASS_IDS["agg-"]]:
                    if self.args.dataset == "cwq":
                        for i, (e_instance_name, _) in enumerate(data["instance_pool"]["edge"][pred_e_class]):
                            if e_instance_name in ["ASK"]:
                                src_mask[b_id][i] = 1
                    elif self.args.dataset == "lcq":
                        for i, (e_instance_name, _) in enumerate(data["instance_pool"]["edge"][pred_e_class]):
                            if e_instance_name in ["MAX", "MIN"]:
                                src_mask[b_id][i] = 1
                    else:
                        pass

            if self.args.cuda:
                src_encodings = src_encodings.to(self.args.gpu)
                src_mask = src_mask.to(self.args.gpu)

            # action_prob:          (beam_num, max_cand_e_num)
            action_prob = self.edge_link_pointer_net(src_encodings=src_encodings, query_vec=query_t.unsqueeze(0),
                                                     src_token_mask=src_mask)
            action_prob = F.log_softmax(action_prob, dim=-1)

            new_aqg_meta = []  # save the possible directions for AQG expansion
            for b_id, aqg in enumerate(beams):
                pred_e_class = pred_aqg_obj_labels[t]
                pred_e_class = pred_e_class - 1 if pred_e_class % 2 == 1 else pred_e_class
                e_id = aqg.get_e_add_history(t_ae)

                # enumerate the edge instance
                for o_id in range(len(e_instance_pool[pred_e_class])):
                    if src_mask[b_id][o_id] == 1:
                        continue

                    if self.args.dataset in ["lcq", "wsp"] and self.args.use_matching_score:
                        e_name, e_true_name = data["instance_pool"]["edge"][pred_e_class][o_id]
                        if self.dataset == "lcq":
                            e_toks = tokenize_word_sentence(e_true_name)
                        else:
                            e_toks = tokenize_word_sentence(get_relation_last_name(e_name, kb="freebase"))
                        match_mode, match_len = cal_literal_matching_score(self.args.dataset,
                                                                           data["question_toks"],
                                                                           e_toks)
                        has_this_instance = False
                        e_last_name = " ".join(e_true_name.split(" ")[1:]).strip("s")
                        for _e_id, _e_instance in aqg.e_instances.items():
                            if aqg.get_edge_label(_e_id) not in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
                                continue
                            _e_true_name = get_relation_true_name(_e_instance[-1], kb=self.kb)
                            _e_last_name = " ".join(_e_true_name.split(" ")[1:]).strip("s")
                            if _e_last_name == e_last_name:
                                has_this_instance = True
                        if self.args.dataset == "lcq":
                            if not has_this_instance and ((match_mode == "ExactMatching" and match_len >= 1) or (match_mode == "PartialMatching" and match_len >= 3)):
                                # new_e_score = aqg.get_e_score() + self.args.beta * action_prob[b_id, o_id].cpu().detach().numpy() - self.args.alpha * 1.0 / match_len
                                new_e_score = aqg.get_e_score() + max(-self.args.alpha * 1.0 / match_len, action_prob[b_id, o_id].cpu().detach().numpy())
                            else:
                                new_e_score = aqg.get_e_score() + action_prob[b_id, o_id].cpu().detach().numpy()
                        else:
                            if (match_mode == "ExactMatching" and match_len >= 1) or (match_mode == "PartialMatching" and match_len >= 1):
                                # new_e_score = aqg.get_e_score() + self.args.beta * action_prob[b_id, o_id].cpu().detach().numpy() - self.args.alpha * 1.0 / match_len
                                new_e_score = aqg.get_e_score() + max(-self.args.alpha * 1.0 / match_len, action_prob[b_id, o_id].cpu().detach().numpy())
                            else:
                                new_e_score = aqg.get_e_score() + action_prob[b_id, o_id].cpu().detach().numpy()
                    else:
                        # update probability
                        new_e_score = aqg.get_e_score() + action_prob[b_id, o_id].cpu().detach().numpy()

                    # get the index of added edge at timestep t.
                    e_id = aqg.get_e_add_history(t_ae)
                    meta_entry = {
                        "edge": e_id,
                        "obj": o_id,
                        "obj_score": action_prob[b_id, o_id],
                        "new_e_score": new_e_score,
                        "prev_aqg_id": b_id
                    }
                    new_aqg_meta.append(meta_entry)

            if not new_aqg_meta:
                break

            # new_v_scores:       (beam_num)
            new_e_scores = self.new_tensor([x["new_e_score"] for x in new_aqg_meta])
            # select top-k beams with highest probs
            k = min(new_e_scores.size(0), beam_size - len(completed_beams))
            top_new_e_scores, meta_ids = torch.topk(new_e_scores, k=k)

            _cnt = 0
            live_aqg_ids = []
            new_beams = []
            for _, meta_id in zip(top_new_e_scores.data.cpu(), meta_ids.data.cpu()):

                aqg_meta_entry = new_aqg_meta[meta_id]
                e = aqg_meta_entry["edge"]
                obj = aqg_meta_entry["obj"]
                new_e_score = aqg_meta_entry["new_e_score"]
                prev_aqg_id = aqg_meta_entry['prev_aqg_id']
                prev_aqg = beams[prev_aqg_id]

                # build new AQG
                new_aqg = copy.deepcopy(prev_aqg)
                new_aqg.update_e_score(new_e_score)

                e_class = pred_aqg_obj_labels[t]
                e_class = e_class - 1 if e_class % 2 == 1 else e_class
                e_instance_name = data["instance_pool"]["edge"][e_class][obj]

                # set the predicted instance to edge v and its inverse edge
                new_aqg.set_edge_instance(e, [obj, e_instance_name[0]])
                new_aqg.set_edge_instance(get_inv_edge(e), [obj, e_instance_name[0]])
                new_aqg.pred_e_instance_labels.append([e_class, obj])
                new_beams.append(new_aqg)
                live_aqg_ids.append(prev_aqg_id)
                _cnt += 1

            # print()
            # print("######################################################################################")
            # print("Timestep: {}, Operation: {} ".format(t, "ae"))
            # for _aqg in new_beams:
            #     _aqg.show_state()

            if not live_aqg_ids:
                break

            h_last = (h_t[live_aqg_ids], cell_t[live_aqg_ids])
            beams = new_beams
            t += 1

        for aqg in beams:
            validity = True
            for e in aqg.edges:
                if e not in aqg.e_instances:
                    validity = False
                    break
            if validity:
                if self.dataset in ["lcq", "wsp"] and self.args.use_matching_score:
                    completed_beams.append([aqg,
                                            cal_edge_matching_total_score(self.args.dataset, aqg, e_instance_literal_score),
                                            aqg.get_e_score()])
                else:
                    completed_beams.append([aqg,
                                            0,
                                            aqg.get_e_score()])
        completed_beams.sort(key=itemgetter(1, 2), reverse=True)

        if self.dataset == "lcq" and self.args.use_matching_score:
            outstanding_idx = -1
            for _i, (aqg, s1, s2) in enumerate(completed_beams):
                if s2 > -0.001:
                    outstanding_idx = _i
            if outstanding_idx != -1:
                _completed_beams = [completed_beams[outstanding_idx]] + [x for _i, x in enumerate(completed_beams) if _i != outstanding_idx]
                completed_beams = [x for x in _completed_beams]

        # print()
        # print("===============================================================================================")
        # print("Complete Edge Beams:")
        # for x, s1, s2 in completed_beams:
        #     x.show_state()
        #     print(s1)

        completed_beams =[x[0] for x in completed_beams if x[0].check_final_query_graph(kb=self.kb)]

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