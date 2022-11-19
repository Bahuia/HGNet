# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : model_nhgg.py
# @Software: PyCharm
"""

import sys
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
sys.path.append("..")
from utils.embedding import Embeddings
from models.rnn import LSTM
from models.gnn import GraphTransformer
from models.attention import dot_prod_attention
from models.pointer_net import PointerNet
from models.nn_utils import *
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
        self.use_plm = self.args.plm_mode != "none"

        self.dataset = args.dataset

        if args.dataset == "lcq":
            self.kb = "dbpedia"
        else:
            self.kb = "freebase"
            
        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        self.tokenizer = pickle.load(open(args.vocab_path, 'rb'))
        self.pad = self.tokenizer.lookup(self.tokenizer.pad_token)

        self.word_embedding = Embeddings(args.d_emb, self.tokenizer)
        self.mention_feature_embedding = nn.Embedding(2, args.d_emb)

        # lstm encoder for question
        self.encoder = LSTM(d_input=args.d_emb, d_h=args.d_h // 2,
                            n_layers=args.n_lstm_layers, birnn=args.birnn, dropout=args.dropout)

        self.v_index_embedding = nn.Embedding(100, args.d_h)             # vertex ID
        self.e_index_embedding = nn.Embedding(100, args.d_h)               # edge ID

        self.v_segment_embedding = nn.Embedding(100, args.d_h)     # segment of vertex (subquery ID of vertex)
        self.e_segment_embedding = nn.Embedding(100, args.d_h)       # segment of edge (subquery ID of edge)

        self.segment_switch_embedding = nn.Embedding(2, args.d_h)       # 0 and 1 denote False and True,
                                                                        # whether switch the segment (subquery)

        self.decoder_cell_init = nn.Linear(args.d_h, args.d_h)

        # decoder for AQG generation
        self.decoder_lstm = nn.LSTMCell(args.d_h, args.d_h)
        self.enc_att_linear = nn.Linear(args.d_h, args.d_h)
        self.dec_input_linear = nn.Linear(args.d_h + args.d_h, args.d_h, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        # encoder for AQG at each time step
        self.graph_encoder = GraphTransformer(n_blocks=args.n_gnn_blocks,
                                              hidden_size=args.d_h, dropout=args.dropout)
        # active function
        self.read_out_active = torch.tanh if args.readout == 'non_linear' else identity

        self.query_vec_to_ae_vec = nn.Linear(args.d_h, args.d_h,
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

    def encode_instance(self, ins, ins_lens, ins_sids):
        """
        Encode the candidate instances of vertices and edges.
        @param ins:             (n_ins, max_len)
        @param ins_lens:        (n_ins)
        @param ins_classes:     (n_ins)
        @param ins_sids:        (n_ins)
        @return:       a candidate pool, List(bs), each element is a Dict {class_id: (_n_ins, d_h)}
        """
        # ins_enc:          (n_ins, max_len, d_h)
        # ins_vec:          (n_ins, d_h)
        ins_enc, _ = self.encode(ins, ins_lens)
        ins_enc = self.dropout(ins_enc)
        ins_vec = max_pooling_by_lens(ins_enc, ins_lens)

        # Split the vertex/edge tensors by their classes.
        ins_pool = instance_tensor_to_pool_without_class(ins_vec, ins_sids)
        return ins_pool

    def encode_question(self, q, q_lens, q_ment):
        """
        Encode natural language questions
        @param q:           (bs, max_q_len)
        @param q_lens:      (bs)
        @param q_ment:      (bs, max_q_len)     feature of the mention in NLQ, 0 denotes mention, 1 denotes not mention.
        @return:
        """
        # q_enc:            (bs, max_q_len, d_h)
        q_enc, (h_last_enc, c_last_enc) = self.encode(src=q,
                                                      src_lens=q_lens,
                                                      src_segment=q_ment if self.args.use_mention_feature else None,
                                                      segment_embedding=self.mention_feature_embedding)
        context = torch.cat([c_last_enc[0], c_last_enc[1]], -1)
        q_enc = self.dropout(q_enc)

        # mask different masks for different matching
        # q_mask:           (bs, max_q_len)     0: not mask, 1: mask
        # q_mask_for_v:     (bs, max_q_len)     0: not mask, 1: mask
        # q_mask_for_e:     (bs, max_q_len)     0: not mask, 1: mask
        q_mask = length_array_to_mask_tensor(q_lens)
        if self.args.dataset != "cwq":
            q_mask_for_v = length_array_to_mask_tensor(q_lens, value=q_ment, mask_symbol=0)
            q_mask_for_e = length_array_to_mask_tensor(q_lens, value=q_ment, mask_symbol=1)
        else:
            q_mask_for_v = q_mask
            q_mask_for_e = q_mask

        if self.args.cuda:
            q_mask = q_mask.to(self.args.gpu)
            q_mask_for_v = q_mask_for_v.to(self.args.gpu)
            q_mask_for_e = q_mask_for_e.to(self.args.gpu)

        # q_enc:            (bs, d_h)
        # q_enc_for_v:      (bs, d_h)
        # q_enc_for_e:      (bs, d_h)
        q_vec = max_pooling_by_lens(q_enc, q_lens)
        q_vec_for_v = max_pooling_by_mask(q_enc, q_mask_for_v)
        q_vec_for_e = max_pooling_by_mask(q_enc, q_mask_for_e)
        return q_enc, q_vec, q_vec_for_v, q_vec_for_e, q_mask, q_mask_for_v, q_mask_for_e, context

    def nhgg_decoding(self, q_enc, q_mask, tgt_inputs, tgt_lens, context):
        """
        Perform AQG decoding
        @param q_enc:           (bs, max_q_len, d_h)
        @param q_lens:          (bs)
        @param tgt_objs:        (bs, max_tgt_len)       tgt sequence, gold objects of AQG generation
        @param tgt_lens:        (bs)
        @param tgt_aqg_inputs:  List(max_tgt_len)       AQG embeddings (vertex and edges) input to the decoder
        @return:
               dec_out:         (max_tgt_len, bs, d_h)
               v_enc:           List(max_tgt_len), each element size (bs, n_v, d_h)
               e_enc:           List(max_tgt_len), each element size (bs, n_e, d_h)
               g_enc:           List(max_tgt_len), each element size (bs, d_h)
        """
        bs = q_enc.size(0)
        max_tgt_len = max(tgt_lens)

        g_enc = []
        v_enc = []
        e_enc = []

        # initialize state for decoder
        # dec_init_state:    (bs, d_h), (bs, d_h)
        dec_init_state = self.init_decoder_state(context, self.aqg_dec_cell_init)
        h_last = dec_init_state

        dec_out = []
        for t in range(max_tgt_len):
            # for the AQG in step t, we have
            # n_v = (t - 2) // 3 + 2        # number of vertices
            # n_e = (t - 1) // 3 * 2        # number of edges
            #
            # v:            (bs, n_v)     vertex id
            # e:            (bs, n_e)     edge id
            # v_class:      (bs, n_v)     vertex class label
            # v_class:      (bs, n_e)     edge class label
            # v_segment:    (bs, n_v)     the subgraph(segment) id of vertex
            # e_segment:    (bs, n_e)     the subgraph(segment) id of edge
            # adj:          (bs, n_v + n_e + 1, n_v + n_e + 1)    adjacency matrix

            h_t, c_t, one_v_enc, one_e_enc, one_g_enc = self.nhgg_decoding_step(t=t,
                                                                                     h_last=h_last,
                                                                                     q_enc=q_enc,
                                                                                     q_mask=q_mask,
                                                                                     tgt_inputs=tgt_inputs,
                                                                                     tgt_lens=tgt_lens)

            g_enc.append(one_g_enc)
            v_enc.append(one_v_enc)
            e_enc.append(one_e_enc)

            dec_out.append(h_t)
            h_last = (h_t, c_t)

        # dec_out = torch.stack(dec_out, dim=0)
        return dec_out, g_enc, v_enc, e_enc

    def nhgg_decoding_step(self, t, h_last, q_enc, q_mask, tgt_inputs, tgt_lens, v_ins_pool, e_ins_pool):
        v_embeds = []
        e_embeds = []
        adjs = []

        bs = q_enc.size(0)

        select_sids = []

        for sid in range(bs):
            n_v = (t - 2) // 3 + 2
            n_e = (t - 1) // 3 * 2

            if t < tgt_lens[sid]:
                v, v_ins, v_segment, \
                e, e_ins, e_segment, adj = tgt_inputs[sid][t]

                v_embed = v_ins_pool[sid].index_select(0, v_ins)
                e_embed = e_ins_pool[sid].index_select(0, e_ins)

                # use the embeddings of vertex ID, edge ID
                if self.args.use_id_embedding:
                    v_embed = torch.add(v_embed, self.v_index_embedding(v))
                    e_embed = torch.add(e_embed, self.e_index_embedding(e))

                # use the embeddings of segment
                if self.args.use_segment_embedding:
                    v_embed = torch.add(v_embed, self.v_segment_embedding(v_segment))
                    e_embed = torch.add(e_embed, self.e_segment_embedding(e_segment))

                v_embeds.append(v_embed)
                e_embeds.append(e_embed)
                adjs.append(adj)

                select_sids.append(sid)

        select_sids.append(bs)

        v_embeds, e_embeds, adjs, padding_mask = pad_input_embeds_for_gnn(v_embeds, e_embeds, adjs)

        one_v_enc, one_e_enc, one_g_enc = self.encode_graph(t, v_embeds, e_embeds, adjs)

        one_v_enc, one_e_enc, one_g_enc = pad_output_embeds_for_gnn(one_v_enc, one_e_enc, one_g_enc, select_sids)

        # tgt_embeds:     (bs, d_h)
        if self.args.use_graph_encoder:
            tgt_embeds = one_g_enc
        else:
            tgt_embeds = h_last[0]

        # one step decoding
        # h_t:      (bs, d_h)
        # c_t:      (bs, d_h)
        (h_t, c_t), ctx = self.decode_step(self.aqg_decoder,
                                           self.aqg_enc_att_aff,
                                           self.aqg_dec_input_aff,
                                           h_last=h_last,
                                           src_embeds=q_enc,
                                           src_mask=q_mask,
                                           tgt_embeds=tgt_embeds)
        return h_t, c_t, one_v_enc, one_e_enc, one_g_enc

    def get_nhgg_loss(self, dec_out, v_enc, v_ins_pool, v_ins_pool_mask, e_ins_pool, e_ins_pool_mask, tgt_objs, tgt_lens, tgt_seg_switch_objs, tgt_v_copy_objs, tgt_e_copy_objs, data):
        """
        Calculate the loss of AQG decoding
        @param dec_out:                 (max_tgt_len, bs, d_h)
        @param v_enc:                   List(max_tgt_len), each element size (bs, n_v, d_h), vertex embeddings
        @param e_enc:                   List(max_tgt_len), each element size (bs, n_v, d_h), edge embeddings
        @param tgt_objs:                (bs, max_tgt_len), ground truth of objects at each AQG decoding step
        @param tgt_lens:                (bs)
        @param tgt_seg_switch_objs:     List(bs), ground truth whether switch segment at each Add Vertex step
        @param tgt_v_copy_objs:         List(bs), ground truth whether copy vertex at each AddVertex step
        @param tgt_e_copy_objs:         List(bs), ground truth whether copy edge at each AddEdge step
        @param data:                    List(bs)
        @return:                        (bs)
        """
        bs = len(tgt_objs)
        max_tgt_len = len(dec_out)

        # scores:           List(bs)                scores of the target action
        # action_probs:     List(bs, action_sz)     Probability distribution for each prediction (without softmax)
        scores = [[] for _ in range(bs)]
        action_probs = [[] for _ in range(bs)]

        for t in range(max_tgt_len):
            # Calculate probabilities for each operation
            action_prob = get_nhgg_action_probability(t=t,
                                                      dec_out=dec_out,
                                                      av_pointer_net=self.av_pointer_net,
                                                      ae_pointer_net=self.ae_pointer_net,
                                                      sv_pointer_net=self.sv_pointer_net,
                                                      v_enc=v_enc,
                                                      v_ins_pool=v_ins_pool,
                                                      v_ins_pool_mask=v_ins_pool_mask,
                                                      e_ins_pool=e_ins_pool,
                                                      e_ins_pool_mask=e_ins_pool_mask,
                                                      args=self.args,
                                                      data=data)

            # Recording all action probabilities (without softmax) for AQG decoding
            # action_probs:     List(bs, tgt_len)     Probability distribution for each prediction (without softmax)
            for sid in range(bs):
                if t < tgt_lens[sid]:
                    action_probs[sid].append(action_prob[sid])

            action_prob = F.softmax(action_prob, dim=-1)

            # Save the probabilities(softmax) of target actions as loss
            # scores:           List(bs)                scores of the target action
            for sid in range(bs):
                if t < tgt_lens[sid]:
                    scores[sid].append(action_prob[sid, tgt_objs[sid][t]])

        # Base structure loss
        # loss:   (bs)
        loss = -torch.stack([torch.stack(score_i, dim=0).log().sum() for score_i in scores], dim=0)

        # Segment switch loss
        if self.args.dataset == "cwq":
            # seg_switch_loss:     (bs)
            seg_switch_loss = self.get_segment_switch_loss(dec_out=dec_out,
                                                           tgt_objs=tgt_objs,
                                                           tgt_lens=tgt_lens,
                                                           tgt_seg_switch_objs=tgt_seg_switch_objs)
            loss = torch.add(loss, seg_switch_loss)

        return loss, action_probs

    def forward(self, batch):
        """
        @param batch:           batch of processed data
        @return:
        """

        # q:                    (bs, max_q_len)
        # q_lens:               (bs)
        # q_ment:               (bs, max_q_len)
        # v_ins:                (total_n_v, max_v_len)
        # v_ins_lens:           (total_n_v)
        # v_ins_classes:        List(total_n_v)   the class label of each vertex
        # v_ins_sids:           List(total_n_v)   the sample(question) id that each vertex belongs to
        # v_ins_names:          List(total_n_v)   the name and true name of each vertex
        # e_ins:                (total_n_e, max_e_len)
        # e_ins_lens:           (total_n_e)
        # e_ins_classes:        List(total_n_e)   the class label of each edge
        # e_ins_sids:           List(total_n_e)   the sample(question) id that each edge belongs to
        # e_ins_names:          List(total_n_e)   the name and true name of each vertex
        # tgt_aqgs:             List(max_tgt_len)    original AQG data structure at each step
        # tgt_aqg_inputs:       List(max_tgt_len)    AQG embeddings (vertex and edges) input to graph encoder,
        #                                            each element: (v, v_class, v_segment, e, e_class, e_segment, adj)
        # tgt_aqg_objs:         (bs, max_tgt_len), ground truth of objects at each step
        # tgt_lens:             (bs)
        # tgt_v_ins_objs:       List(bs), ground truth for vertex instance linking
        # tgt_e_ins_objs:       List(bs), ground truth for edge instance linking
        # tgt_v_copy_objs:      List(bs), ground truth whether copy vertex at each AddVertex step
        # tgt_e_copy_objs:      List(bs), ground truth whether copy edge at each AddEdge step
        # tgt_seg_switch_objs:  List(bs), ground truth whether switch segment at each Add Vertex step
        # data:                 List(bs), original data
        q, q_lens, q_ment, match_f, \
        v_ins, v_ins_lens, v_ins_classes, v_ins_sids, v_ins_names, \
        e_ins, e_ins_lens, e_ins_classes, e_ins_sids, e_ins_names, \
        tgt_aqgs, tgt_aqg_inputs, tgt_aqg_objs, \
        tgt_v_ins_objs, tgt_e_ins_objs, \
        tgt_v_copy_objs, tgt_e_copy_objs, tgt_seg_switch_objs, data = batch

        tgt_aqg_lens = [len(x) for x in tgt_aqgs]

        ################################### Encoding question and instances ############################################
        # v_ins_pool:      List(bs), each element is a DICT {v_class_id: (total_n_v_class, d_h)}
        # e_ins_pool:      List(bs), each element is a DICT {e_class_id: (total_n_e_class, d_h)}
        v_ins_pool = self.encode_instance(v_ins, v_ins_lens, v_ins_sids)
        e_ins_pool = self.encode_instance(e_ins, e_ins_lens, e_ins_sids)

        v_ins_pool, v_ins_pool_lens = pad_tensor_1d(v_ins_pool, 0)
        e_ins_pool, e_ins_pool_lens = pad_tensor_1d(e_ins_pool, 0)

        v_ins_pool_mask = length_array_to_mask_tensor(v_ins_pool_lens)
        e_ins_pool_mask = length_array_to_mask_tensor(e_ins_pool_lens)
        if self.args.cuda:
            v_ins_pool_mask = v_ins_pool_mask.to(self.args.gpu)
            e_ins_pool_mask = e_ins_pool_mask.to(self.args.gpu)


        # q_enc:            (bs, max_q_len, d_h)
        # q_vec:            (bs, d_h)
        # q_vec_for_v:      (bs, d_h)
        # q_vec_for_e:      (bs, d_h)
        q_enc, \
        q_vec, q_vec_for_v, q_vec_for_e, \
        q_mask, q_mask_for_v, q_mask_for_e, context = self.encode_question(q, q_lens, q_ment)

        ##################################### Decoding for AQG outlining ###############################################


        # dec_out:         (max_tgt_len, bs, d_h)
        # v_enc:           List(max_tgt_len), each element size (bs, n_v, d_h)
        # e_enc:           List(max_tgt_len), each element size (bs, n_e, d_h)
        # g_enc:           List(max_tgt_len), each element size (bs, d_h)
        dec_out, g_enc, v_enc, e_enc = self.decoding(q_enc=q_enc,
                                                               q_mask=q_mask_for_e if self.args.use_mention_feature else q_mask,
                                                               tgt_aqg_inputs=tgt_aqg_inputs,
                                                               tgt_lens=tgt_aqg_lens,
                                                               context=context)

        # loss:     (bs)
        loss, action_probs = self.get_nhgg_loss(dec_out=dec_out,
                                                     v_enc=v_enc,
                                                     e_enc=e_enc,
                                                     tgt_objs=tgt_aqg_objs,
                                                     tgt_lens=tgt_aqg_lens,
                                                     tgt_seg_switch_objs=tgt_seg_switch_objs,
                                                     tgt_v_copy_objs=tgt_v_copy_objs,
                                                     tgt_e_copy_objs=tgt_e_copy_objs,
                                                     data=data)

        ##################################### Decoding for vertex filling ##############################################
        # v_dec_out:        (max_tgt_av_len, bs, d_h)
        v_dec_out = self.filling_decoding(q_enc=q_enc,
                                          q_vec=q_vec_for_v if self.args.use_mention_feature else q_vec,
                                          q_mask=q_mask_for_v if self.args.use_mention_feature else q_mask,
                                          tgt_objs=tgt_aqg_objs,
                                          tgt_aqgs=tgt_aqgs,
                                          tgt_ins_objs=tgt_v_ins_objs,
                                          ins_pool=v_ins_pool,
                                          g_enc=g_enc,
                                          ins_enc=v_enc,
                                          context=context,
                                          mode="vertex",
                                          args=self.args)


        # v_loss:   (bs)
        v_loss, v_action_probs = self.get_filling_loss(dec_out=v_dec_out,
                                                       q_vec=q_vec_for_v if self.args.use_mention_feature else q_vec,
                                                       link_pointer_net=self.v_link_pointer_net,
                                                       tgt_objs=tgt_aqg_objs,
                                                       tgt_lens=tgt_aqg_lens,
                                                       tgt_ins_objs=tgt_v_ins_objs,
                                                       ins_pool=v_ins_pool,
                                                       mode="vertex")

        ###################################### Decoding for edge filling ###############################################
        # e_dec_out:        (max_tgt_ae_len, bs, d_h)
        e_dec_out = self.filling_decoding(q_enc=q_enc,
                                          q_vec=q_vec_for_e if self.args.use_mention_feature else q_vec,
                                          q_mask=q_mask_for_e if self.args.use_mention_feature else q_mask,
                                          tgt_objs=tgt_aqg_objs,
                                          tgt_aqgs=tgt_aqgs,
                                          tgt_ins_objs=tgt_e_ins_objs,
                                          ins_pool=e_ins_pool,
                                          g_enc=g_enc,
                                          ins_enc=e_enc,
                                          context=context,
                                          mode="edge",
                                          args=self.args)


        # e_loss:   (bs)
        e_loss, e_action_probs = self.get_filling_loss(dec_out=e_dec_out,
                                                       q_vec=q_vec_for_e if self.args.use_mention_feature else q_vec,
                                                       link_pointer_net=self.e_link_pointer_net,
                                                       tgt_objs=tgt_aqg_objs,
                                                       tgt_lens=tgt_aqg_lens,
                                                       tgt_ins_objs=tgt_e_ins_objs,
                                                       ins_pool=e_ins_pool,
                                                       mode="edge")

        # loss:     (bs)
        loss = torch.add(loss, v_loss)
        loss = torch.add(loss, e_loss)

        return loss, action_probs, v_action_probs, e_action_probs

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

        # v_instance_encodings:     (total_v_num, max_v_len, d_h)
        # v_instance_vec:           (total_v_num, d_h)
        if not self.use_plm:
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
        if not self.use_plm:
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

        v_instance_pool_mask = length_array_to_mask_tensor(v_instance_pool_lens)
        e_instance_pool_mask = length_array_to_mask_tensor(e_instance_pool_lens)
        if self.args.cuda:
            v_instance_pool_mask = v_instance_pool_mask.to(self.args.gpu)
            e_instance_pool_mask = e_instance_pool_mask.to(self.args.gpu)

        # encoding question
        # q_encodings:      (bs, max_q_len, d_h)
        # q_mask:           (bs, max_q_len)     0: True, 1: False
        # enc_h_last:       (2, bs, d_h // 2)
        # enc_cell_last:    (2, bs, d_h // 2)
        if not self.use_plm:
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

                    # use the embeddings of vertex ID, edge ID
                    if self.args.use_id_embedding:
                        v_embed = self.v_embedding(v_tensor)           # Index vertex embedding
                        e_embed = self.e_embedding(e_tensor)             # Index edge embedding
                        vertex_embed = torch.add(vertex_embed, v_embed)
                        edge_embed = torch.add(edge_embed, e_embed)

                    # use the embeddings of segment
                    if self.args.use_segment_embedding:
                        v_segment_embed = self.v_segment_embedding(v_segment_tensor)     # Segment vertex embedding
                        e_segment_embed = self.e_segment_embedding(e_segment_tensor)       # Segment vertex embedding
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
        if not self.use_plm:
            v_instance_encodings, _ = self.encode(v_instance_tensor, v_instance_lens)
        else:
            v_instance_encodings, _ = self.bert_encode(v_instance_tensor, v_instance_lens)
        v_instance_vec = max_pooling_by_lens(v_instance_encodings, v_instance_lens)

        if self.args.use_matching_feature:
            # TODO: vertex matching feature
            pass

        # e_instance_encodings:     (total_e_num, max_e_len, d_h)
        # e_instance_vec:           (total_e_num, d_h)
        if not self.use_plm:
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
        if not self.use_plm:
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
        while len(completed_beams) < beam_size and t < self.args.max_step_n:

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
                    v_embed = self.v_embedding(v_tensor)       # Index vertex embedding
                    e_embed = self.e_embedding(e_tensor)         # Index edge embedding
                    vertex_embed = torch.add(vertex_embed, v_embed)
                    edge_embed = torch.add(edge_embed, e_embed)

                # use the embeddings of segment
                if self.args.use_segment_embedding:
                    v_segment_embed = self.v_segment_embedding(v_segment_tensor)   # Segment vertex embedding
                    e_segment_embed = self.e_segment_embedding(e_segment_tensor)     # Segment edge embedding
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

    def encode_graph(self, t, vertex_tensor, edge_tensor, adj):
        """
        encode a graph by the adjacency matrix
        @param vertex_tensor:       (v_num, d_h)
        @param edge_tensor:         (e_num, d_h)
        @param adj:                 (v_num + e_num + 1, v_num + e_num + 1)
        @return:                    (v_num, d_h), (e_num, d_h), (d_h)
        """
        return self.graph_encoder(t, vertex_tensor, edge_tensor, adj)

    def encode(self, src, src_lens, src_segment=None, segment_embedding=None, init_states=None):
        """
        Encode the source sequence
        @param src:             (bs, max_src_len)
        @param src_lens:        (bs)
        @return:
            src_enc:            (bs, max_src_len, d_h)
        """

        # src_embed:        (bs, max_src_len, d_emb)
        src_embed = self.word_embedding(src)

        # Add 0/1 segment feature for each position, e.g., mention 0, other 1
        # src_embed:        (bs, max_src_len, d_emb)
        if src_segment is not None:
            src_embed = torch.add(src_embed, segment_embedding(src_segment))

        # src_enc:          (bs, max_src_len, d_h)
        src_enc, final_states = self.encoder(src_embed, src_lens, init_states=init_states)
        return src_enc, final_states

    def decode_step(self, decoder_lstm, enc_att_linear, dec_input_linear, h_last, src_embeds, tgt_embeds,
                    src_mask=None):
        """
        one decoding step
        @param h_last:                  (bs, d_h)
        @param src_encodings:           (bs, max_seq_len, d_h)
        @param dec_input_encodings:     (bs, d_h)
        @param src_token_mask:          (bs, max_seq_len)
        @return:
        """
        src_embeds_aff = enc_att_linear(src_embeds)

        # context_t:        (bs, d_h)
        # alpha_t:          (bs, max_seq_len), attention weights
        context_t, alpha_t = dot_prod_attention(tgt_embeds,
                                                src_embeds,
                                                src_embeds_aff,
                                                mask=src_mask)

        tgt_embeds_aff = torch.tanh(dec_input_linear(torch.cat([tgt_embeds, context_t], 1)))
        tgt_embeds_aff = self.dropout(tgt_embeds_aff)

        h_t, cell_t = decoder_lstm(tgt_embeds_aff, h_last)

        return (h_t, cell_t), context_t

    def init_decoder_state(self, enc_last_cell, decoder_cell_init):
        """
        @param enc_last_cell:   (bs, d_h)
        """
        h_0 = decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)
        return h_0, Variable(self.new_tensor(h_0.size()).zero_())