# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/30
# @Author  : Yongrui Chen
# @File    : transformer_model.py
# @Software: PyCharm
"""

import sys
import pickle
import torch.nn as nn
sys.path.append("..")
from utils.embedding import Embeddings, PositionalEncoding
from models.rnn import LSTM
from models.gnn import GraphTransformer
from models.attention import dot_prod_attention
from models.pointer_net import PointerNet
from models.nn_utils import *
from utils.beam import Beam


V_CLASS_NUM = len(V_CLASS_IDS)
E_CLASS_NUM = len(E_CLASS_IDS)


class HGNet(nn.Module):

    def __init__(self, args):

        super(HGNet, self).__init__()
        self.args = args

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

        self.positional_encoding = PositionalEncoding(args.d_h)

        self.tokenizer = pickle.load(open(args.wo_vocab, 'rb'))
        self.pad = self.tokenizer.lookup(self.tokenizer.pad_token)

        self.word_embedding = Embeddings(args.d_emb, self.tokenizer)
        self.mention_feature_embedding = nn.Embedding(2, args.d_emb)

        self.embedding_to_hidden = nn.Linear(args.d_emb, args.d_h, bias=False)

        self.encoder = LSTM(d_input=args.d_emb, d_h=args.d_h // 2,
                            n_layers=args.n_lstm_layers, birnn=args.birnn, dropout=args.dropout)

        self.d_h_tmp = args.d_h

        self.v_class_embedding = nn.Embedding(V_CLASS_NUM, args.d_h)       # embeddings for vertex class
        self.e_class_embedding = nn.Embedding(E_CLASS_NUM, args.d_h)       # embeddings for edge class

        self.v_index_embedding = nn.Embedding(100, args.d_h)                # vertex ID
        self.e_index_embedding = nn.Embedding(100, args.d_h)                # edge ID

        self.v_segment_embedding = nn.Embedding(100, args.d_h)          # segment of vertex (subquery ID of vertex)
        self.e_segment_embedding = nn.Embedding(100, args.d_h)          # segment of edge (subquery ID of edge)

        self.segment_switch_embedding = nn.Embedding(2, args.d_h)       # 0 and 1 denote False and True,
                                                                        # whether switch the segment (subquery)

        self.zero_embed = Variable(self.new_tensor(args.d_h).zero_(), requires_grad=False)   # the embedding of the begining token to the decoder

        self.aqg_dec_cell_init = nn.Linear(self.d_h_tmp, args.d_h)
        self.v_dec_cell_init = nn.Linear(self.d_h_tmp, args.d_h)
        self.e_dec_cell_init = nn.Linear(self.d_h_tmp, args.d_h)

        # decoder for AQG generation
        self.aqg_decoder = nn.LSTMCell(args.d_h, args.d_h)
        self.aqg_enc_att_aff = nn.Linear(self.d_h_tmp, args.d_h)
        self.aqg_dec_input_aff = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)

        # decoder for predicting instance of each vertex (vertex linking)
        self.v_decoder = nn.LSTMCell(args.d_h, args.d_h)
        self.v_enc_att_aff = nn.Linear(self.d_h_tmp, args.d_h)
        self.v_dec_input_aff = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)
        self.v_dec_output_aff = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)

        # decoder for predicting instance of each edge (edge linking)
        self.e_decoder = nn.LSTMCell(args.d_h, args.d_h)
        self.e_enc_att_aff = nn.Linear(self.d_h_tmp, args.d_h)
        self.e_dec_input_aff = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)
        self.e_dec_output_aff = nn.Linear(self.d_h_tmp + args.d_h, args.d_h, bias=False)

        self.v_dec_act_aff = nn.Linear(self.d_h_tmp, args.d_h, bias=False)
        self.e_dec_act_aff = nn.Linear(self.d_h_tmp, args.d_h, bias=False)

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
                                             self.v_class_embedding.weight, self.av_readout_b)
        self.ae_readout = lambda q: F.linear(self.read_out_active(self.query_vec_to_ae_vec(q)),
                                             self.e_class_embedding.weight, self.ae_readout_b)
        # pointer network for "select vertex" operation
        self.sv_pointer_net = PointerNet(args.d_h, args.d_h, attention_type=args.att_type)

        # classifier for predicting whether switch segment
        self.seg_readout_b = nn.Parameter(torch.FloatTensor(2).zero_())
        self.seg_readout = lambda q: F.linear(self.read_out_active(self.query_vec_to_ae_vec(q)),
                                              self.segment_switch_embedding.weight, self.seg_readout_b)

        # pointer networks for predicting to copy which vertex and which edge
        self.v_copy_pointer_net = PointerNet(args.d_h, args.d_h, attention_type=args.att_type)
        self.e_copy_pointer_net = PointerNet(args.d_h, args.d_h, attention_type=args.att_type)

        # placeholder encoding that denotes "do not copy vertex" and "do not copy edge"
        self.not_v_copy_enc = nn.Parameter(self.new_tensor(args.d_h))
        self.not_e_copy_enc = nn.Parameter(self.new_tensor(args.d_h))

        # pointer networks for vertex linking and edge linking
        self.v_link_pointer_net = PointerNet(args.d_h,
                                             self.d_h_tmp,
                                             attention_type=args.att_type)
        self.e_link_pointer_net = PointerNet(args.d_h,
                                             self.d_h_tmp,
                                             attention_type=args.att_type)

    def encode_instance(self, ins, ins_lens, ins_classes, ins_sids):
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
        ins_pool = instance_tensor_to_pool(ins_vec, ins_classes, ins_sids)
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

    def outlining_decoding(self, q_enc, q_mask, tgt_aqg_inputs, tgt_lens, context):
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

            h_t, c_t, one_v_enc, one_e_enc, one_g_enc = self.outlining_decoding_step(t=t,
                                                                                     h_last=h_last,
                                                                                     q_enc=q_enc,
                                                                                     q_mask=q_mask,
                                                                                     tgt_aqg_inputs=tgt_aqg_inputs,
                                                                                     tgt_lens=tgt_lens)

            g_enc.append(one_g_enc)
            v_enc.append(one_v_enc)
            e_enc.append(one_e_enc)

            dec_out.append(h_t)
            h_last = (h_t, c_t)

        # dec_out = torch.stack(dec_out, dim=0)
        return dec_out, g_enc, v_enc, e_enc

    def outlining_decoding_step(self, t, h_last, q_enc, q_mask, tgt_aqg_inputs, tgt_lens):
        v_embeds = []
        e_embeds = []
        adjs = []

        bs = q_enc.size(0)

        select_sids = []

        for sid in range(bs):
            n_v = (t - 2) // 3 + 2
            n_e = (t - 1) // 3 * 2

            if t < tgt_lens[sid]:

                v, v_class, v_segment, \
                e, e_class, e_segment, adj = tgt_aqg_inputs[sid][t]

                v_embed = self.v_class_embedding(v_class)  # Class vertex embedding
                e_embed = self.e_class_embedding(e_class)  # Class edge embedding

                # use id embeddings
                if self.args.use_id_embedding:
                    v_embed = torch.add(v_embed, self.v_index_embedding(v))
                    e_embed = torch.add(e_embed, self.e_index_embedding(e))

                # use segment embeddings
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

    def filling_decoding(self, q_enc, q_vec, q_mask, tgt_objs, tgt_aqgs, tgt_ins_objs, ins_pool, g_enc, ins_enc, context, mode, args):
        """
        Perform vertex/edge instance decoding
        @param q_enc:           (bs, max_q_len, d_h)
        @param q_lens:          (bs)
        @param tgt_objs:        (bs, max_tgt_len), tgt sequence, gold objects of AQG generation
        @param tgt_aqgs:        List(bs), original AQG data structure at each step
        @param tgt_ins_objs:    List(bs), ground truth for vertex/edge instance filling
        @param tgt_lens:        (bs)
        @param ins_pool:        List(bs), candidate pool of vertex/edge, each element is a DICT {class_id: (total_n_class, d_h)}
        @param cur_g_enc:       (bs, d_h)
        @param cur_ins_enc:     (bs, n_ins, d_h)
        @param mode:            "vertex" and "edge"
        @param t_constraint:    when testing, give this constrained step number, denoting the decoding goes t_ins steps.
        @return:
                dec_out:        (max_tgt_ins_len, bs, d_h)
        """

        max_tgt_len = max([len(x) for x in tgt_objs])

        if mode == "vertex":
            dec_init_state = self.init_decoder_state(context, self.v_dec_cell_init)
        else:
            dec_init_state = self.init_decoder_state(context, self.e_dec_cell_init)
        h_last = dec_init_state

        final_g_enc, final_ins_enc = get_final_graph_embeddings(g_enc, ins_enc, tgt_objs)

        dec_out = []
        # Make the tgt input embeddings for the current decoding step
        for t in range(max_tgt_len - 1):
            if not allow_filling_at_t(t, mode):
                continue

            h_t, c_t, query_t = self.filling_decoding_step(t=t,
                                                           h_last=h_last,
                                                           q_enc=q_enc,
                                                           q_vec=q_vec,
                                                           q_mask=q_mask,
                                                           tgt_objs=tgt_objs,
                                                           tgt_aqgs=tgt_aqgs,
                                                           tgt_ins_objs=tgt_ins_objs,
                                                           ins_pool=ins_pool,
                                                           g_enc=final_g_enc,
                                                           ins_enc=final_ins_enc,
                                                           mode=mode,
                                                           args=args)

            dec_out.append(query_t)
            h_last = (h_t, c_t)
        dec_out = torch.stack(dec_out, dim=0)
        return dec_out

    def filling_decoding_step(self, t, h_last, q_enc, q_vec, q_mask, tgt_objs, tgt_aqgs, tgt_ins_objs, ins_pool, g_enc, ins_enc, mode, args):

        # ont_tgt_embed:    (bs, d_h)
        one_tgt_embed = mk_instance_decoder_input(t=t,
                                                  tgt_objs=tgt_objs,
                                                  tgt_ins_objs=tgt_ins_objs,
                                                  ins_pool=ins_pool,
                                                  mode=mode,
                                                  args=args)

        if mode == "vertex":
            one_tgt_embed = self.v_dec_act_aff(one_tgt_embed)
        else:
            one_tgt_embed = self.e_dec_act_aff(one_tgt_embed)

        # Add the final AQG embeddings (global graph structural information)
        # final_g_enc:      (max_tgt_ins_len, bs, d_h)
        if self.args.use_graph_auxiliary_vector:
            one_tgt_embed = combine_graph_auxiliary_encoding(one_tgt_embed=one_tgt_embed,
                                                             g_enc=g_enc)

        # Add the final vertex embeddings (vertex structural information)
        if self.args.use_instance_auxiliary_encoding:
            # final_ins_enc:    (max_tgt_ins_len, bs, d_h)
            # tgt_embeds:       (max_tgt_ins_len, bs, d_h)
            one_tgt_embed = combine_instance_auxiliary_encoding(one_tgt_embed=one_tgt_embed,
                                                                ins_enc=ins_enc,
                                                                t=t,
                                                                tgt_aqgs=tgt_aqgs,
                                                                mode=mode)
        # one step decoding
        # h_t:      (bs, d_h)
        # c_t:      (bs, d_h)
        if mode == "vertex":
            (h_t, c_t), ctx = self.decode_step(self.v_decoder,
                                               self.v_enc_att_aff,
                                               self.v_dec_input_aff,
                                               h_last=h_last,
                                               src_embeds=q_enc,
                                               src_mask=q_mask,
                                               tgt_embeds=one_tgt_embed)
            query_t = enhance_decoder_output(h_t=h_t,
                                             ctx=ctx,
                                             q_vec=q_vec,
                                             decoder_output_aff=self.v_dec_output_aff,
                                             context_mode=args.context_mode)
        else:
            (h_t, c_t), ctx = self.decode_step(self.e_decoder,
                                               self.e_enc_att_aff,
                                               self.e_dec_input_aff,
                                               h_last=h_last,
                                               src_embeds=q_enc,
                                               src_mask=q_mask,
                                               tgt_embeds=one_tgt_embed)
            query_t = enhance_decoder_output(h_t=h_t,
                                             ctx=ctx,
                                             q_vec=q_vec,
                                             decoder_output_aff=self.e_dec_output_aff,
                                             context_mode=args.context_mode)

        return h_t, c_t, query_t

    def get_outlining_loss(self, dec_out, v_enc, e_enc, tgt_objs, tgt_lens, tgt_seg_switch_objs, tgt_v_copy_objs, tgt_e_copy_objs, data):
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
            action_prob = get_outlining_action_probability(t=t,
                                                           dec_out=dec_out,
                                                           av_readout=self.av_readout,
                                                           ae_readout=self.ae_readout,
                                                           sv_pointer_net=self.sv_pointer_net,
                                                           v_enc=v_enc,
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

        # Vertex copy loss
        if self.args.use_v_copy:
            # v_copy_loss:     (bs)
            v_copy_loss = self.get_copy_loss(dec_out=dec_out,
                                             tgt_objs=tgt_objs,
                                             tgt_lens=tgt_lens,
                                             tgt_copy_objs=tgt_v_copy_objs,
                                             enc=v_enc,
                                             mode="vertex")
            loss = torch.add(loss, v_copy_loss)

        # Edge copy loss
        if self.args.use_e_copy:
            # e_copy_loss:     (bs)
            e_copy_loss = self.get_copy_loss(dec_out=dec_out,
                                             tgt_objs=tgt_objs,
                                             tgt_lens=tgt_lens,
                                             tgt_copy_objs=tgt_e_copy_objs,
                                             enc=e_enc,
                                             mode="edge")
            loss = torch.add(loss, e_copy_loss)
        return loss, action_probs

    def get_filling_loss(self, dec_out, q_vec, link_pointer_net, tgt_objs, tgt_lens, tgt_ins_objs, ins_pool, mode):
        """
        Calculate the loss of vertex/edge instance decoding
        @param dec_out:                 (max_tgt_ins_len, bs, d_h)
        @param q_vec:                   (bs, d_h)   question vector
        @param q_vec_for_ins:           (bs, d_h)   question vector with d_h
        @param decoder_output_aff:      transform affine layer
        @param link_pointer_net:        pointer network layer
        @param tgt_objs:                (bs, max_tgt_len), ground truth of objects at each AQG decoding step
        @param tgt_lens:                (bs)
        @param tgt_ins_objs:            List(bs), ground truth for vertex/edge instance filling
        @param ins_pool:                List(bs), candidate pool of vertex/edge, each element is a DICT {class_id: (total_n_class, d_h)}
        @param mode:                    "vertex" or "edge"
        @return:                        (bs)
        """

        bs = q_vec.size(0)
        max_tgt_len = max(tgt_lens)

        # scores:           List(bs)                Scores of the target action
        # action_probs:     List(bs, action_sz)     Probability distribution for each prediction (without softmax)
        scores = [[] for _ in range(bs)]
        action_probs = [[] for _ in range(bs)]

        for t in range(max_tgt_len - 1):
            if not allow_filling_at_t(t, mode):
                continue
            t_ins = step_to_op_step(t, mode)

            # action_prob:    (bs, max_n_ins)
            action_prob = get_filling_action_probability(t=t,
                                                         dec_out=dec_out,
                                                         link_pointer_net=link_pointer_net,
                                                         tgt_objs=tgt_objs,
                                                         tgt_ins_objs=tgt_ins_objs,
                                                         ins_pool=ins_pool,
                                                         mode=mode)

            # Recording all action probabilities (without softmax) for vertex linking
            # action_probs:     List(bs, tgt_av_len)     Probability distribution for each prediction (without softmax)
            for sid in range(bs):
                if allow_filling_at_t_for_sid(sid, t_ins, tgt_ins_objs):
                    action_probs[sid].append(action_prob[sid])
            action_prob = F.softmax(action_prob, dim=-1)

            # Save the probabilities(softmax) of target actions as loss
            # scores:           List(bs)                scores of the target action
            for sid in range(bs):
                if allow_filling_at_t_for_sid(sid, t_ins, tgt_ins_objs):
                    scores[sid].append(action_prob[sid, tgt_ins_objs[sid][t_ins]])

        # Vertex linking loss
        # loss:   (bs)
        loss = -torch.stack([torch.stack(score_i, dim=0).log().sum() for score_i in scores], dim=0)
        return loss, action_probs

    def get_segment_switch_loss(self, dec_out, tgt_objs, tgt_lens, tgt_seg_switch_objs):
        """
        Calculate the loss of segment switch in AQG generation
        @param dec_out:                 (max_tgt_len, bs, d_h)
        @param tgt_objs:                (bs, max_tgt_len), ground truth of objects at each step
        @param tgt_lens:                (bs)
        @param tgt_seg_switch_objs:     List(bs), ground truth whether switch segment at each Add Vertex step
        @return:  score                 (bs)
        """

        bs = len(tgt_objs)
        max_tgt_len = len(dec_out)
        scores = [[] for _ in range(bs)]

        for t in range(max_tgt_len):
            op = get_operator_by_t(t)
            if op == "av":
                # Switch segment only occurs in Add Vertex operation
                # action_prob:   (bs, 2)
                action_prob = get_segment_switch_action_probability(t=t,
                                                                    dec_out=dec_out,
                                                                    seg_readout=self.seg_readout)
                action_prob = F.softmax(action_prob, dim=-1)
                for sid in range(bs):
                    if t < tgt_lens[sid] and tgt_objs[sid][t] != V_CLASS_IDS["end"]:
                        # Get time step of add vertex operation
                        t_av = step_to_av_step(t)
                        scores[sid].append(action_prob[sid, tgt_seg_switch_objs[sid][t_av]])

        # score     (bs)
        score = torch.stack([torch.stack(score_i, dim=0).log().sum() for score_i in scores], dim=0)
        return -score

    def get_copy_loss(self, dec_out, tgt_objs, tgt_lens, tgt_copy_objs, enc, mode):
        """
        Calculate the loss of copy (vertex/edge) in AQG generation
        @param dec_out:             (max_tgt_len, bs, d_h)
        @param tgt_objs:            (bs, max_tgt_len), ground truth of objects at each step
        @param tgt_lens:            (bs)
        @param tgt_copy_objs:       List(bs), ground truth whether copy vertex/edge at each AddVertex/AddEdge step
        @param enc:                 List(bs), each element size (bs, n, d_h), Vertex/Edge encodings from graph encoder
        @param mode:                copy "vertex" or "edge"
        @return:   score            (bs)
        """

        bs = len(tgt_objs)
        max_tgt_len = len(dec_out)

        scores = [[] for _ in range(bs)]

        for t in range(max_tgt_len):
            op = get_operator_by_t(t)
            if (mode == "vertex" and op == "av") or (mode == "edge" and op == "ae"):
                # action_prob:             (bs, n + 1)
                action_prob = get_copy_action_probability(t=t,
                                                          dec_out=dec_out,
                                                          enc=enc,
                                                          not_copy_enc=self.not_copy_enc(mode),
                                                          copy_pointer_net=self.copy_pointer_net(mode))
                action_prob = F.softmax(action_prob, dim=-1)
                for sid in range(bs):
                    if allow_copy_at_t_for_sid(t, sid, tgt_objs, tgt_lens, mode):
                        tt = step_to_op_step(t, mode)
                        # if label == -1, select the last one (embeddings of "do not copy")
                        if tgt_copy_objs[sid][tt] == -1:
                            obj = enc[t].size(1)
                        else:
                            obj = tgt_copy_objs[sid][tt]
                        scores[sid].append(action_prob[sid, obj])

        if mode == "vertex":
            # score:    (bs)
            score = torch.stack([torch.stack(score_i, dim=0).log().sum() if len(score_i) > 0 else Variable(self.new_tensor([0]).squeeze(0))
                                 for score_i in scores], dim=0)
        else:
            score = torch.stack([torch.stack(score_i, dim=0).log().sum() if len(score_i) > 0 else Variable(self.new_tensor([0]).squeeze(0))
                                 for score_i in scores if len(score_i) > 0], dim=0)
        return -score

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
        v_ins_pool = self.encode_instance(v_ins, v_ins_lens, v_ins_classes, v_ins_sids)
        e_ins_pool = self.encode_instance(e_ins, e_ins_lens, e_ins_classes, e_ins_sids)


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
        dec_out, g_enc, v_enc, e_enc = self.outlining_decoding(q_enc=q_enc,
                                                               q_mask=q_mask_for_e if self.args.use_mention_feature else q_mask,
                                                               tgt_aqg_inputs=tgt_aqg_inputs,
                                                               tgt_lens=tgt_aqg_lens,
                                                               context=context)

        # loss:     (bs)
        loss, action_probs = self.get_outlining_loss(dec_out=dec_out,
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

    def generate(self, sample, max_beam_size=5, sparql_cache=None):
        """
        Perform Outlining in testing time.
        @param sample:              only handle one sample(question) in each time
        @param max_beam_size:       the maximum size of the beams
        @param sparql_cache:
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
        tgt_v_copy_objs, tgt_e_copy_objs, tgt_seg_switch_objs, data = sample


        bs = len(q)

        ################################### Encoding question and instances ############################################
        # v_ins_pool:      List(1), each element is a DICT {v_class_id: (total_n_v_class, d_h)}
        # e_ins_pool:      List(1), each element is a DICT {e_class_id: (total_n_e_class, d_h)}
        v_ins_pool = self.encode_instance(v_ins, v_ins_lens, v_ins_classes, v_ins_sids)
        e_ins_pool = self.encode_instance(e_ins, e_ins_lens, e_ins_classes, e_ins_sids)

        # q_enc:            (1, max_q_len, d_h)
        # q_vec:            (1, d_h)
        # q_vec_for_v:      (1, d_h)
        # q_vec_for_e:      (1, d_h)
        q_enc, \
        q_vec, q_vec_for_v, q_vec_for_e, \
        q_mask, q_mask_for_v, q_mask_for_e, context = self.encode_question(q, q_lens, q_ment)

        aqg_beams, g_enc, v_enc, e_enc = self.outlining_with_beam_search(context=context,
                                                                         q_enc=q_enc,
                                                                         q_mask=q_mask_for_e if self.args.use_mention_feature else q_mask,
                                                                         max_beam_sz=max_beam_size,
                                                                         data=data)

        assert  len(aqg_beams) == bs

        v_schedules = [0 for _ in range(bs)]
        v_finished_beams = [[] for _ in range(bs)]
        v_n_beams = [len(aqg_beams[sid]) for sid in range(bs)]

        is_finished = [False for _ in range(bs)]
        final_beams = [[] for _ in range(bs)]

        while not check_schedule(v_schedules, is_finished, v_n_beams):


            vin_context, vin_q_enc, vin_q_vec, vin_q_mask, \
            vin_ins_pool, vin_data, \
            vin_g_enc, vin_ins_enc, vin_beams = mk_input_by_schedule(context=context,
                                                                     q_enc=q_enc,
                                                                     q_vec=q_vec_for_v if self.args.use_mention_feature else q_vec,
                                                                     q_mask=q_mask_for_v if self.args.use_mention_feature else q_mask,
                                                                     ins_pool=v_ins_pool,
                                                                     data=data,
                                                                     g_enc=g_enc,
                                                                     ins_enc=v_enc,
                                                                     beams=aqg_beams,
                                                                     n_beams=v_n_beams,
                                                                     schedules=v_schedules,
                                                                     is_finished=is_finished,
                                                                     mode="vertex")

            vout_beams = self.filling_with_beam_search(context=vin_context,
                                                       in_beams=vin_beams,
                                                       q_enc=vin_q_enc,
                                                       q_vec=vin_q_vec,
                                                       q_mask=vin_q_mask,
                                                       ins_pool=vin_ins_pool,
                                                       max_beam_sz=max_beam_size,
                                                       g_enc=vin_g_enc,
                                                       ins_enc=vin_ins_enc,
                                                       data=vin_data,
                                                       mode="vertex")

            initial_finish_state(finished_beams=v_finished_beams,
                                 is_finished=is_finished)

            update_schedule(schedules=v_schedules,
                            in_beams=vin_beams,
                            out_beams=vout_beams,
                            finished_beams=v_finished_beams)

            e_schedules = [0 for _ in range(bs)]
            e_finished_beams = [[] for _ in range(bs)]
            e_n_beams = [len(v_finished_beams[sid]) for sid in range(bs)]

            while not check_schedule(e_schedules, is_finished, e_n_beams):

                ein_context, ein_q_enc, ein_q_vec, ein_q_mask, \
                ein_ins_pool, ein_data, \
                ein_g_enc, ein_ins_enc, ein_beams = mk_input_by_schedule(context=context,
                                                                         q_enc=q_enc,
                                                                         q_vec=q_vec_for_e if self.args.use_mention_feature else q_vec,
                                                                         q_mask=q_mask_for_e if self.args.use_mention_feature else q_mask,
                                                                         ins_pool=e_ins_pool,
                                                                         data=data,
                                                                         g_enc=g_enc,
                                                                         ins_enc=e_enc,
                                                                         beams=v_finished_beams,
                                                                         n_beams=e_n_beams,
                                                                         schedules=e_schedules,
                                                                         is_finished=is_finished,
                                                                         mode="edge")

                eout_beams = self.filling_with_beam_search(context=ein_context,
                                                           in_beams=ein_beams,
                                                           q_enc=ein_q_enc,
                                                           q_vec=ein_q_vec,
                                                           q_mask=ein_q_mask,
                                                           ins_pool=ein_ins_pool,
                                                           max_beam_sz=max_beam_size,
                                                           g_enc=ein_g_enc,
                                                           ins_enc=ein_ins_enc,
                                                           data=ein_data,
                                                           mode="edge")


                initial_finish_state(finished_beams=e_finished_beams,
                                     is_finished=is_finished)

                if self.args.use_kb_constraint:
                    update_schedule_with_kb_constraint(schedules=e_schedules,
                                                       in_beams=ein_beams,
                                                       out_beams=eout_beams,
                                                       finished_beams=e_finished_beams,
                                                       dataset=self.args.dataset,
                                                       kb=self.kb,
                                                       kb_endpoint=self.args.kb_endpoint)
                else:
                    update_schedule(schedules=e_schedules,
                                    in_beams=ein_beams,
                                    out_beams=eout_beams,
                                    finished_beams=e_finished_beams)

                update_finish_state(finished_beams=e_finished_beams,
                                    is_finished=is_finished,
                                    final_beams=final_beams)

                if sum(is_finished) == bs:
                    return final_beams
        return final_beams

    def outlining_with_beam_search(self, context, q_enc, q_mask, max_beam_sz, data):
        """
        @param q_enc:               (1, max_q_len, d_h)
        @param q_lens:              (1)
        @param max_beam_sz:         int, the maximum size of beam
        @param data:                List(1), original data
        @return:
        """
        t = 0
        bs = q_enc.size(0)
        beams = [Beam(sid) for sid in range(bs)]             # Initially, the beam set only consists of an empty AQG.
        completed_beams = [[] for _ in range(bs)]        # List, each element is a triple: (AQG, time_step, previous aqg id)

        n_beams = [1 for _ in range(bs)]

        g_enc = []
        v_enc = []
        e_enc = []

        dec_init_state = self.init_decoder_state(context, self.aqg_dec_cell_init)
        h_last = dec_init_state

        dec_out = []

        # if the number of completed AQG is equal with beam size, BREAK
        # or if over than the predefined operation numbers, BREAK
        while not is_complete(completed_beams, max_beam_sz) and t < self.args.max_num_op:
        # while t < self.args.max_num_op:

            # exp_q_enc:    (beam_sz, max_q_len, d_h)
            # exp_q_lens:   (beam_sz)
            exp_q_enc = expand_tensor_by_beam_number(q_enc, n_beams)
            exp_q_mask = expand_tensor_by_beam_number(q_mask, n_beams)
            exp_data = expand_data_by_beam_number(data, n_beams)

            tgt_aqg_objs, tgt_aqg_inputs = mk_tgt_from_beams(beams, self.args)
            tgt_aqg_lens = [len(x) for x in tgt_aqg_inputs]

            h_t, c_t, one_v_enc, one_e_enc, one_g_enc = self.outlining_decoding_step(t=t,
                                                                                     h_last=h_last,
                                                                                     q_enc=exp_q_enc,
                                                                                     q_mask=exp_q_mask,
                                                                                     tgt_aqg_inputs=tgt_aqg_inputs,
                                                                                     tgt_lens=tgt_aqg_lens)


            dec_out.append(h_t)

            g_enc.append(one_g_enc)
            v_enc.append(one_v_enc)
            e_enc.append(one_e_enc)

            # The possible expansions of the current beams
            meta_entries = mk_meta_entries(t=t,
                                           dec_out=dec_out,
                                           beams=beams,
                                           n_beams=n_beams,
                                           av_readout=self.av_readout,
                                           ae_readout=self.ae_readout,
                                           sv_pointer_net=self.sv_pointer_net,
                                           seg_readout=self.seg_readout,
                                           not_copy_enc=self.not_copy_enc,
                                           copy_pointer_net=self.copy_pointer_net,
                                           v_enc=v_enc,
                                           e_enc=e_enc,
                                           args=self.args,
                                           data=exp_data)

            # Get the new beams
            beams, n_beams, live_beam_ids = organize_outlining_beams(t=t,
                                                                     beams=beams,
                                                                     meta_entries=meta_entries,
                                                                     completed_beams=completed_beams,
                                                                     max_beam_sz=max_beam_sz,
                                                                     args=self.args,
                                                                     data=data)

            if not live_beam_ids:
                break

            h_last = (h_t[live_beam_ids], c_t[live_beam_ids])
            t += 1

        # sort by total probability
        for i in range(len(completed_beams)):
            completed_beams[i].sort(key=lambda x: -x.cur_aqg.get_score())
        return completed_beams, g_enc, v_enc, e_enc

    def filling_with_beam_search(self, context, in_beams, q_enc, q_vec, q_mask, ins_pool, max_beam_sz, g_enc, ins_enc, data, mode):

        t = 0
        bs = len(in_beams)
        beams = copy.deepcopy(in_beams)             # Initially, the beam set only consists of an empty AQG.
        completed_beams = [[] for _ in range(bs)]    # List, each element is a triple: (AQG, time_step, previous aqg id)

        n_beams = [1 for _ in range(bs)]
        max_aqg_obj_len = max([len(beam.pred_aqg_objs) for beam in beams])

        assert len(g_enc) == bs
        assert len(ins_enc) == bs

        if mode == "vertex":
            dec_init_state = self.init_decoder_state(context, self.v_dec_cell_init)
        else:
            dec_init_state = self.init_decoder_state(context, self.e_dec_cell_init)

        h_last = dec_init_state
        dec_out = []

        # if the number of completed AQG is equal with beam size, BREAK
        # or if over than the predefined operation numbers, BREAK
        while not is_complete(completed_beams, max_beam_sz) and t < max_aqg_obj_len:

            if not allow_filling_at_t(t, mode):
                t += 1
                continue

            exp_q_enc = expand_tensor_by_beam_number(q_enc, n_beams)
            exp_q_vec = expand_tensor_by_beam_number(q_vec, n_beams)
            exp_q_mask = expand_tensor_by_beam_number(q_mask, n_beams)
            exp_ins_pool = expand_data_by_beam_number(ins_pool, n_beams)
            exp_g_enc, exp_ins_enc = expand_graph_embedding_by_beam_number(g_enc, ins_enc, n_beams)
            exp_data = expand_data_by_beam_number(data, n_beams)

            tgt_aqgs, tgt_aqg_objs, tgt_aqg_inputs, tgt_ins_objs, tgt_copy_objs = mk_tgt_from_beams(beams, self.args, mode=mode)

            h_t, c_t, query_t = self.filling_decoding_step(t=t,
                                                           h_last=h_last,
                                                           q_enc=exp_q_enc,
                                                           q_vec=exp_q_vec,
                                                           q_mask=exp_q_mask,
                                                           tgt_objs=tgt_aqg_objs,
                                                           tgt_aqgs=tgt_aqgs,
                                                           tgt_ins_objs=tgt_ins_objs,
                                                           ins_pool=exp_ins_pool,
                                                           g_enc=exp_g_enc,
                                                           ins_enc=exp_ins_enc,
                                                           mode=mode,
                                                           args=self.args)

            dec_out.append(query_t)

            meta_entries, tmp_beams, tmp_n_beams, tmp_live_beam_ids = mk_filling_meta_entries(t=t,
                                                                                              dec_out=dec_out,
                                                                                              beams=beams,
                                                                                              n_beams=n_beams,
                                                                                              max_beam_sz=max_beam_sz,
                                                                                              completed_beams=completed_beams,
                                                                                              link_pointer_net=self.link_pointer_net(mode),
                                                                                              tgt_objs=tgt_aqg_objs,
                                                                                              tgt_copy_objs=tgt_copy_objs,
                                                                                              ins_pool=exp_ins_pool,
                                                                                              data=exp_data,
                                                                                              args=self.args,
                                                                                              kb=self.kb,
                                                                                              mode=mode)
            assert len(tmp_beams) == len(tmp_live_beam_ids)

            beams, n_beams, live_beam_ids = organize_filling_beams(t=t,
                                                                   beams=beams,
                                                                   tgt_objs=tgt_aqg_objs,
                                                                   meta_entries=meta_entries,
                                                                   completed_beams=completed_beams,
                                                                   max_beam_sz=max_beam_sz,
                                                                   data=data,
                                                                   mode=mode)

            beams, n_beams, live_beam_ids = combine_and_sort_by_sid(tmp_beams, beams,
                                                                    tmp_live_beam_ids, live_beam_ids,
                                                                    tmp_n_beams, n_beams,
                                                                    mode=mode)

            if not live_beam_ids:
                break

            h_last = (h_t[live_beam_ids], c_t[live_beam_ids])
            t += 1

        new_completed_beams = [[] for _ in range(bs)]
        for sid in range(bs):
            for beam in completed_beams[sid]:
                if check_filling(beam.cur_aqg, mode=mode):
                    new_completed_beams[sid].append(beam)
            if mode == "vertex":
                new_completed_beams[sid].sort(key=lambda x: -x.cur_aqg.get_v_score())
            else:
                new_completed_beams[sid].sort(key=lambda x: -x.cur_aqg.get_e_score())

        return new_completed_beams

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

    def decode_step(self, decoder_lstm, enc_att_linear, dec_input_linear, h_last, src_embeds, tgt_embeds, src_mask=None):
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

    def not_copy_enc(self, mode):
        assert mode in ["vertex", "edge"]
        if mode == "vertex":
            return self.not_v_copy_enc
        else:
            return self.not_e_copy_enc

    def link_pointer_net(self, mode):
        assert mode in ["vertex", "edge"]
        if mode == "vertex":
            return self.v_link_pointer_net
        else:
            return self.e_link_pointer_net

    def copy_pointer_net(self, mode):
        assert mode in ["vertex", "edge"]
        if mode == "vertex":
            return self.v_copy_pointer_net
        else:
            return self.e_copy_pointer_net

    def init_decoder_state(self, enc_last_cell, decoder_cell_init):
        """
        @param enc_last_cell:   (bs, d_h)
        """
        h_0 = decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)
        return h_0, Variable(self.new_tensor(h_0.size()).zero_())