# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : data_loaders.py
# @Software: PyCharm
"""

import os
import sys
import torch
import copy
import json
import random
import pickle
from transformers import BertTokenizer

sys.path.append("..")
from utils.utils import *
from rules.grammar import AbstractQueryGraph, get_relation_true_name, get_type_true_name, V_CLASS_IDS


class HGNetDataLoader:

    def __init__(self, args):
        self.args = args
        self.use_bert = self.args.plm_mode != "none"
        if not self.use_bert:
            self.tokenizer = pickle.load(open(self.args.wo_vocab, 'rb'))
            self.pad = self.tokenizer.lookup(self.tokenizer.pad_token)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.args.plm_mode)
            self.pad = self.tokenizer.vocab["[PAD]"]

    def process_one_data(self, d):

        if self.use_bert:
            q_toks = d["question_toks_bert"]
            ment_f_ids = d["mention_feature_bert"]
        else:
            q_toks = d["question_toks"]
            ment_f_ids = d["mention_feature"]
        q = text_to_tensor_1d(q_toks, self.tokenizer)

        if d["mention_feature"]:
            ment_f = torch.LongTensor([x for x in ment_f_ids])
        else:
            ment_f = torch.LongTensor([0 for i in range(len(q_toks))])
        match_f = torch.LongTensor(d["matching_feature"])

        gold_aqg_obj_labels = [x for x in d["aqg_obj_labels"]]
        gold_v_instance_obj_labels = [x for x in d["v_instance_obj_labels"]]
        gold_e_instance_obj_labels = [x for x in d["e_instance_obj_labels"]]
        gold_v_copy_labels = [x for x in d["v_copy_labels"]]
        gold_e_copy_labels = [x for x in d["e_copy_labels"]]
        gold_segment_switch_labels = [x for x in d["segment_switch_labels"]]

        vertex_pool = {}
        v_instance_names = []
        for v_class, v_list in d["instance_pool"]["vertex"].items():
            if self.use_bert:
                # for v_name, v_true_name in v_list:
                #     print(tokenize_word_sentence_bert(v_true_name, self.tokenizer))
                #     exit()
                vertex_pool[v_class] = [text_to_tensor_1d(tokenize_word_sentence_bert(v_true_name, self.tokenizer), self.tokenizer)
                                        for v_name, v_true_name in v_list]
            else:
                vertex_pool[v_class] = [text_to_tensor_1d(tokenize_word_sentence(v_true_name), self.tokenizer)
                                        for v_name, v_true_name in v_list]
            for v_name, v_true_name in v_list:
                v_instance_names.append([v_class, v_name, v_true_name])

        edge_pool = {}
        e_instance_names = []
        for e_class, e_list in d["instance_pool"]["edge"].items():
            if self.use_bert:
                edge_pool[e_class] = [text_to_tensor_1d(tokenize_word_sentence_bert(e_true_name, self.tokenizer), self.tokenizer)
                                      for e_name, e_true_name in e_list]
            else:
                edge_pool[e_class] = [text_to_tensor_1d(tokenize_word_sentence(e_true_name), self.tokenizer)
                                      for e_name, e_true_name in e_list]
            for e_name, e_true_name in e_list:
                e_instance_names.append([e_class, e_name, e_true_name])

        gold_aqgs = []
        gold_graphs = []
        aqg = AbstractQueryGraph()
        aqg.init_state()
        for i, obj in enumerate(gold_aqg_obj_labels):
            vertices, v_classes, v_segments, edges, e_classes, e_segments, triples = aqg.get_state()
            gold_aqgs.append(aqg)
            gold_graphs.append(mk_graph_for_gnn(vertices, v_classes, v_segments, 
                                                edges, e_classes, e_segments, triples))
            op = aqg.cur_operation
            if op == "av":
                if i == len(gold_aqg_obj_labels) - 1:
                    break
                j = step_to_av_step(i)
                v_class = obj
                v_copy = gold_v_copy_labels[j] if self.args.use_v_copy else -1
                switch_segment = gold_segment_switch_labels[j] if self.args.use_segment_embedding else False
                new_obj = [v_class, v_copy, switch_segment]
            elif op == "ae":
                j = step_to_ae_step(i)
                e_class = obj
                e_copy = gold_e_copy_labels[j] if self.args.use_e_copy else -1
                new_obj = [e_class, e_copy]
            else:
                new_obj = obj
            aqg.update_state(op, new_obj)

        return q, ment_f, match_f, \
               gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
               gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
               gold_v_copy_labels, gold_e_copy_labels, \
               gold_segment_switch_labels, \
               vertex_pool, v_instance_names,\
               edge_pool, e_instance_names, d

    def load_data(self, datas, bs, training_proportion=1.0, use_small=False, shuffle=True):

        if use_small:
            datas = datas[:1]

        if shuffle:
            random.shuffle(datas)

        data_len = int(len(datas) * training_proportion)
        datas = datas[:data_len]

        bl_x = []
        batch_index = -1  # the index of sequence batches
        sample_index = 0  # sequence index within each batch

        for d in datas:
            # if d["id"] != "WebQTest-538_49b4e9304f18a0a1cbe37bb162f61131":
            #     continue

            if sample_index % bs == 0:
                sample_index = 0
                batch_index += 1
                bl_x.append([])
            x = self.process_one_data(d)
            bl_x[batch_index].append(x)
            sample_index += 1

        self.iters = []
        self.n_batch = len(bl_x)
        for x in bl_x:
            batch = self.fix_batch(x)
            self.iters.append(batch)

    def fix_batch(self, x):

        q, ment_f, match_f, \
        gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
        gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
        gold_v_copy_labels, gold_e_copy_labels, \
        gold_segment_switch_labels, \
        vertex_pool, v_instance_names,\
        edge_pool, e_instance_names, data = zip(*x)

        q, q_lens = pad_tensor_1d(q, self.pad)
        ment_f, _ = pad_tensor_1d(ment_f, self.pad)
        match_f = torch.cat(match_f, dim=0)

        # Candidate instance pool to tensor
        v_instance_tensor, v_instance_lens, v_instance_classes, v_instance_s_ids = instance_pool_to_tensor(vertex_pool, self.pad)
        e_instance_tensor, e_instance_lens, e_instance_classes, e_instance_s_ids = instance_pool_to_tensor(edge_pool, self.pad)

        if self.args.cuda:
            q = q.to(self.args.gpu)
            ment_f = ment_f.to(self.args.gpu)
            match_f = match_f.to(self.args.gpu)
            v_instance_tensor = v_instance_tensor.to(self.args.gpu)
            e_instance_tensor = e_instance_tensor.to(self.args.gpu)
            gold_graphs = [[[y.to(self.args.gpu) for y in g] for g in s] for s in gold_graphs]

        return q, q_lens, ment_f, match_f, \
               v_instance_tensor, v_instance_lens, v_instance_classes, v_instance_s_ids, v_instance_names, \
               e_instance_tensor, e_instance_lens, e_instance_classes, e_instance_s_ids, e_instance_names, \
               gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
               gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
               gold_v_copy_labels, gold_e_copy_labels, gold_segment_switch_labels, \
               data

    def next_batch(self):
        for b in self.iters:
            yield b

class HGNetDataLoaderForPLM:

    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(self.args.plm_mode)
        self.pad = self.tokenizer.vocab["[PAD]"]
        self.sep = self.tokenizer.vocab["[SEP]"]

    def process_one_data(self, d):

        q_toks = d["question_toks_bert"]
        ment_f_ids = d["mention_feature_bert"]

        if d["mention_feature"]:
            ment_f = torch.LongTensor([x for x in ment_f_ids])
        else:
            ment_f = torch.LongTensor([0 for i in range(len(q_toks))])
        match_f = torch.LongTensor(d["matching_feature"])

        gold_aqg_obj_labels = [x for x in d["aqg_obj_labels"]]
        gold_v_instance_obj_labels = [x for x in d["v_instance_obj_labels"]]
        gold_e_instance_obj_labels = [x for x in d["e_instance_obj_labels"]]
        gold_v_copy_labels = [x for x in d["v_copy_labels"]]
        gold_e_copy_labels = [x for x in d["e_copy_labels"]]
        gold_segment_switch_labels = [x for x in d["segment_switch_labels"]]

        # input_toks = []
        # segment_ids = []
        # input_mask = []
        # q_pos = []

        input_toks = ["[CLS]"] + q_toks[1:]
        segment_ids = [0 for _ in input_toks]
        input_mask = [1 for _ in input_toks]
        q_pos = [[1, len(input_toks)]]

        v_ins_pos = {}
        v_ins_names = []
        for v_class, v_list in d["instance_pool"]["vertex"].items():
            v_ins_pos[v_class] = []
            for v_name, v_true_name in v_list:
                st = len(input_toks)
                input_toks += ["[SEP]"] + tokenize_word_sentence_bert(v_true_name, self.tokenizer, start_cls=False)
                segment_ids += [1 for _ in range(len(input_toks) - st)]
                input_mask += [1 for _ in range(len(input_toks) - st)]
                v_ins_names += [[v_class, v_name, v_true_name]]
                v_ins_pos[v_class] += [[st + 1, len(input_toks)]]

        e_ins_pos = {}
        e_ins_names = []
        for e_class, e_list in d["instance_pool"]["edge"].items():
            e_ins_pos[e_class] = []
            for e_name, e_true_name in e_list:
                st = len(input_toks)
                input_toks += ["[SEP]"] + tokenize_word_sentence_bert(e_true_name, self.tokenizer, start_cls=False)
                segment_ids += [1 for _ in range(len(input_toks) - st)]
                input_mask += [1 for _ in range(len(input_toks) - st)]
                e_ins_names += [[e_class, e_name, e_true_name]]
                e_ins_pos[e_class] += [[st + 1, len(input_toks)]]

        # print(input_toks)
        # print(segment_ids)
        # print(input_mask)
        # print(len(input_toks) == len(segment_ids) == len(input_mask))
        # print(d["instance_pool"]["vertex"])
        # print(d["instance_pool"]["edge"])
        # for v_class, content in v_ins_pos.items():
        #     for st, ed in content:
        #         print(input_toks[st:ed])
        # for e_class, content in e_ins_pos.items():
        #     for st, ed in content:
        #         print(input_toks[st:ed])
        # exit()

        input_ids = text_to_tensor_1d(input_toks, self.tokenizer)
        segment_ids = torch.LongTensor(segment_ids)
        input_mask = torch.LongTensor(input_mask)

        gold_aqgs = []
        gold_graphs = []
        aqg = AbstractQueryGraph()
        aqg.init_state()
        for i, obj in enumerate(gold_aqg_obj_labels):
            vertices, v_classes, v_segments, edges, e_classes, e_segments, triples = aqg.get_state()
            gold_aqgs.append(aqg)
            gold_graphs.append(mk_graph_for_gnn(vertices, v_classes, v_segments,
                                                edges, e_classes, e_segments, triples))
            op = aqg.cur_operation
            if op == "av":
                if i == len(gold_aqg_obj_labels) - 1:
                    break
                j = step_to_av_step(i)
                v_class = obj
                v_copy = gold_v_copy_labels[j] if self.args.use_v_copy else -1
                switch_segment = gold_segment_switch_labels[j] if self.args.use_segment_embedding else False
                new_obj = [v_class, v_copy, switch_segment]
            elif op == "ae":
                j = step_to_ae_step(i)
                e_class = obj
                e_copy = gold_e_copy_labels[j] if self.args.use_e_copy else -1
                new_obj = [e_class, e_copy]
            else:
                new_obj = obj
            aqg.update_state(op, new_obj)

        return input_ids, segment_ids, input_mask, \
               gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
               gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
               gold_v_copy_labels, gold_e_copy_labels, \
               gold_segment_switch_labels, \
               q_pos, ment_f, match_f, \
               v_ins_pos, v_ins_names,\
               e_ins_pos, e_ins_names, d

    def load_data(self, datas, bs, training_proportion=1.0, use_small=False, shuffle=True):

        if use_small:
            datas = datas[:20]

        if shuffle:
            random.shuffle(datas)

        data_len = int(len(datas) * training_proportion)
        datas = datas[:data_len]

        bl_x = []
        batch_index = -1  # the index of sequence batches
        sample_index = 0  # sequence index within each batch

        for d in datas:
            # if d["id"] != "WebQTest-832_c334509bb5e02cacae1ba2e80c176499":
            #     continue

            if sample_index % bs == 0:
                sample_index = 0
                batch_index += 1
                bl_x.append([])
            x = self.process_one_data(d)
            bl_x[batch_index].append(x)
            sample_index += 1

        self.iters = []
        self.n_batch = len(bl_x)
        for x in bl_x:
            batch = self.fix_batch(x)
            self.iters.append(batch)

    def fix_batch(self, x):

        input_ids, segment_ids, _, \
        gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
        gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
        gold_v_copy_labels, gold_e_copy_labels, \
        gold_segment_switch_labels, \
        q_pos, ment_f, match_f, \
        v_ins_pos, v_ins_names, \
        e_ins_pos, e_ins_names, data = zip(*x)

        input_ids, input_lens = pad_tensor_1d(input_ids, self.pad)
        segment_ids, _ = pad_tensor_1d(segment_ids, self.pad)
        input_mask = length_array_to_mask_tensor(input_lens, reverse=False)

        # print(input_ids.size())
        # print(input_lens)
        # print(segment_ids.size())
        # print(segment_ids)
        # print(input_mask.size())
        # print(input_mask)
        # exit()

        ment_f, _ = pad_tensor_1d(ment_f, self.pad)
        match_f = torch.cat(match_f, dim=0)

        # # Candidate instance pool to tensor
        # v_instance_tensor, v_instance_lens, v_instance_classes, v_instance_s_ids = instance_pool_to_tensor_for_plm(vertex_pool, self.pad, self.sep)
        # e_instance_tensor, e_instance_lens, e_instance_classes, e_instance_s_ids = instance_pool_to_tensor_for_plm(edge_pool, self.pad, self.sep)
        # exit()

        if self.args.cuda:
            input_ids = input_ids.to(self.args.gpu)
            segment_ids = segment_ids.to(self.args.gpu)
            input_mask = input_mask.to(self.args.gpu)
            ment_f = ment_f.to(self.args.gpu)
            match_f = match_f.to(self.args.gpu)
            gold_graphs = [[[y.to(self.args.gpu) for y in g] for g in s] for s in gold_graphs]

        return input_ids, segment_ids, input_mask,\
               q_pos, ment_f, match_f, \
               v_ins_pos, v_ins_names, \
               e_ins_pos, e_ins_names, \
               gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
               gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
               gold_v_copy_labels, gold_e_copy_labels, gold_segment_switch_labels, \
               data

    def next_batch(self):
        for b in self.iters:
            yield b


class NonHierarchicalGenerationDataLoader:

    def __init__(self, args):
        self.args = args
        if not self.args.use_bert:
            self.tokenizer = pickle.load(open(self.args.wo_vocab, 'rb'))
            self.pad = self.tokenizer.lookup(self.tokenizer.pad_token)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_mode)
            self.pad = self.tokenizer.vocab["[PAD]"]

    def process_one_data(self, d):

        if self.args.use_bert:
            q_toks = d["q_toks_bert"]
            ment_f_ids = d["mention_feature_bert"]
        else:
            q_toks = d["q_toks"]
            ment_f_ids = d["mention_feature"]
        q = text_to_tensor_1d(q_toks, self.tokenizer)

        if d["mention_feature"]:
            ment_f = torch.LongTensor([x for x in ment_f_ids])
        else:
            ment_f = torch.LongTensor([0 for i in range(len(q_toks))])
        match_f = torch.LongTensor(d["matching_feature"])

        gold_aqg_obj_labels = [x for x in d["aqg_obj_labels"]]
        gold_v_instance_obj_labels = [x for x in d["v_instance_obj_labels"]]
        gold_e_instance_obj_labels = [x for x in d["e_instance_obj_labels"]]
        gold_v_copy_labels = [x for x in d["v_copy_labels"]]
        gold_e_copy_labels = [x for x in d["e_copy_labels"]]
        gold_segment_switch_labels = [x for x in d["segment_switch_labels"]]

        d["instance_pool"]["vertex"][V_CLASS_IDS["ans"]] = [["ans", "answer"]]
        d["instance_pool"]["vertex"][V_CLASS_IDS["var"]] = [["var", "variable"]]
        d["instance_pool"]["vertex"][V_CLASS_IDS["end"]] = [["end", "end"]]

        v_st_idx = {}
        vertex_pool = []
        v_instance_names = []
        for v_class, v_list in d["instance_pool"]["vertex"].items():
            v_st_idx[v_class] = len(vertex_pool)
            if self.args.use_bert:
                vertex_pool += [text_to_tensor_1d(tokenize_word_sentence_bert(v_true_name, self.tokenizer), self.tokenizer)
                                for v_name, v_true_name in v_list]
            else:
                vertex_pool += [text_to_tensor_1d(tokenize_word_sentence(v_true_name), self.tokenizer)
                                for v_name, v_true_name in v_list]
            for v_name, v_true_name in v_list:
                v_instance_names.append([v_class, v_name, v_true_name])

        e_st_idx = {}
        edge_pool = []
        e_instance_names = []
        for e_class, e_list in d["instance_pool"]["edge"].items():
            assert e_class % 2 == 0
            e_st_idx[e_class] = len(edge_pool)
            if self.args.use_bert:
                for e_name, e_true_name in e_list:
                    edge_pool.append(text_to_tensor_1d(tokenize_word_sentence_bert(e_true_name + " +", self.tokenizer), self.tokenizer))
                    edge_pool.append(text_to_tensor_1d(tokenize_word_sentence_bert(e_true_name + " -", self.tokenizer), self.tokenizer))
            else:
                for e_name, e_true_name in e_list:
                    edge_pool.append(text_to_tensor_1d(tokenize_word_sentence(e_true_name + " +"), self.tokenizer))
                    edge_pool.append(text_to_tensor_1d(tokenize_word_sentence(e_true_name + " -"), self.tokenizer))
            for e_name, e_true_name in e_list:
                e_instance_names.append([e_class, e_name, e_true_name + " +"])
                e_instance_names.append([e_class + 1, e_name, e_true_name + " -"])

        v_instance_name2id = {v_name: i for i, (v_class, v_name, v_true_name) in enumerate(v_instance_names)}
        e_instance_name2id = {e_name: i // 2 for i, (e_class, e_name, e_true_name) in enumerate(e_instance_names)}

        assert len(e_instance_name2id) * 2 == len(e_instance_names)

        gold_obj_labels = []
        id_v = 0
        id_e = 0
        for t in range(len(gold_aqg_obj_labels)):
            if t == 0 or t % 3 == 1:
                v_class = gold_aqg_obj_labels[t]
                if v_class in v_st_idx:
                    if v_class in [V_CLASS_IDS["ans"], V_CLASS_IDS["var"], V_CLASS_IDS["end"]]:
                        obj = v_st_idx[v_class]
                    else:
                        obj = gold_v_instance_obj_labels[id_v] + v_st_idx[v_class]
                else:
                    obj = -1
                id_v += 1
            elif t % 3 == 0:
                e_class = gold_aqg_obj_labels[t]
                _e_class = e_class - 1 if e_class % 2 == 1 else e_class
                if _e_class in e_st_idx:
                    if e_class % 2 == 0:
                        obj = 2 * gold_e_instance_obj_labels[id_e] + e_st_idx[_e_class]
                    else:
                        obj = 2 * gold_e_instance_obj_labels[id_e] + 1 + e_st_idx[_e_class]
                else:
                    obj = -1
                id_e += 1
            else:
                obj = gold_aqg_obj_labels[t]
            gold_obj_labels.append(obj)

        gold_aqgs = []
        gold_graphs = []
        aqg = AbstractQueryGraph()
        aqg.init_state()
        for i, obj in enumerate(gold_aqg_obj_labels):
            _obj = gold_obj_labels[i]
            vertices, v_classes, v_segments, edges, e_classes, e_segments, triples = aqg.get_state()

            v_tensor, v_class_tensor, v_segment_tensor, \
            e_tensor, e_class_tensor, e_segment_tensor, adj = mk_graph_for_gnn(vertices, v_classes, v_segments,
                                                                               edges, e_classes, e_segments, triples)

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

            gold_aqgs.append(aqg)
            gold_graphs.append([v_tensor, v_ins_tensor, v_segment_tensor,
                                e_tensor, e_ins_tensor, e_segment_tensor, adj])
            op = aqg.cur_operation
            if op == "av":
                if i == len(gold_aqg_obj_labels) - 1:
                    break
                j = step_to_av_step(i)
                v_id = len(aqg.vertices)
                v_class = obj
                v_copy = gold_v_copy_labels[j] if self.args.use_v_copy else -1
                switch_segment = gold_segment_switch_labels[j] if self.args.use_segment_embedding else False
                new_obj = [v_class, v_copy, switch_segment]
                aqg.update_state(op, new_obj)
                aqg.set_vertex_instance(v_id, [_obj, v_instance_names[_obj][1]])
            elif op == "ae":
                j = step_to_ae_step(i)
                e_id = len(aqg.edges)
                e_class = obj
                e_copy = gold_e_copy_labels[j] if self.args.use_e_copy else -1
                new_obj = [e_class, e_copy]
                aqg.update_state(op, new_obj)
                aqg.set_edge_instance(e_id, [_obj, e_instance_names[_obj][1]])
                aqg.set_edge_instance(get_inv_edge(e_id), [_obj, e_instance_names[_obj][1]])
            else:
                new_obj = obj
                aqg.update_state(op, new_obj)

        return q, ment_f, match_f, \
               gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
               gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
               gold_v_copy_labels, gold_e_copy_labels, \
               gold_segment_switch_labels, \
               gold_obj_labels, \
               vertex_pool, v_instance_names,\
               edge_pool, e_instance_names, \
               v_st_idx, e_st_idx, v_instance_name2id, e_instance_name2id, d

    def load_data(self, datas, bs, use_small=False, shuffle=True):

        if use_small:
            datas = datas[:10]

        if shuffle:
            random.shuffle(datas)

        bl_x = []
        batch_index = -1  # the index of sequence batches
        sample_index = 0  # sequence index within each batch

        for d in datas:

            # if d["id"] != 9:
            #     continue

            if sample_index % bs == 0:
                sample_index = 0
                batch_index += 1
                bl_x.append([])
            x = self.process_one_data(d)
            bl_x[batch_index].append(x)
            sample_index += 1

        self.iters = []
        self.n_batch = len(bl_x)
        for x in bl_x:
            batch = self.fix_batch(x)
            self.iters.append(batch)

    def fix_batch(self, x):

        q, ment_f, match_f, \
        gold_aqgs, gold_graphs, gold_aqg_obj_labels, \
        gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
        gold_v_copy_labels, gold_e_copy_labels, \
        gold_segment_switch_labels, \
        gold_obj_labels, \
        vertex_pool, v_instance_names,\
        edge_pool, e_instance_names, \
        v_st_idx, e_st_idx, v_instance_name2id, e_instance_name2id, data = zip(*x)

        q, q_lens = pad_tensor_1d(q, self.pad)
        ment_f, _ = pad_tensor_1d(ment_f, self.pad)
        match_f = torch.cat(match_f, dim=0)

        v_instance_tensor = []
        v_instance_s_ids = []
        for s_id, one_pool in enumerate(vertex_pool):
            for _instance in one_pool:
                v_instance_tensor.append(_instance)
                v_instance_s_ids.append(s_id)
        v_instance_tensor, v_instance_lens = pad_tensor_1d(v_instance_tensor, self.pad)

        e_instance_tensor = []
        e_instance_s_ids = []
        for s_id, one_pool in enumerate(edge_pool):
            for _instance in one_pool:
                e_instance_tensor.append(_instance)
                e_instance_s_ids.append(s_id)
        e_instance_tensor, e_instance_lens = pad_tensor_1d(e_instance_tensor, self.pad)

        if self.args.cuda:
            q = q.to(self.args.gpu)
            ment_f = ment_f.to(self.args.gpu)
            match_f = match_f.to(self.args.gpu)
            v_instance_tensor = v_instance_tensor.to(self.args.gpu)
            e_instance_tensor = e_instance_tensor.to(self.args.gpu)
            gold_graphs = [[[y.to(self.args.gpu) for y in g] for g in s] for s in gold_graphs]

        return q, q_lens, ment_f, match_f, \
               v_instance_tensor, v_instance_lens, v_instance_s_ids, v_instance_names, \
               e_instance_tensor, e_instance_lens, e_instance_s_ids, e_instance_names, \
               gold_aqgs, gold_graphs, gold_aqg_obj_labels, gold_obj_labels, \
               gold_v_instance_obj_labels, gold_e_instance_obj_labels, \
               gold_v_copy_labels, gold_e_copy_labels, gold_segment_switch_labels, \
               v_st_idx, e_st_idx, v_instance_name2id, e_instance_name2id, \
               data

    def next_batch(self):
        for b in self.iters:
            yield b


class RelationRankingDataLoader:

    def __init__(self, args, mode):
        self.args = args
        self.dataset = args.dataset

        assert mode in ["train", "dev", "test"]

        if args.dataset == "lcq":
            self.kb = "dbpedia"
        else:
            self.kb = "freebase"

        self.wo_vocab = pickle.load(open(self.args.wo_vocab, 'rb'))
        self.wo_pad = self.wo_vocab.lookup(self.wo_vocab.pad_token)

        if self.dataset == "lcq":

            def dfs_load(dir):
                datas = []
                files = os.listdir(dir)
                for file in files:
                    file = os.path.join(dir, file)
                    if not os.path.isdir(file):
                        data = json.load(open(file, "r"))
                        datas.append(data)
                    else:
                        datas.extend(dfs_load(file))
                return datas

            if mode == "train":
                cand_pool_dir = "../../data/LC-QuAD/individual_relation_pool_train/"
                datas = dfs_load(cand_pool_dir)
            elif mode == "dev":
                cand_pool_dir = "../../data/LC-QuAD/individual_relation_pool_dev/"
                datas = dfs_load(cand_pool_dir)
            else:
                cand_pool_dir = "../../data/LC-QuAD/individual_relation_pool_test/"
                datas = dfs_load(cand_pool_dir)

            self.cand_pool = {}
            for d in datas:
                tmp_pool = set(["<" + x + ">" for x in d["relation_pool"]])
                tmp_pool.add("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
                self.cand_pool[d["id"]] = [x for x in tmp_pool]
                self.cand_pool[d["id"]].sort()
        else:
            self.cand_pool = json.load(open(self.args.rel_pool, "r"))

    def process_one_data(self, d, training=False):

        q = text_to_tensor_1d(d["question_toks"], self.wo_vocab)
        pos_r = text_to_tensor_1d(d["positive_relation"], self.wo_vocab)

        if d["mention_feature"]:
            ment_f = torch.LongTensor([x for x in d["mention_feature"]])
        else:
            ment_f = torch.LongTensor([0 for i in range(len(d["q_toks"]))])


        if self.dataset == "lcq":
            cand_rels = [r for r in self.cand_pool[d["id"]]]
            if training:
                cand_rels = [r for r in cand_rels if r not in d["gold_relations"]]

                for i in range(len(cand_rels), self.args.ns):
                    tmp = random.randint(0, len(self.cand_pool[d["id"]]) - 1)
                    cand_rels.append(self.cand_pool[d["id"]][tmp])

                cand_rels = random.sample(cand_rels, self.args.ns)

        else:
            cand_rels = [r for r in self.cand_pool]
            if training:
                cand_rels = [r for r in cand_rels if r not in d["gold_relations"]]
                for i in range(len(cand_rels), self.args.ns):
                    tmp = random.randint(0, len(self.cand_pool) - 1)
                    while self.cand_pool[tmp] in cand_rels:
                        tmp = random.randint(0, len(self.cand_pool) - 1)
                    cand_rels.append(self.cand_pool[tmp])

                cand_rels = random.sample(cand_rels, self.args.ns)

        cand_rels = [tokenize_word_sentence(get_relation_true_name(r, kb=self.kb)) for r in cand_rels]
        neg_r, neg_r_lens = text_to_tensor_2d(cand_rels, self.wo_vocab)
        return q, ment_f, pos_r, neg_r, neg_r_lens, d

    def load_data(self, datas, bs, use_small=False, shuffle=True, training=False):

        new_datas = []
        for d in datas:
            conds = [c for c in d["query"]["where"]["union"]] + d["query"]["where"]["notUnion"]
            if len(d["query"]["where"]["subQueries"]) == 1:
                conds += d["query"]["where"]["subQueries"][0]["where"]["notUnion"]
            pos_rels = [c[1][1].strip("<").strip(">") for c in conds if c[0] == "Triple"]

            pos_rels = list(set(pos_rels))
            pos_rels.sort()
            pos_rels_toks = [tokenize_word_sentence(get_relation_true_name(r, kb=self.kb)) for r in pos_rels]
            pos_num = len(pos_rels_toks) if training else 1
            for r in pos_rels_toks[:pos_num]:
                new_d = copy.deepcopy(d)
                new_d["positive_relation"] = r
                new_d["gold_relations"] = [_r for _r in pos_rels]
                new_datas.append(new_d)

        datas = [d for d in new_datas]

        if use_small:
            datas = datas[:10]

        if shuffle:
            random.shuffle(datas)

        bl_x = []
        batch_index = -1  # the index of sequence batches
        sample_index = 0  # sequence index within each batch

        for d in datas:
            # if d["id"] != 4109:
            #     continue
            # print(d["id"])
            if sample_index % bs == 0:
                sample_index = 0
                batch_index += 1
                bl_x.append([])
            x = self.process_one_data(d, training=training)
            bl_x[batch_index].append(x)
            sample_index += 1
        self.iters = []
        self.n_batch = len(bl_x)
        for x in bl_x:
            batch = self.fix_batch(x)
            self.iters.append(batch)

    def fix_batch(self, x):

        q, ment_f, pos_r, neg_r, neg_r_lens, data = zip(*x)

        q, q_lens = pad_tensor_1d(q, self.wo_pad)
        pos_r, pos_r_lens = pad_tensor_1d(pos_r, self.wo_pad)
        ment_f, _ = pad_tensor_1d(ment_f, self.wo_pad)

        neg_r, neg_r_blens = pad_tensor_2d(neg_r, self.wo_pad)
        neg_r_lens = torch.cat(neg_r_lens, 0)

        if self.args.cuda:
            device = self.args.gpu
            return q.to(device), q_lens, ment_f.to(device), \
                   pos_r.to(device), pos_r_lens, \
                   neg_r.to(device), neg_r_lens, neg_r_blens, data
        else:
            return q, q_lens, ment_f, \
                   pos_r, pos_r_lens, \
                   neg_r, neg_r_lens, neg_r_blens, data

    def next_batch(self):
        for b in self.iters:
            yield b
