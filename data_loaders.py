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
import json
import pickle
from transformers import AutoTokenizer
sys.path.append("..")
from utils.utils import *
from rules.grammar import AbstractQueryGraph, get_relation_true_name, V_CLASS_IDS


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.use_plm = self.args.plm_mode != "none"
        if not self.use_plm:
            self.tokenizer = pickle.load(open(self.args.vocab_path, 'rb'))
            self.pad = self.tokenizer.lookup(self.tokenizer.pad_token)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.plm_mode)
            self.pad = self.tokenizer.vocab["[PAD]"]

    def process_one_data(self, d):
        pass
    
    def fix_batch(self, x):
        pass

    def _load_data(self, datas, bs, training_proportion=1.0, use_small=False, shuffle=True):

        if use_small:
            datas = datas[:10]

        if shuffle:
            random.shuffle(datas)

        data_len = int(len(datas) * training_proportion)
        datas = datas[:data_len]

        bl_x = []
        batch_index = -1  # the index of sequence batches
        sample_index = 0  # sequence index within each batch

        for d in datas:
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

    def next_batch(self):
        for b in self.iters:
            yield b


class HGNetDataLoader(DataLoader):

    def __init__(self, args):
        super(HGNetDataLoader, self).__init__(args)

    def process_one_data(self, d):
        if self.use_plm:
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

        tgt_aqg_objs = [x for x in d["aqg_obj_labels"]]
        tgt_v_ins_objs = [x for x in d["v_instance_obj_labels"]]
        tgt_e_ins_objs = [x for x in d["e_instance_obj_labels"]]
        tgt_v_copy_objs = [x for x in d["v_copy_labels"]]
        tgt_e_copy_objs = [x for x in d["e_copy_labels"]]
        tgt_seg_switch_objs = [x for x in d["segment_switch_labels"]]

        vertex_pool = {}
        v_ins_names = []
        for v_class, v_list in d["instance_pool"]["vertex"].items():
            if self.use_plm:
                vertex_pool[v_class] = [text_to_tensor_1d(tokenize_word_sentence_plm(v_true_name, self.tokenizer), self.tokenizer)
                                        for v_name, v_true_name in v_list]
            else:
                vertex_pool[v_class] = [text_to_tensor_1d(tokenize_word_sentence(v_true_name), self.tokenizer)
                                        for v_name, v_true_name in v_list]
            for v_name, v_true_name in v_list:
                v_ins_names.append([v_class, v_name, v_true_name])

        edge_pool = {}
        e_ins_names = []
        for e_class, e_list in d["instance_pool"]["edge"].items():
            if self.use_plm:
                edge_pool[e_class] = [text_to_tensor_1d(tokenize_word_sentence_plm(e_true_name, self.tokenizer), self.tokenizer)
                                      for e_name, e_true_name in e_list]
            else:
                edge_pool[e_class] = [text_to_tensor_1d(tokenize_word_sentence(e_true_name), self.tokenizer)
                                      for e_name, e_true_name in e_list]
            for e_name, e_true_name in e_list:
                e_ins_names.append([e_class, e_name, e_true_name])

        tgt_aqgs = []
        tgt_aqg_inputs = []
        aqg = AbstractQueryGraph()
        aqg.init_state()
        for i, obj in enumerate(tgt_aqg_objs):
            vertices, v_classes, v_segments, edges, e_classes, e_segments, triples = aqg.get_state()
            tgt_aqgs.append(aqg)
            tgt_aqg_inputs.append(mk_graph_for_gnn(vertices, v_classes, v_segments, 
                                                edges, e_classes, e_segments, triples))
            op = aqg.cur_operation
            if op == "av":
                if i == len(tgt_aqg_objs) - 1:
                    break
                j = step_to_av_step(i)
                v_class = obj
                v_copy = tgt_v_copy_objs[j] if self.args.use_v_copy else -1
                switch_segment = tgt_seg_switch_objs[j] if self.args.use_segment_embedding else False
                new_obj = [v_class, v_copy, switch_segment]
            elif op == "ae":
                j = step_to_ae_step(i)
                e_class = obj
                e_copy = tgt_e_copy_objs[j] if self.args.use_e_copy else -1
                new_obj = [e_class, e_copy]
            else:
                new_obj = obj
            aqg.update_state(op, new_obj)

        return q, ment_f, match_f, \
               tgt_aqgs, tgt_aqg_inputs, tgt_aqg_objs, \
               tgt_v_ins_objs, tgt_e_ins_objs, \
               tgt_v_copy_objs, tgt_e_copy_objs, \
               tgt_seg_switch_objs, \
               vertex_pool, v_ins_names,\
               edge_pool, e_ins_names, d

    def fix_batch(self, x):
        q, ment_f, match_f, \
        tgt_aqgs, tgt_aqg_inputs, tgt_aqg_objs, \
        tgt_v_ins_objs, tgt_e_ins_objs, \
        tgt_v_copy_objs, tgt_e_copy_objs, \
        tgt_seg_switch_objs, \
        vertex_pool, v_ins_names,\
        edge_pool, e_ins_names, data = zip(*x)

        q, q_lens = pad_tensor_1d(q, self.pad)
        ment_f, _ = pad_tensor_1d(ment_f, self.pad)
        match_f = torch.cat(match_f, dim=0)

        # Candidate instance pool to tensor
        v_ins, v_ins_lens, v_ins_classes, v_ins_sids = instance_pool_to_tensor(vertex_pool, self.pad)
        e_ins, e_ins_lens, e_ins_classes, e_ins_sids = instance_pool_to_tensor(edge_pool, self.pad)

        if self.args.cuda:
            q = q.to(self.args.gpu)
            ment_f = ment_f.to(self.args.gpu)
            match_f = match_f.to(self.args.gpu)
            v_ins = v_ins.to(self.args.gpu)
            e_ins = e_ins.to(self.args.gpu)
            tgt_aqg_inputs = [[[y.to(self.args.gpu) for y in g] for g in s] for s in tgt_aqg_inputs]

        return q, q_lens, ment_f, match_f, \
               v_ins, v_ins_lens, v_ins_classes, v_ins_sids, v_ins_names, \
               e_ins, e_ins_lens, e_ins_classes, e_ins_sids, e_ins_names, \
               tgt_aqgs, tgt_aqg_inputs, tgt_aqg_objs, \
               tgt_v_ins_objs, tgt_e_ins_objs, \
               tgt_v_copy_objs, tgt_e_copy_objs, tgt_seg_switch_objs, \
               data
    
    def load_data(self, datas, bs, training_proportion=1.0, use_small=False, shuffle=True):
        self._load_data(datas, bs, training_proportion, use_small, shuffle)


class NHGGDataLoader(DataLoader):

    def __init__(self, args):
        super(NHGGDataLoader, self).__init__(args)

    def process_one_data(self, d):
        if self.use_plm:
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

        tgt_aqg_objs = [x for x in d["aqg_obj_labels"]]
        tgt_v_ins_objs = [x for x in d["v_instance_obj_labels"]]
        tgt_e_ins_objs = [x for x in d["e_instance_obj_labels"]]
        tgt_v_copy_objs = [x for x in d["v_copy_labels"]]
        tgt_e_copy_objs = [x for x in d["e_copy_labels"]]
        tgt_seg_switch_objs = [x for x in d["segment_switch_labels"]]

        d["instance_pool"]["vertex"][V_CLASS_IDS["ans"]] = [["ans", "answer"]]
        d["instance_pool"]["vertex"][V_CLASS_IDS["var"]] = [["var", "variable"]]
        d["instance_pool"]["vertex"][V_CLASS_IDS["end"]] = [["end", "end"]]

        v_st_idx = {}
        vertex_pool = []
        v_ins_names = []
        for v_class, v_list in d["instance_pool"]["vertex"].items():
            v_st_idx[v_class] = len(vertex_pool)
            if self.use_plm:
                vertex_pool += [text_to_tensor_1d(tokenize_word_sentence_plm(v_true_name, self.tokenizer), self.tokenizer) for v_name, v_true_name in v_list]
            else:
                vertex_pool += [text_to_tensor_1d(tokenize_word_sentence(v_true_name), self.tokenizer) for v_name, v_true_name in v_list]
            for v_name, v_true_name in v_list:
                v_ins_names.append([v_class, v_name, v_true_name])

        e_st_idx = {}
        edge_pool = []
        e_ins_names = []
        for e_class, e_list in d["instance_pool"]["edge"].items():
            assert e_class % 2 == 0
            e_st_idx[e_class] = len(edge_pool)
            if self.use_plm:
                for e_name, e_true_name in e_list:
                    edge_pool.append(text_to_tensor_1d(tokenize_word_sentence_plm(e_true_name + " +", self.tokenizer), self.tokenizer))
                    edge_pool.append(text_to_tensor_1d(tokenize_word_sentence_plm(e_true_name + " -", self.tokenizer), self.tokenizer))
            else:
                for e_name, e_true_name in e_list:
                    edge_pool.append(text_to_tensor_1d(tokenize_word_sentence(e_true_name + " +"), self.tokenizer))
                    edge_pool.append(text_to_tensor_1d(tokenize_word_sentence(e_true_name + " -"), self.tokenizer))
            for e_name, e_true_name in e_list:
                e_ins_names.append([e_class, e_name, e_true_name + " +"])
                e_ins_names.append([e_class + 1, e_name, e_true_name + " -"])

        v_ins_name2id = {v_name: i for i, (v_class, v_name, v_true_name) in enumerate(v_ins_names)}
        e_ins_name2id = {e_name: i // 2 for i, (e_class, e_name, e_true_name) in enumerate(e_ins_names)}

        assert len(e_ins_name2id) * 2 == len(e_ins_names)

        tgt_nhgg_objs = []
        id_v = 0
        id_e = 0
        for t in range(len(tgt_aqg_objs)):
            if t == 0 or t % 3 == 1:
                v_class = tgt_aqg_objs[t]
                if v_class in v_st_idx:
                    if v_class in [V_CLASS_IDS["ans"], V_CLASS_IDS["var"], V_CLASS_IDS["end"]]:
                        obj = v_st_idx[v_class]
                    else:
                        obj = tgt_v_ins_objs[id_v] + v_st_idx[v_class]
                else:
                    obj = -1
                id_v += 1
            elif t % 3 == 0:
                e_class = tgt_aqg_objs[t]
                _e_class = e_class - 1 if e_class % 2 == 1 else e_class
                if _e_class in e_st_idx:
                    if e_class % 2 == 0:
                        obj = 2 * tgt_e_ins_objs[id_e] + e_st_idx[_e_class]
                    else:
                        obj = 2 * tgt_e_ins_objs[id_e] + 1 + e_st_idx[_e_class]
                else:
                    obj = -1
                id_e += 1
            else:
                obj = tgt_aqg_objs[t]
            tgt_nhgg_objs.append(obj)

        tgt_aqgs = []
        tgt_aqg_inputs = []
        aqg = AbstractQueryGraph()
        aqg.init_state()
        for i, obj in enumerate(tgt_aqg_objs):
            _obj = tgt_nhgg_objs[i]
            vertices, v_classes, v_segments, edges, e_classes, e_segments, triples = aqg.get_state()

            v_tensor, v_class_tensor, v_segment_tensor, \
            e_tensor, e_class_tensor, e_segment_tensor, adj = mk_graph_for_gnn(vertices, v_classes, v_segments,
                                                                               edges, e_classes, e_segments, triples)

            v_ins = []
            for v in vertices:
                v_class = aqg.get_vertex_label(v)
                if v_class == V_CLASS_IDS["var"]:
                    v_ins.append(v_ins_name2id["var"])
                elif v_class == V_CLASS_IDS["ans"]:
                    v_ins.append(v_ins_name2id["ans"])
                else:
                    v_ins.append(v_ins_name2id[aqg.get_vertex_instance(v)[-1]])
            v_ins_tensor = torch.LongTensor(v_ins)

            e_ins = []
            for e in edges:
                e_class = aqg.get_edge_label(e)
                if e_class % 2 == 0:
                    e_ins.append(2 * e_ins_name2id[aqg.get_edge_instance(e)[-1]])
                else:
                    e_ins.append(2 * e_ins_name2id[aqg.get_edge_instance(e)[-1]] + 1)
            e_ins_tensor = torch.LongTensor(e_ins)

            tgt_aqgs.append(aqg)
            tgt_aqg_inputs.append([v_tensor, v_ins_tensor, v_segment_tensor,
                                e_tensor, e_ins_tensor, e_segment_tensor, adj])
            op = aqg.cur_operation
            if op == "av":
                if i == len(tgt_aqg_objs) - 1:
                    break
                j = step_to_av_step(i)
                v_id = len(aqg.vertices)
                v_class = obj
                v_copy = tgt_v_copy_objs[j] if self.args.use_v_copy else -1
                switch_segment = tgt_seg_switch_objs[j] if self.args.use_segment_embedding else False
                new_obj = [v_class, v_copy, switch_segment]
                aqg.update_state(op, new_obj)
                aqg.set_vertex_instance(v_id, [_obj, v_ins_names[_obj][1]])
            elif op == "ae":
                j = step_to_ae_step(i)
                e_id = len(aqg.edges)
                e_class = obj
                e_copy = tgt_e_copy_objs[j] if self.args.use_e_copy else -1
                new_obj = [e_class, e_copy]
                aqg.update_state(op, new_obj)
                aqg.set_edge_instance(e_id, [_obj, e_ins_names[_obj][1]])
                aqg.set_edge_instance(get_inv_edge(e_id), [_obj, e_ins_names[_obj][1]])
            else:
                new_obj = obj
                aqg.update_state(op, new_obj)

        return q, ment_f, match_f, \
               tgt_aqgs, tgt_aqg_inputs, tgt_aqg_objs, \
               tgt_v_ins_objs, tgt_e_ins_objs, \
               tgt_v_copy_objs, tgt_e_copy_objs, \
               tgt_seg_switch_objs, \
               tgt_nhgg_objs, \
               vertex_pool, v_ins_names, \
               edge_pool, e_ins_names, \
               v_st_idx, e_st_idx, v_ins_name2id, e_ins_name2id, d

    def fix_batch(self, x):

        q, ment_f, match_f, \
        tgt_aqgs, tgt_aqg_inputs, tgt_aqg_objs, \
        tgt_v_ins_objs, tgt_e_ins_objs, \
        tgt_v_copy_objs, tgt_e_copy_objs, \
        tgt_seg_switch_objs, \
        tgt_nhgg_objs, \
        vertex_pool, v_ins_names,\
        edge_pool, e_ins_names, \
        v_st_idx, e_st_idx, v_ins_name2id, e_ins_name2id, data = zip(*x)

        q, q_lens = pad_tensor_1d(q, self.pad)
        ment_f, _ = pad_tensor_1d(ment_f, self.pad)
        match_f = torch.cat(match_f, dim=0)

        v_ins = []
        v_ins_sids = []
        for s_id, one_pool in enumerate(vertex_pool):
            for _instance in one_pool:
                v_ins.append(_instance)
                v_ins_sids.append(s_id)
        v_ins, v_ins_lens = pad_tensor_1d(v_ins, self.pad)

        e_ins = []
        e_ins_sids = []
        for s_id, one_pool in enumerate(edge_pool):
            for _instance in one_pool:
                e_ins.append(_instance)
                e_ins_sids.append(s_id)
        e_ins, e_ins_lens = pad_tensor_1d(e_ins, self.pad)

        if self.args.cuda:
            q = q.to(self.args.gpu)
            ment_f = ment_f.to(self.args.gpu)
            match_f = match_f.to(self.args.gpu)
            v_ins = v_ins.to(self.args.gpu)
            e_ins = e_ins.to(self.args.gpu)
            tgt_aqg_inputs = [[[y.to(self.args.gpu) for y in g] for g in s] for s in tgt_aqg_inputs]

        return q, q_lens, ment_f, match_f, \
               v_ins, v_ins_lens, v_ins_sids, v_ins_names, \
               e_ins, e_ins_lens, e_ins_sids, e_ins_names, \
               tgt_aqgs, tgt_aqg_inputs, tgt_aqg_objs, tgt_nhgg_objs, \
               tgt_v_ins_objs, tgt_e_ins_objs, \
               tgt_v_copy_objs, tgt_e_copy_objs, tgt_seg_switch_objs, \
               v_st_idx, e_st_idx, v_ins_name2id, e_ins_name2id, \
               data
    
    def load_data(self, datas, bs, training_proportion=1.0, use_small=False, shuffle=True):
        self._load_data(datas, bs, training_proportion, use_small, shuffle)


class RelationRankingDataLoader(DataLoader):

    def __init__(self, args, mode):
        super(RelationRankingDataLoader, self).__init__(args)
        self.dataset = args.dataset

        assert mode in ["train", "dev", "test"]

        if args.dataset == "lcq":
            self.kb = "dbpedia"
        else:
            self.kb = "freebase"

        self.tokenizer = pickle.load(open(self.args.vocab_path, 'rb'))
        self.pad = self.tokenizer.lookup(self.tokenizer.pad_token)

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

        q = text_to_tensor_1d(d["question_toks"], self.tokenizer)
        pos_r = text_to_tensor_1d(d["positive_relation"], self.tokenizer)

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
        neg_r, neg_r_lens = text_to_tensor_2d(cand_rels, self.tokenizer)
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

        self._load_data(new_datas, bs, 1.0, use_small, shuffle)

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
