# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : utils.py
# @Software: PyCharm
"""

import sys
import re
from builtins import range
import numpy as np
import random
import torch
import copy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import signal, functools


REAL = np.float32
if sys.version_info[0] >= 3:
    unicode = str

big_bracket_pattern = re.compile(r'[{](.*?)[}]', re.S)

date_pattern_0 = re.compile(r'\d{1,2}(/|-)\d{1,2}(/|-)\d{2,4}', re.S)               # 03-02-13 or 03-02-2013 or 03/02/13 or 03/02/2013
date_pattern_1 = re.compile(r'\d{4}(/|-)\d{1,2}(/|-)\d{1,2}', re.S)                 # 2013-03-02
date_pattern_2 = re.compile(r'(Jan|January|Feb|February|Mar|March|Apr|April'
                            r'|May|Jun|June|Jul|July|Aug|August|Sept|September'
                            r'|Oct|October|Nov|November|Dec|December)'
                            r'( |, |,)(\d{1,2},( |))?\d{2,4}', re.S)                   # November 6, 1962
                                                                                       # November, 1962
date_pattern_3 = re.compile(r'\d{1,2} '
                            r'(Jan|January|Feb|February|Mar|March|Apr|April'
                            r'|May|Jun|June|Jul|July|Aug|August|Sept|September'
                            r'|Oct|October|Nov|November|Dec|December) '
                            r'\d{2,4}', re.S)                                       # 6 November 1962

year_pattern = re.compile(r'(10|11|12|13|14|15|16|17|18|19|20)\d{2}', re.S)         # 1995

int_pattern = re.compile(r'\d{1,3},\d{3}(,\d{3})*', re.S)                           # 34,245,023
float_pattern = re.compile(r'(-|)([0-9]+(\.[0-9]*)?)', re.S)                        # 0.05 or -1023.56

quotation_pattern = re.compile(r'\"(.*?)\"', re.S)

STOP_WORDS = ["the", ""]
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
stwords = stopwords.words('english')


synonyms_lcq = [["famous", "notabl"], ["di", "death"], ["writ", "auth"], ["liv", "resid"]]
synonyms_wsp = []

synonyms_dict_lcq = set()
for w1, w2 in synonyms_lcq:
    syn = [w1, w2]
    syn.sort()
    syn = " ".join(syn)
    synonyms_dict_lcq.add(syn)
synonyms_dict_wsp = set()
for w1, w2 in synonyms_wsp:
    syn = [w1, w2]
    syn.sort()
    syn = " ".join(syn)
    synonyms_dict_wsp.add(syn)

STOP_RELATIONS = ["<http://dbpedia.org/property/range>",
                  "<http://dbpedia.org/property/died>",
                  "<http://dbpedia.org/property/built>",
                  "<http://dbpedia.org/property/familia>",
                  "<http://dbpedia.org/property/portrayer>",
                  "<http://dbpedia.org/ontology/significantProject>",
                  "<http://dbpedia.org/property/schoolColours>",
                  "<http://dbpedia.org/ontology/officialSchoolColour>",
                  "<http://dbpedia.org/ontology/notableIdea>",
                  "<http://dbpedia.org/property/launches>",
                  "<http://dbpedia.org/property/stadiumName>",
                  "<http://dbpedia.org/property/platform>",
                  "<http://dbpedia.org/property/platforms>",
                  "<http://dbpedia.org/property/buildingType>",
                  "<http://dbpedia.org/property/locationCity>",
                  "<http://dbpedia.org/ontology/internationalAffiliation>"]

def is_variable(name):
    return name[0] == "?"

def is_entity(name, kb="freebase"):
    if kb == "freebase":
        return "ns:m." in name or "ns:g." in name
    else:
        return "http://dbpedia.org/resource/" in name

def is_type(name, kb="freebase"):
    if kb == "freebase":
        return "ns:m." not in name and "ns:g." not in name and name[:3] == "ns:"
    else:
        return "http://dbpedia.org/ontology/" in name

def is_value(name):
    if name[0] == "\"" or name[0] == "<":
        return True
    if re.match(float_pattern, name):
        return True
    return False

def is_relation(name, kb="freebase"):
    if kb == "freebase":
        return "ns:" in name
    else:
        return "http://dbpedia.org/ontology/" in name or "http://dbpedia.org/property/" in name \
               or "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in name

def get_operator_by_t(t):
    if t == 0 or t % 3 == 1:
        return "av"
    elif t % 3 == 0:
        return "ae"
    else:
        return "sv"

def get_inv_op(op):
    OPS = ["<", ">", "<=", ">="]
    INV_OPS = [">", "<", ">=", "<="]
    for i in range(len(OPS)):
        if OPS[i] == op:
            return INV_OPS[i]

def get_inv_edge(e):
    if e % 2 == 0:
        return e + 1
    return e - 1

def get_inv_name(e):
    return e + "@@@INV"

def get_origin_name(e):
    return e.split("@@@")[0]

def normalize_relation(relation):
    if relation == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
        return relation.lower()
    if "http://dbpedia.org/" in relation:
        relation = relation.strip("<").strip(">")
        relation = relation.split("/")[-1]
        relation = relation.rstrip("s")
    return relation.lower()

def combine_mapping(mapping1, mapping2, cover=False):
    for k, v in mapping2.items():
        if not cover:
            if k not in mapping1:
                mapping1[k] = v
        else:
            mapping1[k] = v
    return mapping1

def rename(name, depth):
    return name + "##" + str(depth)

def rename_inv(name):
    return name.split("##")[0]

def remove_type(var):
    if "(" in var:
        assert "xsd:datetime" in var.lower() or "str" in var.lower() or "xsd:integer" in var.lower()
        return get_content_from_outermost_brackets(var, 0, "(")
    return var

def step_to_op_step(t, mode):
    assert mode in ["vertex", "edge"]
    if mode == "vertex":
        return step_to_av_step(t)
    else:
        return step_to_ae_step(t)

def av_step_to_step(t_ins):
    return 0 if t_ins == 0 else 3 * t_ins - 2

def ae_step_to_step(t_ins):
    return (t_ins + 1) * 3

def step_to_av_step(t):
    return 0 if t == 0 else (t + 2) // 3

def step_to_ae_step(t):
    return t // 3 - 1

def get_copy_v_label_idx(idx):
    return 0 if idx == 0 else (idx + 2) // 3

def get_copy_e_label_idx(idx):
    return idx // 3 - 1

def get_right_bracket(l_bracket):
    if l_bracket == "{":
        return "}"
    elif l_bracket == "[":
        return "]"
    elif l_bracket == "(":
        return ")"
    else:
        raise ValueError("Left bracket must in [\"{\", \"[\", \"(\"] !")

def get_content_from_outermost_brackets(string, begin, l_bracket, return_index=False):
    """
    Find the content in the outermost matched bracket in the string from the position "begin".
    @param string:          input string
    @param begin:           the begin position of str for lookup
    @param l_bracket:       left bracket in ["{", "[", "("]
    @param return_index:    return the index of matched bracket
    @return:                If return_index is True, return content (without bracket), l_idx, r_idx;
                            else return content.
    """
    # when two brackets are matched, balance is 0
    balance = 1
    r_bracket = get_right_bracket(l_bracket)
    l_idx = string.find(l_bracket, begin)
    if l_idx == -1:
        if return_index:
            return None, -1, -1
        else:
            return None
    else:
        for r_idx in range(l_idx + 1, len(string)):
            if string[r_idx] == l_bracket:
                balance += 1
            elif string[r_idx] == r_bracket:
                balance -= 1
            if balance == 0:
                if return_index:
                    return string[l_idx + 1: r_idx], l_idx + 1, r_idx
                else:
                    return string[l_idx + 1: r_idx]
        raise ValueError("\"{}\" do not have matched bracket".format(string[begin:]))

def get_content_behind_key_word(string, begin, key_word, only_index=False):
    """
    Find the content behind the key work, from the position "begin".
    @param string:      input string
    @param begin:       beginning position
    @param key_word:    key word
    @param only_index:  whether only return beginning index of the content.
    @return:            If there is some character except " " and "\t" between the begin position and the key word, return -1
    """
    pos = string.find(key_word, begin)
    if pos == -1:
        if only_index:
            return -1
        else:
            return string, -1
    else:
        for idx in range(begin, pos):
            # there is other character
            if string[idx] != " " and string[idx] != "\t":
                if only_index:
                    return -1
                else:
                    return string, -1
        if only_index:
            return pos + len(key_word)
        else:
            return string[pos + len(key_word):], pos + len(key_word)

def identity(x):
    return x

def pad(tensor, length, pad_idx):
    """
    :param tensor: Size(src_sent_len, ...)
    :param length: target_sent_length
    :param pad_idx: index of padding token
    :return: Size(target_sent_length, ...)
    """
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(pad_idx)])

def text_to_tensor_1d(ex, dict):
    """
    :param ex: text, 1-d list of tokens
    :return: Size(sent_len)
    """
    return torch.LongTensor(dict.convert_tokens_to_ids(ex))

def text_to_tensor_2d(ex, dict):
    """
    :param ex: [text_1, text_2, ... text_n]
    :return: Size(n, max_sent_len)
    """
    return pad_tensor_1d([dict.convert_tokens_to_ids(y) for y in ex], dict.lookup(dict.pad_token))

def pad_tensor_1d(l, pad_idx):
    """
    :param l: [Size(len1), Size(len2), ..., Size(lenn)]
    :return: Size(n, max_sent_len), Size(n)
    """
    lens = [len(x) for x in l]
    m = max(lens)
    return torch.stack([pad(torch.LongTensor(x) if not torch.is_tensor(x) else x, m, pad_idx) for x in l], 0), torch.LongTensor(lens)

def pad_tensor_2d(l, pad_idx):
    """
    :param l: [Size(n1, len1), Size(n2, len2) ...]
    :return: Size(n1 + n2 + .. nn, max(len1, len2... lenn))
    """
    lens = [x.size(0) for x in l]
    m = max([x.size(1) for x in l])
    data = [pad(x.transpose(0, 1), m, pad_idx).transpose(0, 1) for x in l]
    data = torch.cat(data, 0)
    return data, torch.LongTensor(lens)

def pad_graph(tgt_aqgs):
    bs = len(tgt_aqgs)
    tgt_lens = [len(x) for x in tgt_aqgs]
    max_len = max(tgt_lens)

    new_tgt_aqgs = []
    for t in range(max_len):

        v_list = []
        e_list = []
        v_class_list = []
        e_class_list = []
        v_segment_list = []
        e_segment_list = []
        adj_list = []

        for sid in range(bs):

            v_num = (t - 2) // 3 + 2
            e_num = (t - 1) // 3 * 2

            if t < tgt_lens[sid]:
                v, v_class, v_segment, \
                e, e_class, e_segment, adj = tgt_aqgs[sid][t]
            else:
                # padding vectors
                v = torch.LongTensor(v_num).zero_()
                v_class = torch.LongTensor(v_num).zero_()
                v_segment = torch.LongTensor(v_num).zero_()

                e = torch.LongTensor(e_num).zero_()
                e_class = torch.LongTensor(e_num).zero_()
                e_segment = torch.LongTensor(e_num).zero_()

                adj = torch.ones(v_num + e_num + 1, v_num + e_num + 1)

            v_list.append(v)
            e_list.append(e)
            v_class_list.append(v_class)
            e_class_list.append(e_class)
            v_segment_list.append(v_segment)
            e_segment_list.append(e_segment)
            adj_list.append(adj)

        v = torch.stack(v_list)
        e = torch.stack(e_list)
        v_classes = torch.stack(v_class_list)
        e_classes = torch.stack(e_class_list)
        v_segments = torch.stack(v_segment_list)
        e_segments = torch.stack(e_segment_list)
        adjs = torch.stack(adj_list)

        new_tgt_aqgs.append([v, v_classes, v_segments,
                             e, e_classes, e_segments,
                             adjs])
    return new_tgt_aqgs, torch.LongTensor(tgt_lens)

def length_array_to_mask_tensor(length_array, value=None, reverse=True, mask_symbol=1, use_bool=False):
    max_len = max(length_array)
    batch_size = len(length_array)

    if reverse:
        mask = np.ones((batch_size, max_len), dtype=np.uint8)
        for i, seq_len in enumerate(length_array):
            mask[i][:seq_len] = 0
    else:
        mask = np.zeros((batch_size, max_len), dtype=np.uint8)
        for i, seq_len in enumerate(length_array):
            mask[i][:seq_len] = 1

    if value is not None:
        for b_id in range(len(value)):
            for c_id, c in enumerate(value[b_id]):
                if c == mask_symbol:
                    mask[b_id][c_id] = 1 if reverse else 0

    if not use_bool:
        mask = torch.ByteTensor(mask)
    else:
        mask = torch.BoolTensor(mask)
    return mask

def mk_graph_for_gnn(vertices, v_classes, v_segments, edges, e_classes, e_segments, triples):
    """
    make a graph for the input format of graph transformer.
    @param vertices:    LIST, index of each vertex
    @param v_classes:   LIST, class of each vertex
    @param v_segments:  LIST, segment of each vertex (the subquery ID)
    @param edges:       LIST, index of each edge
    @param e_classes:   LIST, class of each edge
    @param e_segments:  LIST, segment of each edge
    @param triples:     [s, o, p] LIST
    @return: tensor of vertex ID, class, segment
             tensor of edge ID, class, segment
             adjacency matrix
    """
    v_index = {v: i for i, v in enumerate(vertices)}
    e_index = {e: i for i, e in enumerate(edges)}

    v_class_tensor = torch.LongTensor([v_classes[x] for x in vertices])
    e_class_tensor = torch.LongTensor([e_classes[x] for x in edges])

    v_tensor = torch.LongTensor([x for x in vertices])
    e_tensor = torch.LongTensor([x for x in edges])

    v_segment_tensor = torch.LongTensor([v_segments[x] for x in vertices])
    e_segment_tensor = torch.LongTensor([e_segments[x] for x in edges])

    v_num = len(vertices)
    e_num = len(edges)
    adj_sz = v_num + 1 + e_num

    adj = torch.zeros(adj_sz, adj_sz)
    for i in range(adj_sz):
        adj[i, v_num] = 1
        adj[v_num, i] = 1
    for i in range(adj_sz):
        adj[i, i] = 1
    for i, t in enumerate(triples):
        a = v_index[t[0]]
        b = v_index[t[1]]
        c = e_index[t[2]] + v_num + 1
        adj[a, c] = 1
        adj[c, b] = 1
    return v_tensor, v_class_tensor, v_segment_tensor,  \
           e_tensor, e_class_tensor, e_segment_tensor, adj

def tokenize_word_sentence(text):
    text = text.lower()
    text = text.strip(" ").strip("?").strip(".").strip(" ")
    text = text.replace(",", " ,")
    text = text.replace("'", " '")
    text = text.replace("'s", " 's")
    text = text.replace("'re", " 're")
    tokens = [wordnet_lemmatizer.lemmatize(x.strip(" ").strip(".")) for x in text.split(" ") if x not in STOP_WORDS]
    return tokens

def lemmatize_word(word):

    if word == "almamater":
        return ["alma", "mat"]

    if len(word) > 4 and word[-4:] == "ship":
        return [word[:-4]]

    if len(word) > 2 and word[-2:] == "or":
        return [word[:-2]]

    if word[-4:] == "ence":
        return [word[:-4]]
    if word[-4:] == "tten":
        return [word[:-3]]
    if word[-3:] == "ing":
        return [word[:-3]]
    if word[-2:] == "es":
        return [word[:-2]]
    if word[-2:] == "ed":
        return [word[:-2]]
    if word[-2:] == "er":
        return [word[:-2]]
    if word[-1] == "e":
        return [word[:-1]]
    return [word]

def tokenize_word_sentence_bert(text, bert_tokenizer, start_cls=True):
    text = text.lower()
    text = text.strip(" ").strip("?").strip(".").strip(" ")
    if start_cls:
        tokens = ['[cls]']
    else:
        tokens = []
    for token in text.split(" "):
        if token == "[unused0]":
            tokens += [token]
        else:
            tokens += bert_tokenizer.tokenize(token)
    return tokens

def instance_pool_to_tensor(instance_pool, pad_idx):
    instance_tensor = []
    instance_classes = []
    instance_s_ids = []
    for s_id, one_pool in enumerate(instance_pool):
        for _class, instance_list in one_pool.items():
            for _instance in instance_list:
                instance_tensor.append(_instance)
                instance_classes.append(_class)
                instance_s_ids.append(s_id)
    instance_tensor, instance_lens = pad_tensor_1d(instance_tensor, pad_idx)
    return instance_tensor, instance_lens, instance_classes, instance_s_ids

def instance_pool_to_tensor_for_plm(instance_pool, pad_idx, sep_idx):
    instance_tensor = []
    instance_classes = []
    instance_s_ids = []
    for s_id, one_pool in enumerate(instance_pool):
        for _class, instance_list in one_pool.items():
            print(_class)
            print(len(instance_list), [x.size() for x in instance_list])
            for _instance in instance_list:
                instance_tensor.append(_instance)
                instance_classes.append(_class)
                instance_s_ids.append(s_id)
    instance_tensor, instance_lens = pad_tensor_1d(instance_tensor, pad_idx)
    return instance_tensor, instance_lens, instance_classes, instance_s_ids

def instance_tensor_to_pool(instance_vec, instance_classes, instance_s_ids):
    assert instance_vec.size(0) == len(instance_s_ids)
    assert len(instance_s_ids) == len(instance_classes)
    st = 0
    instance_pool = []
    while st < len(instance_s_ids):
        ed = st
        while ed < len(instance_s_ids) and instance_s_ids[ed] == instance_s_ids[st]:
            ed += 1
        one_pool = {}
        st_1 = st
        while st_1 < ed:
            ed_1 = st_1
            _class = instance_classes[st_1]
            while ed_1 < ed and instance_classes[ed_1] == _class:
                ed_1 += 1
            assert _class not in one_pool
            one_pool[_class] = instance_vec[st_1:ed_1]
            st_1 = ed_1
        instance_pool.append(one_pool)
        st = ed
    return instance_pool

def instance_tensor_to_pool_without_class(instance_vec, instance_s_ids):
    assert instance_vec.size(0) == len(instance_s_ids)
    st = 0
    instance_pool = []
    while st < len(instance_s_ids):
        ed = st
        while ed < len(instance_s_ids) and instance_s_ids[ed] == instance_s_ids[st]:
            ed += 1
        instance_pool.append(instance_vec[st: ed])
        st = ed
    return instance_pool

def exact_sequence_matching(toks1, toks2):
    for st in range(len(toks2) - len(toks1) + 1):
        flag = True
        for i in range(len(toks1)):
            if toks1[i] != toks2[st + i]:
                flag = False
        if flag:
            return True
    return False

def revert_period_constraint_in_conds(periods, seg_conds):
    tmp_seg_conds = []
    for s, p, o in seg_conds:
        if "$$$" in p:
            periods.append(o)
            o_st = o + "_st"
            o_ed = o + "_ed"
            p_from = p.split("$$$")[0]
            p_to = ".".join(p.split("$$$")[0].split(".")[:-1]) + "." + p.split("$$$")[1]
            tmp_seg_conds.append([s, p_from, o_st])
            tmp_seg_conds.append([s, p_to, o_ed])
        else:
            tmp_seg_conds.append([s, p, o])
    return tmp_seg_conds

def revert_period_constraint(periods, seg_sels, seg_conds, seg_filters, seg_filters_main):
    seg_filters_list = [seg_filters, seg_filters_main]
    tmp_seg_filters_list = []
    tmp_seg_conds = revert_period_constraint_in_conds(periods, seg_conds)

    for segs in seg_filters_list:
        tmp_seg_filters = []
        for s, p, o in segs:
            if s in periods:
                assert p == "during" or p == "overlap"
                if o[0] == "?":
                    o_st = o + "_st"
                    o_ed = o + "_ed"
                elif "$$$" in o:
                    o_st = o.split("$$$")[0]
                    o_ed = o.split("$$$")[1]
                else:
                    o_st = o
                    o_ed = o
                s_st = s + "_st"
                s_ed = s + "_ed"
                if o_st[0] != "?":
                    s_st = "xsd:datetime(" + s_st + ")"
                if o_ed[0] != "?":
                    s_ed = "xsd:datetime(" + s_ed + ")"

                if p == "during":
                    tmp_seg_filters.append([s_st, ">=", o_st])
                    tmp_seg_filters.append([s_ed, "<=", o_ed])
                else:
                    tmp_seg_filters.append([s_st, "<=", o_st])
                    tmp_seg_filters.append([s_ed, ">=", o_ed])
            else:
                tmp_seg_filters.append([s, p, o])
        tmp_seg_filters_list.append(tmp_seg_filters)

    tmp_seg_sels = []
    for x in seg_sels:
        if x in periods:
            tmp_seg_sels.append(x + "_st")
            tmp_seg_sels.append(x + "_ed")
        else:
            tmp_seg_sels.append(x)
            
    seg_sels = [x for x in tmp_seg_sels]
    seg_conds = [x for x in tmp_seg_conds]
    seg_filters = [x for x in tmp_seg_filters_list[0]]
    seg_filters_main = [x for x in tmp_seg_filters_list[1]]
    return seg_sels, seg_conds, seg_filters, seg_filters_main

def formalize_time_constraint(conds, filters):
    new_time_filters = []
    for rel in ["from", "to", "start_date", "end_date"]:
        t1, t2 = -1, -1
        for i, triple in enumerate(conds):
            s, p, o = triple
            if p.split(".")[-1] == rel:
                oo = "xsd:datetime(" + o + ")"
                for j, triple_1 in enumerate(filters):
                    v1, cmp, v2 = triple_1
                    if v1 == oo and v2[0] != "?":
                        t1 = i
                        t2 = j

        if t1 != -1 and t2 != -1:
            s, p, o = copy.deepcopy(conds[t1])
            v1, cmp, v2 = copy.deepcopy(filters[t2])
            conds.pop(t1)
            filters.pop(t2)
            o_1 = o + "_1"
            oo_1 = "xsd:datetime(" + o_1 + ")"
            new_filter = "Filter( NOT EXISTS {" + s + " " + p + " " + o + " }\n" + \
                         " || EXISTS {" + s + " " + p + " " + o_1 + " . " + \
                         "FILTER(" + oo_1 + " " + cmp + " " + v2 + ")})\n"
            new_time_filters.append(new_filter)
    return new_time_filters

def simplify_variable_in_filter(conds):
    mapping = {}
    for v1, e, v2 in conds:
        if v2[0] == "\"" and e == "=":
            mapping[v1] = v2
    new_conds = []
    for v1, e, v2 in conds:
        if v2[0] == "\"" and e == "=":
            continue
        for k, v in mapping.items():
            assert v1 != k
            if v2 == k:
                v2 = v
                break
        new_conds.append([v1, e, v2])
    return new_conds

def expand_variable_in_filter(conds):
    new_filters = []
    for i, triple in enumerate(conds):
        if triple[-1][0] == "\"":
            if "xsd:dateTime" in triple[-1]:
                new_o = "?date_v_" + str(len(new_filters))
                new_type = "xsd:datetime"
            else:
                new_o = "?str_v_" + str(len(new_filters))
                new_type = "str"
            new_filter = "Filter(" + new_type + "(" + new_o + ") = " + triple[-1] + ")"
            new_filters.append(new_filter)
            conds[i][-1] = new_o
    return new_filters

def split_pooling(tensor, split_lens):
    data = tensor.split(split_lens.tolist())
    data = [x.max(dim=0)[0] for x in data]
    return torch.stack(data, 0)

def split_padding(tensor, split_lens):
    data = tensor.split(split_lens.tolist())
    m = max([y.size(0) for y in data])
    return torch.stack([pad(y, m, 0) for y in data], 0)

def mask_seq(seq, seq_lens):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    mask = torch.zeros_like(seq)
    for i, l in enumerate(seq_lens):
        mask[i, :l].fill_(1)
    return mask

def max_pooling_by_lens(seq, seq_lens):
    mask = mask_seq(seq, seq_lens)
    seq = seq.masked_fill(mask == 0, -1e18)
    return seq.max(dim=1)[0]

def max_pooling_by_mask(seq, mask):
    mask = mask.unsqueeze(-1).expand_as(seq)
    seq = seq.masked_fill(mask.bool(), -1e18)
    return seq.max(dim=1)[0]

def tokenize_by_uppercase(s):
    tokens = []
    last = 0
    for i, c in enumerate(s):
        if c.isupper():
            tokens.append(s[last: i])
            last = i
    tokens.append(s[last: len(s)])
    tokens = [x for x in tokens if x != ""]
    return tokens

def check_query_equal(query1, query2):
    r1 = set([" ".join([y for y in x if y not in ["property", "ontology"]]) for x in query1])
    r2 = set([" ".join([y for y in x if y not in ["property", "ontology"]]) for x in query2])

    insect_r = r1 & r2

    if len(insect_r) == len(r1) and len(insect_r) == len(r2):
        return True
    return False

def check_in(query, query_list):
    for q in query_list:
        if check_query_equal(query, q):
            return True
    return False

def check_relation(rel):
    if rel.find("http://dbpedia.org/property/") != -1 or \
        rel.find("http://dbpedia.org/ontology/") != -1:
        return True
    return False

def get_rels_from_query(query):
    where_clauses = re.findall(big_bracket_pattern, query)[0]
    where_clauses = where_clauses.strip(" ").strip(".").strip(" ")
    triples = [[y.strip(" ") for y in x.strip(" ").split(" ") if y != ""]
               for x in where_clauses.split(". ")]
    relaions = [x[1] for x in triples if check_relation(x[1])]
    return relaions

def cal_scores(pred_answers, gold_answers):
    tp = 0.
    for a in pred_answers:
        if a in gold_answers:
            tp += 1
    if tp == 0:
        return (0., 0., 0., 0.)
    p = 1.0 * tp / len(pred_answers)
    r = 1.0 * tp / len(gold_answers)
    f1 = 2.0 * p / (p + r) * r

    random_answer = random.choice(pred_answers)
    if random_answer in gold_answers:
        hit_1 = 1.
    else:
        hit_1 = 0.
    return (p, r, f1, hit_1)

def cal_hit_at_1(pred_answers, gold_answers):
    if len(pred_answers) == 0:
        return 0.0
    random_answer = random.choice(pred_answers)
    if random_answer in gold_answers:
        return 1.0
    return 0.0

stop_tok_lcq = ["place", "in", "of", "name", "by", "on"]
stop_tok_wsp = ["in", "of", "by", "on", "to", "from"]


def cal_literal_matching_score(dataset, q_seq, r_seq, n=5):

    def matching(w1, w2):
        if dataset == "lcq" and w1 == w2:
            return True

        ww = [w1, w2]
        ww.sort()
        ww = " ".join(ww)
        if dataset == "lcq":
            return ww in synonyms_dict_lcq
        else:
            return ww in synonyms_dict_wsp

    def partial_matching(w1, w2, n):
        if dataset == "lcq":
            if "".join(w1[:n]) == "".join(w2[:n]):
                return True
            if "".join(w1[:n]) == "".join(w2[-n:]):
                return True
            if "".join(w1[-n:]) == "".join(w2[-n:]):
                return True
            if "".join(w1[-n:]) == "".join(w2[:n]):
                return True
        return matching(w1, w2)

    if q_seq[0] == "name":
        q_seq = q_seq[1:]
    if r_seq[0] in ["ontology", "property"]:
        r_seq = r_seq[1:]

    tmp_q_seq = []
    for tok in q_seq:
        if dataset == "lcq" and tok == "[sep]":
            break
        for _tok in nltk.word_tokenize(tok):
            tmp_q_seq.extend(lemmatize_word(_tok))

    tmp_r_seq = []
    for tok in r_seq:
        for _tok in nltk.word_tokenize(tok):
            tmp_r_seq.extend(lemmatize_word(_tok))

    q_seq = [x for x in tmp_q_seq]
    r_seq = [x for x in tmp_r_seq]

    # print(q_seq)
    # print(r_seq)

    # exact matching
    for i in range(len(q_seq) - len(r_seq) + 1):
        flag = True
        for j in range(len(r_seq)):
            if not matching(q_seq[i + j], r_seq[j]):
                flag = False
                break
        if flag:
            return "ExactMatching", len(r_seq)

    if dataset == "lcq":
        gama = 0.5
        stop_tok = [x for x in stop_tok_lcq]
    else:
        gama = 1.0
        stop_tok = [x for x in stop_tok_wsp]
    # partial matching
    _cnt = 0
    for r_tok in r_seq:
        if r_tok in stop_tok:
            continue
        for q_tok in q_seq:
            if partial_matching(r_tok, q_tok, n=n):
                _cnt += 1
    if _cnt > 0:
        if _cnt == len(r_seq):
            return "PartialMatching", _cnt
        else:
            return "PartialMatching", _cnt * gama
    return "NoMatching", 0

def literal_matching_score_tmp(seq1, seq2):
    if seq2[0] in ["ontology", "property"]:
        seq2 = seq2[1:]
    for i in range(len(seq1) - len(seq2) + 1):
        flag = True
        for j in range(len(seq2)):
            if seq1[i + j] != seq2[j]:
                flag = False
                break
        if flag:
            return 1
    return 0

def intersection(list1, list2):
    res = list(set(list1) & set(list2))
    res.sort()
    return res

def aeq(*args):
    base = args[0]
    for a in args[1:]:
        assert a == base, str(args)

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode.
    :param text:
    :param encoding:
    :param errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)

def load_glove_vocab(filename, vocab_size=None, binary=False, encoding='utf8', unicode_errors='ignore'):
    vocab = set()

    with open(filename, 'rb') as fin:
        # header = to_unicode(fin.readline(), encoding=encoding)
        # vocab_size, vector_size = map(int, header.split())  # throws for invalid file format

        if binary:
            for _ in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                vocab.add(word)
        else:
            for line_no, line in enumerate(fin):
                parts = to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                word = parts[0]
                vocab.add(word)
    return vocab

def load_word2vec_format(filename, word_idx, binary=False, normalize=False,
                         encoding='utf8', unicode_errors='ignore'):
    """
    load Word Embeddings
    If you trained the C model using non-utf8 encoding for words, specify that
    encoding in `encoding`.
    :param filename :
    :param word_idx :
    :param binary   : a boolean indicating whether the data is in binary word2vec format.
    :param normalize:
    :param encoding :
    :param unicode_errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    vocab = set()

    with open(filename, 'rb') as fin:
        # header = to_unicode(fin.readline(), encoding=encoding)
        # vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
        vocab_size = 400000
        vector_size = 300
        word_matrix = torch.zeros(len(word_idx), vector_size)

        def add_word(_word, _weights):
            if _word not in word_idx:
                return
            vocab.add(_word)
            word_matrix[word_idx[_word]] = _weights

        if binary:
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for _ in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                weights = torch.from_numpy(np.fromstring(fin.read(binary_len), dtype=REAL))
                add_word(word, weights)
        else:
            for line_no, line in enumerate(fin):
                parts = to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                word, weights = parts[0], list(map(float, parts[1:]))
                weights = torch.Tensor(weights)
                add_word(word, weights)
    if word_idx is not None:
        assert (len(word_idx), vector_size) == word_matrix.size()
    if normalize:
        # each row normalize to 1
        word_matrix = torch.renorm(word_matrix, 2, 0, 1)
    return word_matrix, vector_size, vocab

class TimeoutError(Exception): pass

def timeout(seconds, error_message="Timeout Error: the cmd 30s have not finished."):
    def decorated(func):
        result = ""

        def _handle_timeout(signum, frame):
            global result
            result = "TimeOut"
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            global result
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                return result
            return result

        return functools.wraps(func)(wrapper)
    return decorated

def update_model(step, accumulation_steps, model, optimizer, clip_grad=1.0):
    # Caculate gradients and update parameters
    if (step + 1) % accumulation_steps == 0:
        # clip the gradient.
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad()

def eval_train_accuracy(tgt_objs_records, action_probs_records):
    assert len(action_probs_records) == len(tgt_objs_records)

    n_q_total, n_q_correct, n_aqg_correct = 0, 0, 0
    n_aqg_step_correct, n_aqg_step_total = 0, 0
    n_v_step_correct, n_v_step_total = 0, 0
    n_e_step_correct, n_e_step_total = 0, 0

    for x, y in zip(action_probs_records, tgt_objs_records):
        action_probs, v_action_probs, e_action_probs = x
        tgt_objs, tgt_v_ins_objs, tgt_e_ins_objs = y

        for s_id in range(len(tgt_objs)):
            # AQG generation
            is_aqg_correct = True
            for j in range(len(tgt_objs[s_id])):
                pred_obj = torch.argmax(action_probs[s_id][j], dim=-1).item()
                if pred_obj == tgt_objs[s_id][j]:
                    n_aqg_step_correct += 1
                else:
                    is_aqg_correct = False
            n_aqg_step_total += len(tgt_objs[s_id])
            n_aqg_correct += is_aqg_correct

            # vertex instance
            is_v_correct = True
            v_ins_objs = [x for x in tgt_v_ins_objs[s_id] if x != -1]
            for j in range(len(v_ins_objs)):
                pred_obj = torch.argmax(v_action_probs[s_id][j], dim=-1).item()
                if pred_obj == v_ins_objs[j]:
                    n_v_step_correct += 1
                else:
                    is_v_correct = False
            n_v_step_total += len(v_ins_objs)

            # edge instance
            is_e_correct = True
            e_ins_objs = [x for x in tgt_e_ins_objs[s_id] if x != -1]
            for j in range(len(tgt_e_ins_objs[s_id])):
                pred_obj = torch.argmax(e_action_probs[s_id][j], dim=-1).item()
                if pred_obj == e_ins_objs[j]:
                    n_e_step_correct += 1
                else:
                    is_e_correct = False
            n_e_step_total += len(e_ins_objs)

            n_q_correct += is_aqg_correct and is_v_correct and is_e_correct

        n_q_total += len(tgt_objs)

    train_aqg_acc = 100. * n_aqg_correct / n_q_total
    train_aqg_step_acc = 100. * n_aqg_step_correct / n_aqg_step_total
    train_v_step_acc = 100. * n_v_step_correct / n_v_step_total
    train_e_step_acc = 100. * n_e_step_correct / n_e_step_total
    train_q_acc = 100. * n_q_correct / n_q_total
    return train_aqg_acc, train_aqg_step_acc, train_v_step_acc, train_e_step_acc, train_q_acc