# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : preprocess.py
# @Software: PyCharm
"""

import re
import os
import sys
import json
import copy
import pickle
import random
import itertools
import argparse
import numpy as np
from transformers import BertTokenizer

sys.path.append("..")
from rules.sparql import Annotator
from rules.grammar import build_instance_pool, V_CLASS_IDS, E_CLASS_IDS
from utils.utils import tokenize_word_sentence, tokenize_word_sentence_bert, big_bracket_pattern, step_to_ae_step


def annotate(dataset, data_path, ent_pool_path, rel_pool_path, type_pool_path, val_pool_path, output, rel_topk, kb_endpoint, training=False):
    if dataset == "lcq":
        kb = "dbpedia"
    else:
        kb = "freebase"

    datas = json.load(open(data_path, "r"))

    rel_pool = load_relation_pool(rel_pool_path, rel_topk)
    
    if ent_pool_path:
        ent_pool = load_entity_pool_for_wsp(ent_pool_path)
    else:
        ent_pool = None

    if type_pool_path:
        type_pool = load_type_pool(type_pool_path)
    else:
        type_pool = None

    if val_pool_path:
        val_pool = load_value_pool(val_pool_path)
    else:
        val_pool = None

    annotated_datas = []
    annotator = Annotator(dataset=dataset)

    n_total = 0
    for idx, d in enumerate(datas):
        n_total += 1

        if d["id"] not in rel_pool:
            continue

        instance_pool = build_instance_pool(d, ent_pool, rel_pool, type_pool, val_pool, kb, kb_endpoint)

        aqg_list, aqg_obj_labels_list, \
        v_instance_obj_labels_list, \
        e_instance_obj_labels_list, \
        v_copy_labels_list, \
        e_copy_labels_list, \
        segment_switch_labels_list, \
        instance_pool_with_gold_list = annotator.annotate_query(d["query"],
                                                                instance_pool=instance_pool,
                                                                kb=kb,
                                                                kb_endpoint=kb_endpoint,
                                                                training=training)

        if not aqg_obj_labels_list:
            continue

        if training:
            label_num = len(aqg_obj_labels_list)
        else:
            # when testing, only use one group of gold label of UNION
            label_num = 1
            if V_CLASS_IDS["ent"] not in instance_pool["vertex"]:
                continue

        for i in range(label_num):

            new_d = copy.deepcopy(d)
            new_d["aqg_obj_labels"] = [x for x in aqg_obj_labels_list[i]]
            new_d["v_instance_obj_labels"] = [x for x in v_instance_obj_labels_list[i]]
            new_d["e_instance_obj_labels"] = [x for x in e_instance_obj_labels_list[i]]
            new_d["v_copy_labels"] = [x for x in v_copy_labels_list[i]]
            new_d["e_copy_labels"] = [x for x in e_copy_labels_list[i]]
            new_d["segment_switch_labels"] = [x for x in segment_switch_labels_list[i]]
            new_d["gold_aqg"] = copy.deepcopy(aqg_list[i])

            if training:
                # when training, use the instance set that includes the gold
                new_d["instance_pool"] = copy.deepcopy(instance_pool_with_gold_list[i])
            else:
                # when testing, only use the instance set
                new_d["instance_pool"] = copy.deepcopy(instance_pool)

            new_d["matching_feature"] = enhance_edge_matching(new_d["question"], new_d["instance_pool"]["edge"])

            # print(new_d["query"])
            # print()
            # print("aqg_obj:", new_d["aqg_obj_labels"])
            # print("v_instance_obj:", new_d["v_instance_obj_labels"])
            # print("e_instance_obj:", new_d["e_instance_obj_labels"])
            # print("v_copy_labels:", new_d["v_copy_labels"])
            # print("e_copy_labels:", new_d["e_copy_labels"])
            # print("segment_switch_labels:", new_d["segment_switch_labels"])
            # print("rel_instance_pool:", new_d["instance_pool"]["edge"][6][:5])
            # print()
            # new_d["gold_aqg"].show_state()
            # exit()

            annotated_datas.append(new_d)

    pickle.dump(annotated_datas, open(output, "wb"))
    print(len(annotated_datas), n_total)

def load_relation_pool(dir, rel_topk):
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
    datas = dfs_load(dir)
    relation_pool = {}
    for d in datas:
        relation_pool[d["id"]] = [x[0] for x in d["candidate_relations"][:rel_topk]]
    return relation_pool

def load_relation_pool_from_v1(dir, rel_topk):
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

    def parse_one_query(query):
        where_clause = re.search(big_bracket_pattern, query).group(0)
        where_clause = where_clause.strip("{").strip("}").strip(" ")
        tmp = [x.strip(".") for x in where_clause.split(" ") if x not in [".", "", " "]]
        one_query = []
        for i, x in enumerate(tmp):
            if i % 3 == 1:
                one_query.append(x)
        return one_query

    datas = dfs_load(dir)
    query_pool = {}
    relation_pool = {}
    for d in datas:
        if not d["cand_queries"]:
            continue

        queries = []
        gold_one_query = parse_one_query(d["query"])
        tmp_gold_one_query = copy.deepcopy(gold_one_query)
        tmp_gold_one_query.sort()
        for query in d["cand_queries"]:
            one_query = parse_one_query(query)
            tmp_one_query = copy.deepcopy(one_query)
            if " ".join(tmp_one_query) != " ".join(tmp_gold_one_query):
                queries.append(one_query)

        if rel_topk != -1:
            query_pool[d["id"]] = random.sample([x for x in queries], min(rel_topk, len(queries)))
        else:
            query_pool[d["id"]] = [x for x in queries]
        query_pool[d["id"]] = [gold_one_query] + query_pool[d["id"]]
        relation_pool[d["id"]] = []
        for one_query in query_pool[d["id"]]:
            relation_pool[d["id"]].extend([rel for rel in one_query])
        relation_pool[d["id"]] = list(set(relation_pool[d["id"]]))
    return relation_pool, query_pool

def build_edge_instance_mask_for_lcq(query_pool, aqg_obj_labels, e_instance_obj_labels, e_intances):

    gold_query = []
    for t, obj in enumerate(aqg_obj_labels):
        if t == 0 or t % 3 != 0:
            continue
        if obj not in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
            continue
        t_ae = step_to_ae_step(t)
        e_id = e_instance_obj_labels[t_ae]
        gold_query.append(e_intances[E_CLASS_IDS["rel+"]][e_id][0])
    gold_query_str = " ".join(gold_query)

    gold_perm = None
    for query in query_pool:
        for perm in itertools.permutations([i for i in range(len(query))], len(query)):
            tmp_query = [query[perm[i]] for i, x in enumerate(query)]
            tmp_query_str = " ".join(tmp_query)
            if tmp_query_str == gold_query_str:
                gold_perm = perm
                break
        if gold_perm is not None:
            break
    cand_rels_in_each_step = []
    for idx in gold_perm:
        cand_rels_in_each_step.append([query[idx] for query in query_pool])

    edge_instance_mask = []
    idx = 0
    for t, obj in enumerate(aqg_obj_labels):
        if t == 0 or t % 3 != 0:
            continue
        if obj in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
            one_mask = [1 for _ in e_intances[E_CLASS_IDS["rel+"]]]
            for j, (_rel, _) in enumerate(e_intances[E_CLASS_IDS["rel+"]]):
                if _rel in cand_rels_in_each_step[idx]:
                    one_mask[j] = 0
            idx += 1
        else:
            one_mask = [1 for _ in e_intances[E_CLASS_IDS["agg+"]]]
            for j, (_agg, _) in enumerate(e_intances[E_CLASS_IDS["agg+"]]):
                if _agg in ["ASK", "COUNT"]:
                    one_mask[j] = 0
        edge_instance_mask.append(one_mask)
    return edge_instance_mask

def load_type_pool(data_path):
    type_pool = {}
    datas = json.load(open(data_path, "r"))
    for d in datas:
        if len(d["candidate_types"]) > 0:
            type_pool[d["id"]] = [x for x in d["candidate_types"]]
    return type_pool

def load_value_pool(data_path):
    val_pool = json.load(open(data_path, "r"))
    return val_pool

def load_entity_pool_for_wsp(path):
    ent_pool = {}
    with open(path, "r", encoding="utf-8") as fin:
        for row in fin:
            row = row.split("\t")
            idx = row[0]
            mid = "ns:" + row[4].strip("/").replace("/", ".")
            name = row[5].replace("_", " ")
            if idx not in ent_pool:
                ent_pool[idx] = []
            ent_pool[idx].append([mid, name])
    return ent_pool

def enhance_edge_matching(q, e_instances):
    rel_match_f = enhance_relation_matching(q, e_instances[E_CLASS_IDS["rel+"]])
    agg_match_f = enhance_aggregation_matching(q, e_instances[E_CLASS_IDS["agg+"]])
    cmp_match_f = enhance_comparison_matching(q, e_instances[E_CLASS_IDS["cmp+"]])
    ord_match_f = enhance_order_matching(q, e_instances[E_CLASS_IDS["ord+"]])
    match_f = rel_match_f + agg_match_f + cmp_match_f + ord_match_f
    return match_f

def enhance_relation_matching(q, rel_instances):
    q_toks = tokenize_word_sentence(q)
    q_toks_bert = tokenize_word_sentence_bert(q, bert_tokenizer, start_cls=False)
    rel_match_f = []
    for rel_name, rel_true_name in rel_instances:
        one_f = 0
        rel_toks = tokenize_word_sentence(rel_true_name)
        rel_toks_bert = tokenize_word_sentence_bert(rel_true_name, bert_tokenizer, start_cls=False)
        for tok in rel_toks:
            if tok in q_toks:
                one_f = 1
        for tok in rel_toks_bert:
            if tok in q_toks_bert:
                one_f = 1
        rel_match_f.append(one_f)
    return rel_match_f

def enhance_aggregation_matching(q, agg_instances):
    agg_match_f = [0 for _ in range(len(agg_instances))]
    return agg_match_f

def enhance_comparison_matching(q, cmp_instances):
    cmp_match_f = [0 for _ in range(len(cmp_instances))]
    return cmp_match_f

def enhance_order_matching(q, ord_instances):
    ord_match_f = [0 for _ in range(len(ord_instances))]
    return ord_match_f


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='lcq', choices=['lcq', 'cwq', 'wsp'])
    arg_parser.add_argument('--bert_mode', type=str, default='bert-base-uncased', choices=['bert-base-uncased', 'bert-large-uncased'])
    args = arg_parser.parse_args()

    random.seed(2021)

    bert_tokenizer = BertTokenizer.from_pretrained("/home/cyr/resources/BERT_model/bert-base-uncased/")

    if args.dataset == "cwq":
        print("Now annotating ComplexWebQuestions ... ")

        kb_endpoint = "http://10.201.69.194:8890//sparql"

        annotate(dataset="cwq",
                 data_path="../data/ComplexWebQuestions/parsed_train.json",
                 ent_pool_path=None,
                 type_pool_path="../data/ComplexWebQuestions/candidate_types_train.json",
                 rel_pool_path="../data/ComplexWebQuestions/candidate_relations_train",
                 val_pool_path="../data/ComplexWebQuestions/value_pool.json",
                 output="../data/ComplexWebQuestions/annotated_train.pkl",
                 rel_topk=50,
                 kb_endpoint=kb_endpoint,
                 training=True)

        annotate(dataset="cwq",
                 data_path="../data/ComplexWebQuestions/parsed_dev.json",
                 ent_pool_path=None,
                 type_pool_path="../data/ComplexWebQuestions/candidate_types_dev.json",
                 rel_pool_path="../data/ComplexWebQuestions/candidate_relations_dev",
                 val_pool_path="../data/ComplexWebQuestions/value_pool.json",
                 output="../data/ComplexWebQuestions/annotated_dev.pkl",
                 rel_topk=50,
                 kb_endpoint=kb_endpoint)

        annotate(dataset="cwq",
                 data_path="../data/ComplexWebQuestions/parsed_test.json",
                 ent_pool_path=None,
                 type_pool_path="../data/ComplexWebQuestions/candidate_types_test.json",
                 rel_pool_path="../data/ComplexWebQuestions/candidate_relations_test",
                 val_pool_path="../data/ComplexWebQuestions/value_pool.json",
                 output="../data/ComplexWebQuestions/annotated_test.pkl",
                 rel_topk=50,
                 kb_endpoint=kb_endpoint)

    elif args.dataset == "lcq":
        print("Now annotating LC-QuAD ... ")

        kb_endpoint = "http://10.201.61.163:8890//sparql"

        annotate(dataset="lcq",
                 data_path="../data/LC-QuAD/parsed_train.json",
                 ent_pool_path=None,
                 type_pool_path="../data/LC-QuAD/candidate_types_train.json",
                 rel_pool_path="../data/LC-QuAD/candidate_relations_train/",
                 val_pool_path=None,
                 output="../data/LC-QuAD/annotated_train.pkl",
                 rel_topk=50,
                 kb_endpoint=kb_endpoint,
                 training=True)

        annotate(dataset="lcq",
                 data_path="../data/LC-QuAD/parsed_dev.json",
                 ent_pool_path=None,
                 type_pool_path="../data/LC-QuAD/candidate_types_dev.json",
                 rel_pool_path="../data/LC-QuAD/candidate_relations_dev/",
                 val_pool_path=None,
                 output="../data/LC-QuAD/annotated_dev.pkl",
                 rel_topk=50,
                 kb_endpoint=kb_endpoint)

        annotate(dataset="lcq",
                 data_path="../data/LC-QuAD/parsed_test.json",
                 ent_pool_path=None,
                 type_pool_path="../data/LC-QuAD/candidate_types_test.json",
                 rel_pool_path="../data/LC-QuAD/candidate_relations_test/",
                 val_pool_path=None,
                 output="../data/LC-QuAD/annotated_test.pkl",
                 rel_topk=50,
                 kb_endpoint=kb_endpoint)

    elif args.dataset == "wsp":
        print("Now annotating WebQSP ... ")

        kb_endpoint = "http://10.201.7.66:8890//sparql"

        annotate(dataset="wsp",
                 data_path="../data/WebQSP/parsed_train.json",
                 ent_pool_path=None,
                 type_pool_path=None,
                 rel_pool_path="../data/WebQSP/candidate_relations_train",
                 val_pool_path="../data/WebQSP/value_pool.json",
                 output="../data/WebQSP/annotated_train.pkl",
                 rel_topk=50,
                 kb_endpoint=kb_endpoint,
                 training=True)

        annotate(dataset="wsp",
                 data_path="../data/WebQSP/parsed_test.json",
                 ent_pool_path=None,
                 type_pool_path=None,
                 rel_pool_path="../data/WebQSP/candidate_relations_test",
                 val_pool_path="../data/WebQSP/value_pool.json",
                 output="../data/WebQSP/annotated_test.pkl",
                 rel_topk=50,
                 kb_endpoint=kb_endpoint)

    else:
        pass