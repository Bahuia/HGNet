# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : preprocess.py
# @Software: PyCharm
"""

import os
import sys
import json
import copy
import pickle
import argparse
import jsonlines
sys.path.append("..")
from rules.sparql import SPARQLParserCWQ, SPARQLParserLCQ, SPARQLParserWSP
from utils.utils import tokenize_word_sentence, tokenize_word_sentence_plm, is_value, exact_sequence_matching
from utils.dictionary import init_vocab
from utils.query_interface import KB_query
from transformers import AutoTokenizer
from rules.grammar import get_relation_true_name, get_type_true_name


def prepare_data_for_bart_baseline(dataset, data_path, seq2seq_data_path):
    datas = json.load(open(data_path))
    with jsonlines.open(seq2seq_data_path, "w") as fout:
        for d in datas:
            sparql = copy.deepcopy(d["sparql"])
            if dataset == "lcq":
                prefix = ["<http://dbpedia.org/resource/", "<http://dbpedia.org/property/",
                          "<http://dbpedia.org/ontology/", "<http://www.w3.org/1999/02/22-rdf-syntax-ns#", ">"]
            else:
                prefix = ["PREFIX ns: <http://rdf.freebase.com/ns/>", "ns:",
                          "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))"]
                st = sparql.find("{\nFILTER (?x != ")
                ed = sparql.find(")\n", st)
                sparql = sparql.replace(sparql[st + 2:ed + 2], "")
                parser = SPARQLParserCWQ()
                sparql = parser.preprocess(sparql)
                sparql = sparql.replace(". ", "$ ")
            for p in prefix:
                sparql = sparql.replace(p, "")
            fout.write({"text": d["question"], "summary": sparql})

######################################## ComplexWebQuestions Preprocessing #############################################
def parse_sparql_for_cwq(source_path, data_path, relation_pool, training=False):
    source_datas = json.load(open(source_path))
    parser = SPARQLParserCWQ()
    processed_datas = []
    for data in source_datas:
        query, _ = parser.parse_sparql(data["sparql"], depth=0)
        new_data = {
            "id": data["ID"],
            "question": data["question"],
            "question_toks": tokenize_word_sentence(data["question"]),
            "question_toks_bert": tokenize_word_sentence_plm(data["question"], bert_tokenizer),
            "mention_feature": None,
            "mention_feature_bert": None,
            "sparql": data["sparql"],
            "query": query
        }
        processed_datas.append(new_data)
    json.dump(processed_datas, open(data_path, "w", encoding="utf-8"), indent=2)

    if training:
        word_vocab = mk_vocabs_for_cwq(processed_datas, relation_pool)
        print("Word vocabulary size: {}".format(len(word_vocab)))
        pickle.dump(word_vocab, open(os.path.join(vocab_dir, "word_vocab_cwq.pkl"), "wb"))
        print("Word vocabulary save to \"{}\".".format(
            os.path.abspath(os.path.join(vocab_dir, "word_vocab_cwq.pkl"))))

def build_relation_pool_for_cwq(kb_endpoint):
    query = "SELECT DISTINCT ?r WHERE {?x ?r ?y .}"
    results = KB_query(query, kb_endpoint)
    relation_pool = []
    for res in results:
        if "http://rdf.freebase.com/ns/" in res["r"]:
            relation_pool.append(res["r"].replace("http://rdf.freebase.com/ns/", "ns:"))
    relation_pool = set(relation_pool)
    relation_pool_1 = set()
    for rel in relation_pool:
        if rel.split(".")[-1] == "from":
            rel_to = ".".join(rel.split(".")[:-1] + ["to"])
            if rel_to in relation_pool:
                relation_pool_1.add(rel + "$$$" + rel_to.split(".")[-1])
        elif rel.split(".")[-1] == "start_date":
            rel_to = ".".join(rel.split(".")[:-1] + ["end_date"])
            if rel_to in relation_pool:
                relation_pool_1.add(rel + "$$$" + rel_to.split(".")[-1])
    _relation_pool = list(relation_pool) + list(relation_pool_1)
    _relation_pool.sort()
    return _relation_pool

def build_value_pool_for_cwq(data_path, out_path):
    datas = json.load(open(data_path, "r"))
    val_pool = []
    for i, d in enumerate(datas):
        if not d["query"]["where"]["union"]:
            d["query"]["where"]["union"].append([])
        for union_conds in d["query"]["where"]["union"]:
            conds = union_conds + d["query"]["where"]["notUnion"]
            for type, cond in conds:
                if type == "Triple" or type == "Comparison":
                    s, p, o = cond
                    if is_value(s) and "^^xsd:dateTime" not in s \
                        and not s.strip("\"").strip("\"@en").strip("-").replace(".", "").isdigit():
                        val_pool.append(s)
                    if is_value(o) and "^^xsd:dateTime" not in o \
                        and not o.strip("\"").strip("\"@en").strip("-").replace(".", "").isdigit():
                        val_pool.append(o)
    val_pool = list(set(val_pool))
    val_pool.sort()
    print("Value pool size: {}".format(len(val_pool)))
    json.dump(val_pool, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("Value pool save to \"{}\".".format(os.path.abspath(os.path.join(out_path))))

def mk_vocabs_for_cwq(processed_datas, relation_pool):
    word_vocab = init_vocab()
    for d in processed_datas:
        for tok in d["question_toks"]:
            word_vocab.add(tok)
    for r in relation_pool:
        for tok in get_relation_true_name(r, "freebase").split(" "):
            word_vocab.add(tok)
    return word_vocab

############################################## LC-QuAD Preprocessing ###################################################
def parse_sparql_for_lcq(source_path, data_path, individual_relation_pool_path, relation_pool, training=False):
    source_datas = json.load(open(source_path))
    parser = SPARQLParserLCQ()
    processed_datas = []

    for data in source_datas:

        query = parser.parse_sparql(data["query"])

        question = data["question"].lower()

        if data["entity1_mention"] != "":
            question = question.replace(data["entity1_mention"].lower(), "[unused0]")

        if data["entity2_mention"] != "":
            question = question.replace(data["entity2_mention"].lower(), "[unused0]")

        question_toks = tokenize_word_sentence(question)
        mention_feature = [0 for _ in range(len(question_toks))]

        question_toks_bert = tokenize_word_sentence_plm(question, bert_tokenizer)
        mention_feature_bert = [0 for _ in range(len(question_toks_bert))]

        if data["entity1_mention"] != "":
            ment_toks_1 = tokenize_word_sentence(data["entity1_mention"])
            question_toks += ["[sep]"] + ment_toks_1
            mention_feature += [1 for _ in range(len(ment_toks_1) + 1)]

            ment_toks_bert_1 = tokenize_word_sentence_plm(data["entity1_mention"], bert_tokenizer, start_cls=False)
            question_toks_bert += ["[sep]"] + ment_toks_bert_1
            mention_feature_bert += [1 for _ in range(len(ment_toks_bert_1) + 1)]

        if data["entity2_mention"] != "":
            ment_toks_2 = tokenize_word_sentence(data["entity2_mention"])
            question_toks += ["[sep]"] + ment_toks_2
            mention_feature += [1 for _ in range(len(ment_toks_2) + 1)]

            ment_toks_bert_2 = tokenize_word_sentence_plm(data["entity2_mention"], bert_tokenizer, start_cls=False)
            question_toks_bert += ["[sep]"] + ment_toks_bert_2
            mention_feature_bert += [1 for _ in range(len(ment_toks_bert_2) + 1)]

        new_data = {
            "id": data["id"],
            "question": data["question"],
            "question_toks": question_toks,
            "question_toks_bert": question_toks_bert,
            "mention_feature": mention_feature,
            "mention_feature_bert": mention_feature_bert,
            "sparql": data["query"],
            "query": query
        }
        processed_datas.append(new_data)
    json.dump(processed_datas, open(data_path, "w", encoding="utf-8"), indent=2)

    build_individual_relation_pool_for_lcq(source_datas, individual_relation_pool_path, kb_endpoint)

    if training:
        word_vocab = mk_vocabs_for_lcq(processed_datas, relation_pool)
        print("Word vocabulary size: {}".format(len(word_vocab)))
        pickle.dump(word_vocab, open(os.path.join(vocab_dir, "word_vocab_lcq.pkl"), "wb"))
        print("Word vocabulary save to \"{}\".".format(
            os.path.abspath(os.path.join(vocab_dir, "word_vocab_lcq.pkl"))))

def build_relation_pool_for_lcq(kb_endpoint):
    query = "SELECT DISTINCT ?r from <http://dbpedia.org/> WHERE {?x ?r ?y .}"
    results = KB_query(query, kb_endpoint)
    relation_pool = []
    for res in results:
        if "http://dbpedia.org/property/" in res["r"] or "http://dbpedia.org/ontology/" in res["r"]:
            relation_pool.append(res["r"])
    relation_pool.append("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    relation_pool = list(set(relation_pool))
    relation_pool.sort()
    return relation_pool

def build_individual_relation_pool_for_lcq(source_datas, output_path, kb_endpoint):
    queries = [
        "SELECT DISTINCT ?r FROM <dbpedia> WHERE { #ent# ?r ?x .}",
        "SELECT DISTINCT ?r FROM <dbpedia> WHERE { ?x ?r #ent# .}",
        "SELECT DISTINCT ?r1 ?r2 FROM <dbpedia> WHERE { #ent# ?r1 ?x . ?x ?r2 ?y .}",
        "SELECT DISTINCT ?r1 ?r2 FROM <dbpedia> WHERE { #ent# ?r1 ?x . ?y ?r2 ?x .}",
        "SELECT DISTINCT ?r1 ?r2 FROM <dbpedia> WHERE { ?x ?r1 #ent# . ?y ?r2 ?x .}",
        "SELECT DISTINCT ?r1 ?r2 FROM <dbpedia> WHERE { ?x ?r1 #ent# . ?x ?r2 ?y .}"
    ]
    relation_pool = []
    for data in source_datas:
        print(data["id"])
        ents = []
        rels = []
        if data["entity1_uri"] != "":
            ents.append("<" + data["entity1_uri"] + ">")
        if data["entity2_uri"] != "":
            ents.append("<" + data["entity2_uri"] + ">")
        for ent in ents:
            for query in queries:
                _query = copy.deepcopy(query)
                _query = _query.replace("#ent#", ent)
                results = KB_query(_query, kb_endpoint)
                for r in results:
                    for k, v in r.items():
                        rels.append(v)
                rels = [r for r in rels if "http://dbpedia.org/property/" in r
                        or "http://dbpedia.org/ontology/" in r]
        rels = list(set(rels))
        rels.sort()
        relation_pool.append({"id": data["id"], "relation_pool": rels})

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, d in enumerate(relation_pool):
        if i % 100 == 0:
            out_dir = os.path.join(output_path, str(i) + "-" + str(i + 99))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        out_path = os.path.join(out_dir, str(d["id"]) + ".json")
        json.dump(d, open(out_path, "w"), indent=4)

def build_type_pool_for_lcq(in_path, out_path):

    test_mode = "test" in in_path.split("/")[-1]
    datas = json.load(open(in_path))
    cand_types = []

    if not test_mode:
        for d in datas:
            cand_types.append({"id": d["id"], "candidate_types": [x for x in d["cand_types"]]})
    else:
        for d in datas:
            if d["cand_types"][0][0] == "http://dbpedia.org/ontology/None" \
                    and d["cand_types"][0][1] - d["cand_types"][1][1] > 0.42:
                cands = []
            else:
                cands =[x[0] for x in d["cand_types"] if x[0] != "http://dbpedia.org/ontology/None"]

            q = d["question"].lower()

            if len(cands) > 0:
                cands_tmp = [cands[0]]
            else:
                cands_tmp = []
            for one_type in cands[1:]:
                t_toks = tokenize_word_sentence(get_type_true_name(one_type, kb="dbpedia"))
                q_toks = tokenize_word_sentence(q)
                if exact_sequence_matching(t_toks, q_toks):
                    cands_tmp.append(one_type)

            cands_tmp = list(set(cands_tmp))
            cand_types.append({"id": d["id"], "candidate_types": cands_tmp})

    json.dump(cand_types, open(out_path, "w"), indent=2)

def mk_vocabs_for_lcq(processed_datas, relation_pool):
    word_vocab = init_vocab()
    for d in processed_datas:
        for tok in d["question_toks"]:
            word_vocab.add(tok)
    for r in relation_pool:
        for tok in get_relation_true_name(r, "dbpedia").split(" "):
            word_vocab.add(tok)
    return word_vocab

############################################## WebQSP Preprocessing ####################################################
def parse_sparql_for_wsp(source_path, data_path, relation_pool, training=False):
    source_datas = json.load(open(source_path))
    parser = SPARQLParserWSP()
    processed_datas = []
    for data in source_datas["Questions"]:

        if not data["Parses"][0]["InferentialChain"]:
            continue

        question = data["ProcessedQuestion"]
        ment = data["Parses"][0]["PotentialTopicEntityMention"]
        if ment is not None:
            question = question.replace(ment, "[unused0]")

        question_toks = tokenize_word_sentence(question)
        mention_feature = [0 for _ in range(len(question_toks))]

        question_toks_bert = tokenize_word_sentence_plm(question, bert_tokenizer)
        mention_feature_bert = [0 for _ in range(len(question_toks_bert))]

        if ment is not None:
            ment_toks = tokenize_word_sentence(ment)
            question_toks += ["[sep]"] + ment_toks
            mention_feature += [1 for _ in range(len(ment_toks) + 1)]

            ment_toks_bert = tokenize_word_sentence_plm(ment, bert_tokenizer, start_cls=False)
            question_toks_bert += ["[sep]"] + ment_toks_bert
            mention_feature_bert += [1 for _ in range(len(ment_toks_bert) + 1)]

        query = parser.parse_sparql(data["Parses"][0])
        new_data = {
            "id": data["QuestionId"],
            "question": data["ProcessedQuestion"],
            "question_toks": question_toks,
            "question_toks_bert": question_toks_bert,
            "mention_feature": mention_feature,
            "mention_feature_bert": mention_feature_bert,
            "sparql": data["Parses"][0]["Sparql"],
            "query": query
        }
        processed_datas.append(new_data)
    json.dump(processed_datas, open(data_path, "w", encoding="utf-8"), indent=2)

    if training:
        word_vocab = mk_vocab_for_wsp(processed_datas, relation_pool)
        print("Word vocabulary size: {}".format(len(word_vocab)))
        pickle.dump(word_vocab, open(os.path.join(vocab_dir, "word_vocab_wsp.pkl"), "wb"))
        print("Word vocabulary save to \"{}\".".format(
            os.path.abspath(os.path.join(vocab_dir, "word_vocab_wsp.pkl"))))

def build_relation_pool_for_wsp(kb_endpoint):
    query = "SELECT DISTINCT ?r WHERE {?x ?r ?y .}"
    results = KB_query(query, kb_endpoint)
    relation_pool = []
    for res in results:
        if "http://rdf.freebase.com/ns/" in res["r"]:
            relation_pool.append(res["r"].replace("http://rdf.freebase.com/ns/", "ns:"))
    relation_pool = set(relation_pool)
    relation_pool_1 = set()
    for rel in relation_pool:
        if rel.split(".")[-1] == "from":
            rel_to = ".".join(rel.split(".")[:-1] + ["to"])
            if rel_to in relation_pool:
                relation_pool_1.add(rel + "$$$" + rel_to.split(".")[-1])
        elif rel.split(".")[-1] == "start_date":
            rel_to = ".".join(rel.split(".")[:-1] + ["end_date"])
            if rel_to in relation_pool:
                relation_pool_1.add(rel + "$$$" + rel_to.split(".")[-1])

    _relation_pool = list(relation_pool) + list(relation_pool_1)
    _relation_pool.sort()
    return _relation_pool

def build_mention_pool_for_wsp(path):
    ment_pool = {}
    with open(path, "r", encoding="utf-8") as fin:
        for row in fin:
            row = row.split("\t")
            idx = row[0]
            ment = row[1]
            if idx not in ment_pool:
                ment_pool[idx] = set()
            ment_pool[idx].add(ment)
    return ment_pool

def build_value_pool_for_wsp(data_path, out_path):
    datas = json.load(open(data_path, "r"))
    val_pool = []
    for i, d in enumerate(datas):
        if not d["query"]["where"]["union"]:
            d["query"]["where"]["union"].append([])
        for union_conds in d["query"]["where"]["union"]:
            conds = union_conds + d["query"]["where"]["notUnion"]
            for type, cond in conds:
                if type == "Triple" or type == "Comparison":
                    s, p, o = cond
                    if is_value(s) and "^^xsd:dateTime" not in s \
                        and not s.strip("\"").strip("\"@en").strip("-").replace(".", "").isdigit():
                        val_pool.append(s)
                    if is_value(o) and "^^xsd:dateTime" not in o \
                        and not o.strip("\"").strip("\"@en").strip("-").replace(".", "").isdigit():
                        val_pool.append(o)
    val_pool = list(set(val_pool))
    val_pool.sort()
    print("Value pool size: {}".format(len(val_pool)))
    json.dump(val_pool, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("Value pool save to \"{}\".".format(os.path.abspath(os.path.join(out_path))))

def mk_vocab_for_wsp(processed_datas, relation_pool):
    word_vocab = init_vocab()
    for d in processed_datas:
        for tok in d["question_toks"]:
            word_vocab.add(tok)
    for r in relation_pool:
        for tok in get_relation_true_name(r, "freebase").split(" "):
            word_vocab.add(tok)
    return word_vocab

def mk_bert_vocab(bert_tokenizer):
    word_vocab = init_vocab(pad="[PAD]", unk="[UNK]")
    words = []
    for token, index in bert_tokenizer.vocab.items():
        words.append([token, index])
    words.sort(key=lambda x: x[1])
    for token, index in words:
        word_vocab.add(token)
    return word_vocab


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='lcq', choices=['lcq', 'cwq', 'wsp'])
    arg_parser.add_argument('--bert_mode', type=str, default='bert-base-uncased', choices=['bert-base-uncased', 'bert-large-uncased'])
    args = arg_parser.parse_args()

    vocab_dir = "../vocab/"
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    word_vocab = mk_bert_vocab(bert_tokenizer)
    print("BERT Word vocabulary size: {}".format(len(word_vocab)))
    pickle.dump(word_vocab, open(os.path.join(vocab_dir, "word_vocab_bert.pkl"), "wb"))
    print("BERT Word vocabulary save to \"{}\".".format(
        os.path.abspath(os.path.join(vocab_dir, "word_vocab_bert.pkl"))))

    if args.dataset == "cwq":
        print("Now preprocessing ComplexWebQuestions ... ")

        kb_endpoint = "http://10.201.69.194:8890//sparql"

        rel_pool = build_relation_pool_for_cwq(kb_endpoint)
        rel_pool_path = "../data/ComplexWebQuestions/relation_pool.json"
        json.dump(rel_pool, open(rel_pool_path, "w", encoding="utf-8"), indent=2)
        rel_pool = json.load(open(rel_pool_path))

        parse_sparql_for_cwq("../data/ComplexWebQuestions/ComplexWebQuestions_train.json",
                             "../data/ComplexWebQuestions/parsed_train.json",
                             rel_pool,
                             training=True)
        parse_sparql_for_cwq("../data/ComplexWebQuestions/ComplexWebQuestions_dev.json",
                             "../data/ComplexWebQuestions/parsed_dev.json",
                             rel_pool)
        parse_sparql_for_cwq("../data/ComplexWebQuestions/ComplexWebQuestions_test.json",
                             "../data/ComplexWebQuestions/parsed_test.json",
                             rel_pool)

        build_value_pool_for_cwq("../data/ComplexWebQuestions/parsed_train.json",
                                 "../data/ComplexWebQuestions/value_pool.json")

        prepare_data_for_bart_baseline("cwq",
                                       "../data/ComplexWebQuestions/parsed_train.json",
                                       "../data/ComplexWebQuestions/seq2seq_train.json")
        prepare_data_for_bart_baseline("cwq",
                                       "../data/ComplexWebQuestions/parsed_dev.json",
                                       "../data/ComplexWebQuestions/seq2seq_dev.json")
        prepare_data_for_bart_baseline("cwq",
                                       "../data/ComplexWebQuestions/parsed_test.json",
                                       "../data/ComplexWebQuestions/seq2seq_test.json")

    elif args.dataset == "lcq":
        print("Now preprocessing LC-QuAD ... ")

        kb_endpoint = "http://10.201.102.90:8890//sparql"

        rel_pool = build_relation_pool_for_lcq(kb_endpoint)
        rel_pool_path = "../data/LC-QuAD/relation_pool.json"
        json.dump(rel_pool, open(rel_pool_path, "w", encoding="utf-8"), indent=2)
        rel_pool = json.load(open(rel_pool_path))

        parse_sparql_for_lcq("../data/LC-QuAD/LC-QuAD_train.json",
                             "../data/LC-QuAD/parsed_train.json",
                             "../data/LC-QuAD/individual_relation_pool_train",
                             rel_pool,
                             training=True)
        parse_sparql_for_lcq("../data/LC-QuAD/LC-QuAD_dev.json",
                             "../data/LC-QuAD/parsed_dev.json",
                             "../data/LC-QuAD/individual_relation_pool_dev",
                             rel_pool)
        parse_sparql_for_lcq("../data/LC-QuAD/LC-QuAD_test.json",
                             "../data/LC-QuAD/parsed_test.json",
                             "../data/LC-QuAD/individual_relation_pool_test",
                             rel_pool)

        prepare_data_for_bart_baseline("lcq",
                                       "../data/LC-QuAD/parsed_train.json",
                                       "../data/LC-QuAD/seq2seq_train.json")
        prepare_data_for_bart_baseline("lcq",
                                       "../data/LC-QuAD/parsed_dev.json",
                                       "../data/LC-QuAD/seq2seq_dev.json")
        prepare_data_for_bart_baseline("lcq",
                                       "../data/LC-QuAD/parsed_test.json",
                                       "../data/LC-QuAD/seq2seq_test.json")

    elif args.dataset == "wsp":
        print("Now preprocessing WebQSP ... ")

        kb_endpoint = "http://10.201.69.194:8890//sparql"

        rel_pool = build_relation_pool_for_wsp(kb_endpoint)
        rel_pool_path = "../data/WebQSP/relation_pool.json"
        json.dump(rel_pool, open(rel_pool_path, "w", encoding="utf-8"), indent=2)
        rel_pool = json.load(open(rel_pool_path))

        parse_sparql_for_wsp("../data/WebQSP/WebQSP.train.json",
                             "../data/WebQSP/parsed_train.json",
                             rel_pool,
                             training=True)

        parse_sparql_for_wsp("../data/WebQSP/WebQSP.test.json",
                             "../data/WebQSP/parsed_test.json",
                             rel_pool)
        build_value_pool_for_wsp("../data/WebQSP/parsed_train.json",
                                 "../data/WebQSP/value_pool.json")

        prepare_data_for_bart_baseline("wsp",
                                       "../data/WebQSP/parsed_train.json",
                                       "../data/WebQSP/seq2seq_train.json")
        prepare_data_for_bart_baseline("wsp",
                                       "../data/WebQSP/parsed_test.json",
                                       "../data/WebQSP/seq2seq_test.json")
    else:
        pass