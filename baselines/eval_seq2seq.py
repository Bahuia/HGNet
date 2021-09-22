# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : eval_seq2seq.py
# @Software: PyCharm
"""


import re
import sys
import json
import copy
import pickle
import argparse
import Levenshtein
from operator import itemgetter
sys.path.append("..")
sys.path.append("../..")
from utils.utils import cal_scores, big_bracket_pattern
from utils.query_interface import KB_query


def normalize_pred_query(dataset, data, relations, pred_query):
    def get_entities(data):
        conds = data["query"]["where"]["notUnion"]
        for union_conds in data["query"]["where"]["union"]:
            conds += union_conds
        if len(data["query"]["where"]["subQueries"]) == 1:
            conds += data["query"]["where"]["subQueries"][0]["where"]["notUnion"]
        entities = []
        for type, cond in conds:
            if type == "Triple":
                if cond[0][0] != "?":
                    entities.append(cond[0].split("##")[0])
                if cond[2][0] != "?":
                    entities.append(cond[2].split("##")[0])
        entities = list(set(entities))
        return entities

    entities = get_entities(data)

    if dataset == "lcq":
        pred_query = pred_query.replace("?", " ?")
        pred_query = pred_query.replace("?uri", " ?ur")
        pred_query = pred_query.replace("?ur", " ?uri")
        pred_query = pred_query.replace(".", " . ")
        where_clauses = re.findall(big_bracket_pattern, pred_query)[0]
        where_clauses = where_clauses.strip(" ").strip(".").strip(" ")
        triples = [[y.strip(" ") for y in x.strip(" ").split(" ") if y != ""]
                   for x in where_clauses.split(". ")]
        triples = [x for x in triples if len(x) == 3]

        new_triples = []
        for s, p, o in triples:
            if p == "type":
                p = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
                o = "<http://dbpedia.org/ontology/" + o + ">"
            else:
                p = "<http://dbpedia.org/property/" + p + ">"
                if s[0] != "?":
                    s = "<http://dbpedia.org/resource/" + s + ">"
                    tmp = [[e, Levenshtein.distance(s, e)] for e in entities]
                    tmp.sort(key=lambda x: x[-1])
                    s = tmp[0][0]

                if o[0] != "?":
                    o = "<http://dbpedia.org/resource/" + o + ">"
                    tmp = [[e, Levenshtein.distance(o, e)] for e in entities]
                    tmp.sort(key=lambda x: x[-1])
                    o = tmp[0][0]
            new_triples.append([s, p, o])
        new_where_clauses = " . ".join([" ".join(x) for x in new_triples])
        pred_query = pred_query.replace(where_clauses, new_where_clauses)
        pred_query = pred_query.replace("  ", " ")
        pred_query = pred_query.replace("  ", " ")
        pred_query = pred_query.replace("(  ?", "(?")
    else:
        pred_query = pred_query.replace("?", " ?")
        pred_query = pred_query.replace("( ?", "(?")
        pred_query = pred_query.replace("_ ", "_")
        pred_query = pred_query.replace("  ", " ")

        seqs = pred_query.split(" ")
        new_seqs = []
        for x in seqs:
            if "." in x or "?" in x:
                x = x.lower()
            if "." in x and len(x) > 1:
                y = "ns:" + x
                if x[1] == ".": # entity
                    tmp = [[e, Levenshtein.distance(y, e)] for e in entities]
                    tmp.sort(key=lambda x: x[-1])
                    y = tmp[0][0]
                else: # relation
                    tmp = [[r,
                            Levenshtein.distance(y.split(".")[-1], r.split(".")[-1]),
                            Levenshtein.distance(y, r),] for r in relations]
                    tmp.sort(key=itemgetter(1, 2))
                    y = tmp[0][0]
                new_seqs.append(y)
            else:
                new_seqs.append(x)

        pred_query = " ".join(new_seqs)
        pred_query = pred_query.replace("WHERE", "\nWHERE")
        pred_query = pred_query.replace("ORDER BY", "\nORDER BY")
        pred_query = pred_query.replace("LIMIT", "\nLIMIT")
        pred_query = pred_query.replace("{", "{\n")
        pred_query = pred_query.replace("}", "\n}")
        pred_query = pred_query.replace("_ ?", "_ $\n?")
        pred_query = pred_query.replace("$", ".\n")
        pred_query = "PREFIX ns: <http://rdf.freebase.com/ns/>\n" + pred_query
    return pred_query

def find_property_pos(pred_query):
    indexs = []
    st_pos = pred_query.find("property", 0)
    while st_pos != -1:
        indexs.append(st_pos)
        st_pos = pred_query.find("property", st_pos + 1)
    return indexs


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='lcq', choices=['lcq', 'cwq', 'wsp'])
    arg_parser.add_argument('--test_data', type=str, default="")
    arg_parser.add_argument('--rel_pool', type=str, default="")
    arg_parser.add_argument('--sparql_path', type=str, default="")
    arg_parser.add_argument('--results_path', type=str, default="")
    arg_parser.add_argument('--kb_endpoint', type=str, default="")
    args = arg_parser.parse_args()

    test_data = json.load(open(args.test_data))
    pred_queries = []
    with open(args.sparql_path) as fin:
        for line in fin:
            pred_queries.append(line.strip("\n"))

    assert len(test_data) == len(pred_queries)

    if args.dataset != "lcq":
        relations = json.load(open(args.rel_pool))
    else:
        relations = None

    avg_p, avg_r, avg_f1, avg_hit_1 = 0, 0, 0, 0

    results = []

    for i, d in enumerate(test_data):

        # if d["id"] != "WebQTest-832_c334509bb5e02cacae1ba2e80c176499":
        #     continue
        gold_query = d["sparql"]
        _pred_query = pred_queries[i]

        pred_query = normalize_pred_query(args.dataset, d, relations, _pred_query)

        print(d["id"])
        # print(pred_query)
        # print()
        # # print(pred_answers)
        # print(gold_query)
        # print(gold_answers)

        try:
            ans_x = "uri" if args.dataset == "lcq" else "x"

            pred_answers = KB_query(pred_query, args.kb_endpoint)
            if args.dataset == "lcq":
                if len(pred_answers) == 0:
                    indexs = find_property_pos(pred_query)
                    for enum in range(1 << len(indexs)):
                        tmps = ["ontology" if (enum >> x) & 1 == 0 else "property" for x in range(len(indexs))]
                        tmp_pred_query = ""
                        last = 0
                        for j, st_pos in enumerate(indexs):
                            tmp_pred_query += pred_query[last:st_pos]
                            tmp_pred_query += tmps[j]
                            last = st_pos + len(tmps[j])
                        tmp_pred_query += pred_query[last:]
                        tmp_pred_answers = KB_query(tmp_pred_query, args.kb_endpoint)
                        if tmp_pred_answers:
                            pred_answers = copy.deepcopy(tmp_pred_answers)
                            break

            if 'ASK' not in pred_query and 'COUNT' not in pred_query:  # ask
                pred_answers = [r[ans_x] for r in pred_answers]

            gold_answers = KB_query(gold_query, args.kb_endpoint)
            if 'ASK' not in pred_query and 'COUNT' not in pred_query:  # ask
                gold_answers = [r[ans_x] for r in gold_answers]

            # print(pred_answers)
            # print(gold_answers)
            p, r, f1, hit_1 = cal_scores(pred_answers, gold_answers)
        except:
            pred_answers = []
            gold_answers = []

            p, r, f1, hit_1 = 0., 0., 0., 0.

        print(p, r, f1, hit_1)
        print()

        d["pred_query"] = pred_query
        d["pred_answers"] = [x for x in pred_answers]
        d["gold_answers"] = [x for x in gold_answers]
        d["score"] = [p, r, f1, hit_1]
        results.append(d)

        avg_p += p
        avg_r += r
        avg_f1 += f1
        avg_hit_1 += hit_1

    c_avg_p = avg_p * 100. / len(test_data)
    c_avg_r = avg_r * 100. / len(test_data)
    c_avg_f1 = avg_f1 * 100. / len(test_data)
    c_avg_hit_1 = avg_hit_1 * 100. / len(test_data)
    print(c_avg_p, c_avg_r, c_avg_f1, c_avg_hit_1)

    pickle.dump(results, open(args.results_path, "wb"))
    print("Results save to \"{}\"\n".format(args.results_path))
