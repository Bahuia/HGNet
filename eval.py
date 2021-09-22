# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/31
# @Author  : Yongrui Chen
# @File    : eval.py
# @Software: PyCharm
"""

import os
import sys
import torch
import pickle
import datetime

sys.path.append("..")
import pargs
from data_loaders import AQGGenerationDataLoader
from models.model import AQGNet
from utils.utils import cal_scores
from utils.query_interface import KB_query


if __name__ == '__main__':
    args = pargs.aqgnet_pargs()

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('\nNote: You are using GPU for evaluation.\n')
        torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available() and not args.cuda:
        print('\nWarning: You have Cuda but do not use it. You are using CPU for evaluation.\n')

    if args.dataset == "lcq":
        kb = "dbpedia"
    else:
        kb = "freebase"

    test_data = pickle.load(open(args.test_data, "rb"))

    test_loader = AQGGenerationDataLoader(args)
    test_loader.load_data(test_data, bs=1, use_small=args.use_small, shuffle=False)
    print("Load valid data from \"%s\"." % (args.test_data))
    print("Test data, batch size: %d, batch number: %d" % (1, test_loader.n_batch))

    model = AQGNet(args)
    model.load_state_dict(torch.load(args.cpt, map_location='cpu'))
    print("Load checkpoint from \"%s\"." % os.path.abspath(args.cpt))
    if args.cuda:
        model.cuda()
        print('Shift model to GPU.\n')

    model.eval()

    if not os.path.exists(args.sparql_cache_path):
        sparql_cache = {}
        pickle.dump(sparql_cache, open(args.sparql_cache_path, "wb"))
    else:
        sparql_cache = pickle.load(open(args.sparql_cache_path, "rb"))

    # in_rels = pickle.load(open("/home/cyr/projects/AQGNet_v2/cache/cwq/in_relations.pkl", "rb"))
    # out_rels = pickle.load(open("/home/cyr/projects/AQGNet_v2/cache/cwq/out_relations.pkl", "rb"))

    results = []
    avg_p, avg_r, avg_f1, avg_hit_1 = 0, 0, 0, 0
    test_n_q_correct, test_n_aqg_correct, test_n_q_total = 0, 0, 0

    st_time = datetime.datetime.now()

    with torch.no_grad():

        for b in test_loader.next_batch():
            st = datetime.datetime.now()
            test_n_q_total += 1
            data = b[-1][0]
            gold_query = data["sparql"]

            print("----------------------------", data["id"])

            pred_aqgs = model.generate(b,
                                       beam_size=args.beam_size,
                                       sparql_cache=sparql_cache)

            if not pred_aqgs:
                continue

            pred_aqg = pred_aqgs[0]
            pred_aqg.normalize(args.dataset)

            try:
                aqg_matching, matching = pred_aqg.is_equal(data["gold_aqg"])
            except:
                aqg_matching, matching = False, False

            try:
                pred_query = pred_aqg.to_final_sparql_query(kb=kb)
            except:
                pred_query = None

            if matching:
                pred_answers = []
                gold_answers = []

                p, r, f1, hit_1 = 1., 1., 1., 1.

            else:
                if pred_query:
                    try:
                        ans_x = "uri" if args.dataset == "lcq" else "x"

                        pred_answers = KB_query(pred_query, args.kb_endpoint)
                        if 'ASK' not in pred_query and 'COUNT' not in pred_query:  # ask
                            pred_answers = [r["x_0"] for r in pred_answers]

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
                else:

                    pred_answers = []
                    gold_answers = []

                    p, r, f1, hit_1 = 0., 0., 0., 0.

            data["pred_aqg"] = pred_aqg
            data["pred_query"] = pred_query
            data["pred_answers"] = [x for x in pred_answers]
            data["gold_answers"] = [x for x in gold_answers]
            data["matching"] = [aqg_matching, matching]
            data["score"] = [p, r, f1, hit_1]
            results.append(data)

            test_n_aqg_correct += aqg_matching
            test_n_q_correct += matching

            avg_p += p
            avg_r += r
            avg_f1 += f1
            avg_hit_1 += hit_1

            c_avg_p = avg_p * 100. / test_n_q_total
            c_avg_r = avg_r * 100. / test_n_q_total
            c_avg_f1 = avg_f1 * 100. / test_n_q_total
            c_avg_hit_1 = avg_hit_1 * 100. / test_n_q_total


        test_aqg_acc = test_n_aqg_correct * 100. / test_n_q_total
        test_q_acc = test_n_q_correct * 100. / test_n_q_total

        avg_p = avg_p * 100. / test_n_q_total
        avg_r = avg_r * 100. / test_n_q_total
        avg_f1 = avg_f1 * 100. / test_n_q_total
        avg_hit_1 = avg_hit_1 * 100. / test_n_q_total

        print("AQG Acc. :", test_aqg_acc)
        print("Query Graph Acc. :", test_q_acc)
        print("Prec. :", avg_p)
        print("Rec. :", avg_r)
        print("F1-score :", avg_f1)
        print("Hit@1 :", avg_hit_1)

        # pickle.dump(sparql_cache, open(args.sparql_cache_path, "wb"))

        checkpoint_dir = '/'.join(args.cpt.split('/')[:-2])
        results_path = os.path.join(checkpoint_dir, args.result_name)
        pickle.dump(results, open(results_path, "wb"))
        print("Results save to \"{}\"\n".format(results_path))
