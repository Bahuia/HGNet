# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : predict.py
# @Software: PyCharm
"""

import os
import sys
import copy
import json
import torch
import pickle
sys.path.append("..")
sys.path.append("../..")
import pargs
from data_loaders import RelationRankingDataLoader
from models.ranking_model import RankingModel
from utils.utils import cal_scores


if __name__ == '__main__':

    args = pargs.relation_ranking_pargs()

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('\nNote: You are using GPU for evaluation.\n')
        torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available() and not args.cuda:
        print('\nWarning: You have Cuda but do not use it. You are using CPU for evaluation.\n')

    wo_vocab = pickle.load(open(args.wo_vocab, 'rb'))
    print("load word vocab, size: %d" % len(wo_vocab))

    test_datas = json.load(open(args.test_data, "r"))
    test_datas = test_datas[args.st_pos : args.ed_pos]

    if "train" in args.output:
        mode = "train"
    elif "dev" in args.output:
        mode = "dev"
    else:
        mode = "test"

    test_loader = RelationRankingDataLoader(args, mode=mode)
    test_loader.load_data(test_datas, bs=1, use_small=args.use_small, shuffle=False)
    print("Test data, batch size: %d, batch number: %d" % (1, test_loader.n_batch))
    rel_pool = copy.deepcopy(test_loader.cand_pool)

    model = RankingModel(wo_vocab, args)
    model.load_state_dict(torch.load(args.cpt, map_location='cpu'))
    print("Load checkpoint from \"%s\"." % os.path.abspath(args.cpt))

    if args.cuda:
        model.cuda()
        print('Shift model to GPU.\n')

    model.eval()

    test_recall = 0
    test_n_total = 0
    cand_relations = []
    for s in test_loader.next_batch():
        data = s[-1][0]

        scores = model.ranking(s[:-1])

        if args.dataset == "lcq":
            results = [[rel_pool[data["id"]][i], float(scores[i].cpu().detach().numpy())] for i in
                       range(len(rel_pool[data["id"]]))]
        else:
            results = [[rel_pool[i], float(scores[i].cpu().detach().numpy())] for i in range(len(rel_pool))]


        results = [x for x in sorted(results, key=lambda x: x[-1], reverse=True)[:args.rel_topk]]

        _, recall, _, _ = cal_scores([x[0].strip("<").strip(">") for x in results],
                                     data["gold_relations"])

        cand_relations.append({"id": data["id"], "candidate_relations": results})
        test_recall += recall
        test_n_total += 1

    test_recall = test_recall * 100. / test_n_total

    for i, d in enumerate(cand_relations):
        pos = args.st_pos + i
        if pos % 100 == 0:
            out_dir = os.path.join(args.output, str(pos) + "-" + str(pos + 99))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        out_path = os.path.join(out_dir, str(d["id"]) + ".json")
        json.dump(d, open(out_path, "w"), indent=4)

    print("\nAverage Recall: %.2f" % test_recall)
    print("Results save to \"{}\"".format(args.output))
