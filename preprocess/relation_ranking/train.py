# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : train.py
# @Software: PyCharm
"""

import os
import sys
import time
import json
import glob
import copy
import torch
import pickle
import random

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
        print('\nNote: You are using GPU for training.\n')
        torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available() and not args.cuda:
        print('\nWarning: You have Cuda but do not use it. You are using CPU for training.\n')

    print("#########################################################")
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print("#########################################################\n")

    wo_vocab = pickle.load(open(args.wo_vocab, 'rb'))
    print("load word vocab, size: %d" % len(wo_vocab))

    train_datas = json.load(open(args.train_data, "r"))
    if args.shuffle:
        random.shuffle(train_datas)

    train_loader = RelationRankingDataLoader(args, mode="train")
    train_loader.load_data(train_datas, args.bs, use_small=args.use_small,
                           shuffle=args.shuffle, training=True)
    print("Load training data from \"%s\"." % (args.train_data))
    print("training data, batch size: %d, batch number: %d" % (args.bs, train_loader.n_batch))

    valid_datas = json.load(open(args.valid_data, "r"))
    valid_loader = RelationRankingDataLoader(args, mode="dev")
    valid_loader.load_data(valid_datas, bs=1, use_small=args.use_small, shuffle=False)
    print("Load valid data from \"%s\"." % (args.valid_data))
    print("valid data, batch size: %d, batch number: %d" % (1, valid_loader.n_batch))
    rel_pool = copy.deepcopy(valid_loader.cand_pool)

    model = RankingModel(wo_vocab, args)
    if args.cuda:
        model.cuda()
        print('Shift model to GPU.\n')

    # load pretrain embeddings.
    if args.glove:
        print('Loading pretrained word vectors from \"%s\" ...' % (args.glove_path))
        if os.path.isfile(args.emb_cache):
            pretrained_emb = torch.load(args.emb_cache)
            model.word_embedding.word_lookup_table.weight.data.copy_(pretrained_emb)
        else:
            pretrained_emb, random_init_words = model.word_embedding.load_pretrained_vectors(
                args.glove_path, binary=False, normalize=args.word_normalize)
            torch.save(pretrained_emb, args.emb_cache)

    # loss function
    criterion = torch.nn.MarginRankingLoss(args.margin)

    # optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # create runs directory.
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', args.dataset, timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('\nModel writing to \"{}\"\n'.format(out_dir))
    with open(os.path.join(out_dir, 'param.log'), 'w') as fin:
        fin.write(str(args))
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for name, value in model.named_parameters():
        print(name, value.size())

    iters = 0
    start = time.time()
    best_val_recall = 0

    header = '\n  Time Epoch         Loss        Valid_Recall'
    val_log_template = ' '.join(
        '{:>6.0f},{:>5.0f},{:>12.6f},{:16.4f}'.split(','))
    best_snapshot_prefix = os.path.join(checkpoint_dir, 'best_snapshot')

    print('\nTraining start.')

    print(header)

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        avg_loss = 0.
        n_pair_total = 0
        for i, b in enumerate(train_loader.next_batch()):
            data = b[-1]
            optimizer.zero_grad()

            pos_scores, neg_scores = model(b[:-1])

            pos_scores = torch.cat(pos_scores, 0)
            neg_scores = torch.cat(neg_scores, 0)
            ones = torch.ones(pos_scores.size(0))
            if args.cuda:
                ones = ones.cuda()

            loss = criterion(pos_scores, neg_scores, ones)
            loss.backward()
            # clip the gradient.
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            avg_loss += loss.data.cpu().numpy() * len(data) * args.ns
            n_pair_total += len(data) * args.ns
        avg_loss /= n_pair_total

        model.eval()

        val_recall = 0
        val_n_total = 0
        for s in valid_loader.next_batch():
            data = s[-1][0]

            scores = model.ranking(s[:-1])

            if args.dataset == "lcq":
                results = [[rel_pool[data["id"]][i], float(scores[i].cpu().detach().numpy())] for i in range(len(rel_pool[data["id"]]))]
            else:
                results = [[rel_pool[i], float(scores[i].cpu().detach().numpy())] for i in range(len(rel_pool))]

            results = [x[0] for x in sorted(results, key=lambda x: x[-1], reverse=True)[:args.rel_topk]]

            _, recall, _, _ = cal_scores([x.strip("<").strip(">") for x in results], data["gold_relations"])

            val_recall += recall
            val_n_total += 1

        val_recall = val_recall * 100. / val_n_total

        print(val_log_template.format(time.time() - start, epoch, avg_loss, val_recall))

        # update checkpoint.
        if val_recall >= best_val_recall:
            best_val_recall = val_recall
            snapshot_path = best_snapshot_prefix + \
                            '_epoch_{}_best_val_recall_{}_model.pt'.format(epoch, best_val_recall)
            # save model, delete previous 'best_snapshot' files.
            torch.save(model.state_dict(), snapshot_path)
            for f in glob.glob(best_snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

    print('\nTraining finished.')
    print("\nBest {} candidate relation recall: {:.2f}\nModel writing to \"{}\"\n".format(args.rel_topk,
                                                                                          best_val_recall, out_dir))