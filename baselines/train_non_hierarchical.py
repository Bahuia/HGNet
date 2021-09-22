# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/30
# @Author  : Yongrui Chen
# @File    : train_non_hierarchical.py
# @Software: PyCharm
"""

import os
import sys
import time
import torch
import pickle

sys.path.append("..")
sys.path.append("../..")
import pargs
from data_loaders import NonHierarchicalGenerationDataLoader
from models.non_hierarchical_model import NonHierarchicalModel


if __name__ == '__main__':
    args = pargs.aqgnet_pargs()

    if args.use_bert:
        args.use_mention_feature = True

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

    train_datas = pickle.load(open(args.train_data, "rb"))
    train_loader = NonHierarchicalGenerationDataLoader(args)
    train_loader.load_data(train_datas, args.bs, use_small=args.use_small, shuffle=args.shuffle)
    print("Load training data from \"%s\"."% (args.train_data))
    print("training data, batch size: %d, batch number: %d" % (args.bs, train_loader.n_batch))

    valid_datas = pickle.load(open(args.valid_data, "rb"))
    valid_loader = NonHierarchicalGenerationDataLoader(args)
    valid_loader.load_data(valid_datas, bs=1, use_small=args.use_small, shuffle=False)
    print("Load valid data from \"%s\"." % (args.valid_data))
    print("valid data, batch size: %d, batch number: %d" % (1, valid_loader.n_batch))

    model = NonHierarchicalModel(args)
    if args.cuda:
        model.cuda()
        print('Shift model to GPU.\n')

    if args.cpt != "":
        # load pretrained checkpoint.
        model.load_state_dict(torch.load(args.cpt, map_location='cpu'))
        print("Load pre-trained checkpoint from \"%s\"." % os.path.abspath(args.cpt))
    else:
        # load pretrain embeddings.
        if not args.use_bert and args.use_glove:
            print('Loading pretrained word vectors from \"%s\" ...' % (args.glove_path))
            if os.path.isfile(args.emb_cache):
                pretrained_emb = torch.load(args.emb_cache)
                model.word_embedding.word_lookup_table.weight.data.copy_(pretrained_emb)
            else:
                pretrained_emb, random_init_words = model.word_embedding.load_pretrained_vectors(
                    args.glove_path, binary=False, normalize=args.word_normalize)
                torch.save(pretrained_emb, args.emb_cache)

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

    iters = 0
    start = time.time()
    best_val_q_acc = 0
    best_val_aqg_acc = 0

    header = '\n  Time  Epoch        Loss    Train_Step_Acc       Train_Acc     Valid_AQG_Acc        Valid_Acc'
    val_log_template = ' '.join(
        '{:>6.0f},{:>5.0f},{:>12.6f},{:16.4f},{:16.4f},{:16.4f},{:16.4f}'.split(','))
    best_snapshot_prefix = os.path.join(checkpoint_dir, 'best_snapshot')

    print('\nTraining start.')

    print(header)

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        avg_loss = 0.
        n_q_total, n_aqg_correct = 0, 0
        n_aqg_step_correct, n_aqg_step_total = 0, 0

        for i, b in enumerate(train_loader.next_batch()):
            data = b[-1]
            gold_obj_labels = b[-11]

            optimizer.zero_grad()
            loss, action_probs = model(b)

            loss = torch.mean(loss)

            # Caculate gradients and update parameters
            if i % args.ag == 0:
                optimizer.zero_grad()
                loss.backward()
                if args.ag == 1:
                    # clip the gradient.
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            elif i % args.ag == (args.ag - 1):
                loss.backward()
                # clip the gradient.
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            else:
                loss.backward()

            for s_id in range(len(gold_obj_labels)):
                is_aqg_correct = True
                for j in range(len(gold_obj_labels[s_id])):
                    pred_obj = torch.argmax(action_probs[s_id][j], dim=-1).item()
                    if pred_obj == gold_obj_labels[s_id][j]:
                        n_aqg_step_correct += 1
                    else:
                        is_aqg_correct = False
                        # if j == 0 or j % 3 == 1:
                        #     print(b[7][s_id][pred_obj], b[7][s_id][gold_obj_labels[s_id][j]])
                        # elif j % 3 == 0:
                        #     print(b[11][s_id][pred_obj], b[11][s_id][gold_obj_labels[s_id][j]])
                # if not is_aqg_correct:
                #     print(data[s_id]["id"])
                #     print([torch.argmax(action_probs[s_id][j], dim=-1).item() for j in range(len(gold_obj_labels[s_id]))])
                #     print(gold_obj_labels[s_id])
                #     print()
                n_aqg_step_total += len(gold_obj_labels[s_id])
                n_aqg_correct += is_aqg_correct

            avg_loss += loss.data.cpu().numpy() * len(data)
            n_q_total += len(data)

        avg_loss /= n_q_total

        train_aqg_acc = 100. * n_aqg_correct / n_q_total
        train_aqg_step_acc = 100. * n_aqg_step_correct / n_aqg_step_total

        model.eval()
        val_n_q_correct, val_n_aqg_correct, val_n_q_total = 0, 0, 0
        for b in valid_loader.next_batch():
            val_n_q_total += 1

            data = b[-1][0]
            pred_aqgs = model.generate(b, beam_size=args.beam_size)

            if not pred_aqgs:
                continue

            pred_aqg = pred_aqgs[0]

            try:
                aqg_matching, matching = pred_aqg.is_equal(data["gold_aqg"])
            except:
                val_n_aqg_correct += 0
                val_n_q_correct += 0
            else:
                val_n_aqg_correct += aqg_matching
                val_n_q_correct += matching

                # if matching:
                #     print(data["id"])
                #     pred_aqg.show_state()
                #     data["gold_aqg"].show_state()
                #     print()

        val_aqg_acc = val_n_aqg_correct * 100. / val_n_q_total
        val_q_acc = val_n_q_correct * 100. / val_n_q_total

        print(val_log_template.format(time.time() - start, epoch, avg_loss,
                                      train_aqg_step_acc, train_aqg_acc,
                                      val_aqg_acc, val_q_acc))

        # pickle.dump(sparql_cache, open(args.sparql_cache_path, "wb"))

        snapshot_path = best_snapshot_prefix + \
                        '_epoch_{}_val_aqg_acc_{}_val_acc_{}_model.pt'.format(epoch, val_aqg_acc, val_q_acc)
        # save model, delete previous 'best_snapshot' files.
        torch.save(model.state_dict(), snapshot_path)

        # # update checkpoint.
        # if args.save_best_aqg_acc:
        #     if val_aqg_acc >= best_val_aqg_acc:
        #         best_val_aqg_acc = val_aqg_acc
        #         snapshot_path = best_snapshot_prefix + \
        #                         '_epoch_{}_best_val_aqg_acc_{}_model.pt'.format(epoch, best_val_aqg_acc)
        #         # save model, delete previous 'best_snapshot' files.
        #         torch.save(model.state_dict(), snapshot_path)
        #         # for f in glob.glob(best_snapshot_prefix + '*'):
        #         #     if f != snapshot_path:
        #         #         os.remove(f)
        # else:
        #     if val_q_acc >= best_val_q_acc:
        #         best_val_q_acc = val_q_acc
        #         snapshot_path = best_snapshot_prefix + \
        #                         '_epoch_{}_best_val_acc_{}_model.pt'.format(epoch, best_val_q_acc)
        #         # save model, delete previous 'best_snapshot' files.
        #         torch.save(model.state_dict(), snapshot_path)
        #         # for f in glob.glob(best_snapshot_prefix + '*'):
        #         #     if f != snapshot_path:
        #         #         os.remove(f)

    print('\nTraining finished.')
    print("\nBest AQG Acc: {:.2f}\nModel writing to \"{}\"\n".format(best_val_q_acc, out_dir))