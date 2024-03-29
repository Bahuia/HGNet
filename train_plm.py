# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/31
# @Author  : Yongrui Chen
# @File    : main_train_plm.py
# @Software: PyCharm
"""

import os
import sys
import time
import glob
import torch
import pickle
sys.path.append("..")
import pargs
from data_loaders import HGNetDataLoader
from models.model_plm import HGNet
from utils.utils import update_model, eval_train_accuracy


if __name__ == '__main__':
    args = pargs.hgnet_pargs()

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

    train_datas = pickle.load(open(args.train_path, "rb"))
    train_loader = HGNetDataLoader(args)
    train_loader.load_data(train_datas, args.bs, args.training_proportion, use_small=args.toy_size, shuffle=args.shuffle)
    print("Load training data from \"%s\"."% (args.train_path))
    print("training data, batch size: %d, batch number: %d" % (args.bs, train_loader.n_batch))

    valid_datas = pickle.load(open(args.valid_path, "rb"))
    valid_loader = HGNetDataLoader(args)
    valid_loader.load_data(valid_datas, bs=50, use_small=args.toy_size, shuffle=False)
    print("Load valid data from \"%s\"." % (args.valid_path))
    print("valid data, batch size: %d, batch number: %d" % (1, valid_loader.n_batch))

    model = HGNet(args)
    if args.cuda:
        model.cuda()
        print('Shift model to GPU.\n')

    if args.cpt_path != "":
        # load pretrained checkpoint.
        model.load_state_dict(torch.load(args.cpt_path, map_location='cpu'))
        print("Load pre-trained checkpoint from \"%s\"." % os.path.abspath(args.cpt_path))

    # optimizer.
    plm_params = list(map(id, model.encoder.parameters()))
    other_params = filter(lambda p: id(p) not in plm_params, model.parameters())
    params = [
        {"params": other_params, "lr": args.lr},
        {"params": model.encoder.parameters(), "lr": args.lr_plm}
    ]
    optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)

    # create runs directory.
    if args.save_cpt:
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
        best_snapshot_prefix = os.path.join(checkpoint_dir, 'best_snapshot')
    else:
        print('\nModel is not saved.\n')

    iters = 0
    start_time = time.time()
    best_val_q_acc = 0
    best_val_aqg_acc = 0

    header = '\n  Time  Epoch        Loss    Train_AQG_Step_Acc    Train_AQG_Acc' \
             '    Train_V_Step_Acc    Train_E_Step_Acc        Train_Acc     Valid_AQG_Acc       Valid_Acc'
    val_log_template = ' '.join(
        '{:>6.0f},{:>5.0f},{:>12.6f},{:20.4f},{:16.4f},{:19.4f},{:19.4f},{:16.4f},{:16.4f},{:16.4f}'.split(','))

    print('\nTraining start.')

    print(header)

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        avg_loss = 0.
        n_q_total = 0

        tgt_objs_records = []
        action_probs_records = []

        for i, b in enumerate(train_loader.next_batch()):
            data = b[-1]
            tgt_objs, tgt_v_ins_objs, tgt_e_ins_objs = b[-7:-4]

            loss, action_probs, v_action_probs, e_action_probs = model(batch=b)

            loss = torch.mean(loss)
            loss.backward()

            update_model(step=i,
                         model=model,
                         accumulation_steps=args.ag,
                         optimizer=optimizer,
                         clip_grad=args.clip_grad)

            tgt_objs_records.append([tgt_objs, tgt_v_ins_objs, tgt_e_ins_objs])
            action_probs_records.append([action_probs, v_action_probs, e_action_probs])

            avg_loss += loss.data.cpu().numpy() * len(data)
            n_q_total += len(data)

        avg_loss /= n_q_total

        train_aqg_acc, train_aqg_step_acc, \
        train_v_step_acc, train_e_step_acc, train_q_acc = eval_train_accuracy(tgt_objs_records=tgt_objs_records,
                                                                              action_probs_records=action_probs_records)

        model.eval()
        val_n_q_correct, val_n_aqg_correct, val_total = 0, 0, 0

        if epoch >= args.n_valid_epochs:
            for b in valid_loader.next_batch():
                data = b[-1]
                val_total += len(data)

                with torch.no_grad():
                    beams = model.generate(sample=b,
                                           max_beam_size=args.beam_size)

                for sid in range(len(data)):
                    if not beams[sid]:
                        continue

                    try:
                        aqg_matching, matching = beams[sid][0].cur_aqg.is_equal(data[sid]["gold_aqg"])
                    except:
                        val_n_aqg_correct += 0
                        val_n_q_correct += 0
                    else:
                        val_n_aqg_correct += aqg_matching
                        val_n_q_correct += matching

        if val_total == 0:
            val_aqg_acc, val_q_acc = 0., 0.
        else:
            val_aqg_acc = val_n_aqg_correct * 100. / val_total
            val_q_acc = val_n_q_correct * 100. / val_total

        print(val_log_template.format(time.time() - start_time, epoch, avg_loss,
                                      train_aqg_step_acc, train_aqg_acc,
                                      train_v_step_acc, train_e_step_acc, train_q_acc,
                                      val_aqg_acc, val_q_acc))

        # update checkpoint.
        if args.save_cpt:
            if args.save_all_cpt:
                if val_q_acc >= best_val_q_acc:
                    best_val_q_acc = val_q_acc
                snapshot_path = best_snapshot_prefix + \
                                '_epoch_{}_val_aqg_acc_{}_val_acc_{}_model.pt'.format(epoch, val_aqg_acc, val_q_acc)
                # save model, delete previous 'best_snapshot' files.
                torch.save(model.state_dict(), snapshot_path)
            else:
                if val_q_acc >= best_val_q_acc:
                    best_val_q_acc = val_q_acc
                    snapshot_path = best_snapshot_prefix + \
                                    '_epoch_{}_val_aqg_acc_{}_val_acc_{}_model.pt'.format(epoch, val_aqg_acc, val_q_acc)
                    # save model, delete previous 'best_snapshot' files.
                    torch.save(model.state_dict(), snapshot_path)
                    for f in glob.glob(best_snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)

    print('\nTraining finished.')
    if args.save_cpt:
        print("\nBest AQG Acc: {:.4f}\nModel writing to \"{}\"\n".format(best_val_q_acc, out_dir))