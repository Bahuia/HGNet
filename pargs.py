# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/31
# @Author  : Yongrui Chen
# @File    : pargs.py
# @Software: PyCharm
"""

import os
import torch
import random
import numpy as np
import argparse


def aqgnet_pargs():
    parser = argparse.ArgumentParser(description='AQGNet Hyper-parameters')
    parser.add_argument('--dataset', type=str, default='lcq', choices=['lcq', 'cwq', 'wsp'])
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_plm', type=float, default=2e-5)
    parser.add_argument("--ag", type=int, default=1, help="accumulate gradients for training")
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--clip_grad', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--training_proportion', type=float, default=1.0, help='propotion of training set')

    parser.add_argument("--d_emb", default=300, type=int)
    parser.add_argument("--d_h", default=256, type=int)
    parser.add_argument("--d_f", default=32, type=int)
    parser.add_argument("--n_lstm_layers", default=1, type=int)
    parser.add_argument("--n_gnn_blocks", default=3, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument('--not_birnn', action='store_false', dest='birnn')
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--beta', default=0.2, type=float)

    parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    parser.add_argument('--att_type', default='affine', choices=['dot_prod', 'affine'])
    parser.add_argument('--use_gold_aqg', action='store_true', help='use the gold aqg to predict vertex and edge')
    parser.add_argument('--plm_mode', type=str, default='none', choices=['bert-base-uncased', 'roberta-base-uncased', 'none'])
    parser.add_argument('--context_mode', type=str, default='attention', choices=['attention', 'pooling'])
    parser.add_argument('--not_graph_encoder', action='store_false', help='do not use graph encoder when generate aqg',
                        dest='use_graph_encoder')
    parser.add_argument('--not_matching_score', action='store_false',
                        help='do not use matching score for select the final query',
                        dest='use_matching_score')
    parser.add_argument('--not_matching_feature', action='store_false', help='do not use matching feature to enhance matching information',
                        dest='use_matching_feature')
    parser.add_argument('--not_mention_feature', action='store_false', help='do not use mention feature to enhance question information',
                        dest='use_mention_feature')
    parser.add_argument('--not_kb_constraint', action='store_false', help='do not use KB constraint for predict edges',
                        dest='use_kb_constraint')
    parser.add_argument('--not_subgraph', action='store_false', help='do not use KB constraint for predict edges',
                        dest='use_subgraph')
    parser.add_argument('--not_copy_v', action='store_false', help='do not use copy mechanism for vertex',
                        dest='use_v_copy')
    parser.add_argument('--not_copy_e', action='store_false', help='do not use copy mechanism for edge',
                        dest='use_e_copy')
    parser.add_argument('--not_segment_embedding', action='store_false', help='do not use query segment embedding for graph',
                        dest='use_segment_embedding')
    parser.add_argument('--not_id_embedding', action='store_false', help='do not use embedding of index of vertices and edges',
                        dest='use_id_embedding')
    parser.add_argument('--not_graph_auxiliary_encoding', action='store_false',
                        help='do not use aqg encoding vector for predict the instance',
                        dest='use_graph_auxiliary_vector')
    parser.add_argument('--not_vertex_auxiliary_encoding', action='store_false',
                        help='do not use final vertex encoding vector of the aqg for predict the vertex instance',
                        dest='use_vertex_auxiliary_encoding')
    parser.add_argument('--not_edge_auxiliary_encoding', action='store_false',
                        help='do not use final vertex encoding vector of the aqg for predict the edge instance',
                        dest='use_edge_auxiliary_encoding')
    parser.add_argument('--not_instance_auxiliary_encoding', action='store_false',
                        help='do not use final vertex encoding vector of the aqg for predict the edge instance',
                        dest='use_instance_auxiliary_encoding')
    parser.add_argument('--not_mask_aqg_prob', action='store_false',
                        help='do not mask action probability of av and ae',
                        dest='mask_aqg_prob')
    parser.add_argument("--v_num_start_switch_segment", default=2, type=int, help="the number of vertices at least to start switch segment")


    parser.add_argument("--max_num_op", default=20, type=int, help='maximum number of time steps used in decoding')
    parser.add_argument("--start_valid_epoch", default=30, type=int)

    parser.add_argument('--not_save_result', action='store_false', dest='save_result')
    parser.add_argument('--not_save_cpt', action='store_false', dest='save_cpt')
    parser.add_argument('--save_all_cpt', action='store_true', help='save all the checkpoints', dest='save_all_cpt')
    parser.add_argument('--toy_size', action='store_true', help='use small data', dest='use_small')
    parser.add_argument('--not_shuffle', action='store_false', help='do not shuffle training data', dest='shuffle')

    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--word_normalize', action='store_true')

    parser.add_argument('--train_data', type=str, default="")
    parser.add_argument('--valid_data', type=str, default="")
    parser.add_argument('--test_data', type=str, default="")

    parser.add_argument('--sparql_cache_path', type=str, default="")
    parser.add_argument('--wo_vocab', type=str, default="")
    parser.add_argument('--not_glove', action='store_false', help='do not use GloVe', dest='use_glove')
    parser.add_argument('--glove_path', type=str, default="")
    parser.add_argument('--emb_cache', type=str, default="")
    parser.add_argument('--cpt', type=str, default="")
    parser.add_argument('--result_name', type=str, default="results.pkl")

    parser.add_argument('--kb_endpoint', type=str, default="")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    return args

def relation_ranking_pargs():
    parser = argparse.ArgumentParser(description='Relation ranking for preprocess')
    parser.add_argument('--dataset', type=str, default='lcq', choices=['lcq', 'cwq', 'wsp'])

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--clip_grad', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--ns', type=int, default=30)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--rel_topk', type=int, default=500)

    parser.add_argument("--d_emb_wo", default=300, type=int)
    parser.add_argument("--d_h_wo", default=256, type=int)
    parser.add_argument("--n_lstm_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument('--not_birnn', action='store_false', dest='birnn')

    parser.add_argument('--toy_size', action='store_true', help='use small data', dest='use_small')
    parser.add_argument('--not_shuffle', action='store_false',
                        help='do not shuffle training data', dest='shuffle')

    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--word_normalize', action='store_true')

    parser.add_argument('--train_data', type=str, default="")
    parser.add_argument('--valid_data', type=str, default="")
    parser.add_argument('--test_data', type=str, default="")
    parser.add_argument('--output', type=str, default="")

    parser.add_argument('--st_pos', type=int, default=0)
    parser.add_argument('--ed_pos', type=int, default=0)
    parser.add_argument('--wo_vocab', type=str, default="")
    parser.add_argument('--rel_pool', type=str, default="")
    parser.add_argument('--not_glove', action='store_false', help='do not use GloVe', dest='glove')
    parser.add_argument('--glove_path', type=str, default=os.path.abspath(''))
    parser.add_argument('--emb_cache', type=str,default="")
    parser.add_argument('--cpt', type=str, default="")

    parser.add_argument('--kb_endpoint', type=str, default="")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    return args