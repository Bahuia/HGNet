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

def hgnet_pargs():
    parser = argparse.ArgumentParser(description='AQGNet Hyper-parameters')
    parser.add_argument('--dataset', type=str, default='lcq', choices=['lcq', 'cwq', 'wsp'])
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--n_epochs', type=int, default=30, help="Maximum number of training epochs")
    parser.add_argument("--n_valid_epochs", type=int, default=30, help="Number of epochs to start validation")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--lr_plm', type=float, default=2e-5, help="Learning rate for the pre-trained model")
    parser.add_argument("--ag", type=int, default=1, help="Steps to accumulate gradients to reduce GPU memory usage")
    parser.add_argument('--bs', type=int, default=16, help="Batch size")
    parser.add_argument('--clip_grad', type=float, default=0.6, help='Gradient clipping')
    parser.add_argument('--training_proportion', type=float, default=1.0, help='The proportion of the training set used in the experiments')
    parser.add_argument('--toy_size', action='store_true', help="Use a small amount of data (top-10) for debug")
    parser.add_argument('--not_shuffle_data', action='store_false', help='Do not shuffle the training data', dest='shuffle')

    parser.add_argument("--d_emb", default=300, type=int)
    parser.add_argument("--d_h", default=256, type=int)
    parser.add_argument("--n_lstm_layers", default=1, type=int)
    parser.add_argument("--n_gnn_blocks", default=3, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument('--not_birnn', action='store_false', dest='birnn')
    parser.add_argument('--beam_size', default=5, type=int)

    parser.add_argument("--max_n_step", default=20, type=int, help='Maximum number of decoding steps')
    parser.add_argument("--n_v_switch_segment", default=2, type=int, help="The minimum number of vertices that can start switch segment (subquery)")
    parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    parser.add_argument('--att_type', default='affine', choices=['dot_prod', 'affine'])
    parser.add_argument('--plm_mode', type=str, default='none', choices=['bert-base-uncased', 'roberta-base-uncased', 'none'])
    parser.add_argument('--context_mode', type=str, default='attention', choices=['attention', 'pooling'])
    parser.add_argument('--not_use_graph_encoder', action='store_false', help='When generating AQG, do not use the graphics encoder to encode the previous graphs', dest='use_graph_encoder')
    parser.add_argument('--not_use_mention_feature', action='store_false', help='Do not use the features of entity mentions to strengthen the question encoding', dest='use_mention_feature')
    parser.add_argument('--not_use_eg', action='store_false', help='Do not use the execution-guided strategy to predict edges', dest='use_eg')
    parser.add_argument('--not_use_subgraph_cache', action='store_false', help='No acceleration with subgraphs during EG execution', dest='use_subgraph')
    parser.add_argument('--not_copy_v', action='store_false', help='Do not use the copy mechanism for vertices', dest='use_v_copy')
    parser.add_argument('--not_copy_e', action='store_false', help='Do not use the copy mechanism for edges', dest='use_e_copy')
    parser.add_argument('--not_use_segment_embedding', action='store_false', help='Do not use the embeddings of query segment when encoding the graphs by the graph encoder', dest='use_segment_embedding')
    parser.add_argument('--not_use_id_embedding', action='store_false', help='Do not use the index embedding of vertices and edges when encoding the graph in the graph encoder', dest='use_id_embedding')
    parser.add_argument('--not_use_graph_auxiliary_encoding', action='store_false', help='Do not use AQG encoded vectors to predict instances', dest='use_graph_auxiliary_vector')
    parser.add_argument('--not_use_vertex_auxiliary_encoding', action='store_false', help='Do not use the final vertex encoding vector of AQG to predict vertex instances.', dest='use_vertex_auxiliary_encoding')
    parser.add_argument('--not_use_edge_auxiliary_encoding', action='store_false', help='Do not use the final edge encoding vector of AQG to predict edge instances', dest='use_edge_auxiliary_encoding')
    parser.add_argument('--not_use_instance_auxiliary_encoding', action='store_false', help='Do not use the final instance encoding vector of AQG', dest='use_instance_auxiliary_encoding')
    parser.add_argument('--not_mask_aqg_prob', action='store_false', help='No masking of AddVertex and AddEdge action probabilities', dest='mask_aqg_prob')

    parser.add_argument('--no_cuda', action='store_false', help='Do not use GPU', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')

    parser.add_argument('--train_path', type=str, default="")
    parser.add_argument('--valid_path', type=str, default="")
    parser.add_argument('--test_path', type=str, default="")
    parser.add_argument('--vocab_path', type=str, default="")
    parser.add_argument('--not_use_glove', action='store_false', help='Do not use GloVe word embeddings', dest='use_glove')
    parser.add_argument('--glove_path', type=str, default="", help='Path of GloVe word embeddings')
    parser.add_argument('--embed_cache_path', type=str, default="")
    parser.add_argument('--cpt_path', type=str, default="")
    parser.add_argument('--not_save_cpt', action='store_false', help='Do not save any checkpoints', dest='save_cpt')
    parser.add_argument('--save_all_cpt', action='store_true', help='Save all checkpoints', dest='save_all_cpt')
    parser.add_argument('--not_save_result', action='store_false', help='Do not save results', dest='save_result')
    parser.add_argument('--result_path', type=str, default="./results.pkl")

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