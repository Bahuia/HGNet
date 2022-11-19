#!/bin/bash

devices=3
cpt_path=$1

CUDA_LAUNCH_BLOCKING=1 python -u eval.py \
--dataset cwq \
--seed 2021 \
--test_path ./data/ComplexWebQuestions/annotated_test.pkl \
--vocab_path ./vocab/word_vocab_cwq.pkl \
--context_mode attention \
--not_use_mention_feature \
--d_h 256 \
--d_emb 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 5 \
--max_n_step 45 \
--cpt_path $cpt_path \
--result_path result.pkl \
--kb_endpoint http://10.201.89.70:8890//sparql \
--not_use_eg