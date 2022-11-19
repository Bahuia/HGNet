#!/bin/bash

devices=3
cpt_path=$1

CUDA_LAUNCH_BLOCKING=1 python -u eval.py \
--dataset wsp \
--seed 2021 \
--test_path ./data/WebQSP/annotated_test.pkl \
--vocab_path ./vocab/word_vocab_wsp.pkl \
--not_use_segment_embedding \
--context_mode attention \
--d_h 256 \
--d_emb 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 7 \
--max_n_step 45 \
--cpt_path $cpt_path \
--result_path result.pkl \
--kb_endpoint http://10.201.89.70:8890//sparql \
--not_use_eg
