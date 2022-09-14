#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u main_train_plm.py \
--dataset cwq \
--seed 2021 \
--train_data ./data/ComplexWebQuestions/annotated_train.pkl \
--valid_data ./data/ComplexWebQuestions/annotated_dev.pkl \
--plm_mode bert-base-uncased \
--dropout 0.1 \
--context_mode attention \
--not_matching_score \
--not_mention_feature \
--not_matching_feature \
--not_kb_constraint \
--readout identity \
--att_type affine \
--d_h 256 \
--d_emb 300 \
--d_f 32 \
--gpu $devices \
--start_valid_epoch 0 \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--n_epochs 200 \
--bs 4 \
--ag 4 \
--lr 2e-4 \
--lr_plm 2e-5 \
--max_num_op 45 \
--beam_size 7