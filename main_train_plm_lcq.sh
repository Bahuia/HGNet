#!/bin/bash

devices=1

CUDA_LAUNCH_BLOCKING=1 python -u main_train_plm.py \
--dataset lcq \
--seed 2021 \
--train_data ./data/LC-QuAD/annotated_train.pkl \
--valid_data ./data/LC-QuAD/annotated_dev.pkl \
--plm_mode bert-large-uncased \
--dropout 0.1 \
--context_mode attention \
--not_segment_embedding \
--not_matching_feature \
--not_matching_score \
--not_kb_constraint \
--readout identity \
--att_type affine \
--d_h 256 \
--d_emb 300 \
--d_f 32 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--start_valid_epoch 0 \
--n_epochs 100 \
--bs 1 \
--ag 16 \
--lr 2e-4 \
--lr_plm 2e-5 \
--max_num_op 20 \
--beam_size 7