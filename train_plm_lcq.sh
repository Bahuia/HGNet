#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u train_plm.py \
--dataset lcq \
--seed 2021 \
--train_path ./data/LC-QuAD/annotated_train.pkl \
--valid_path ./data/LC-QuAD/annotated_dev.pkl \
--plm_mode bert-base-uncased \
--dropout 0.1 \
--context_mode attention \
--not_use_segment_embedding \
--not_use_eg \
--readout identity \
--att_type affine \
--d_h 256 \
--d_emb 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--n_valid_epochs 50 \
--n_epochs 65 \
--bs 4 \
--ag 4 \
--lr 2e-4 \
--lr_plm 2e-5 \
--max_n_step 20 \
--beam_size 7