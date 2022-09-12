#!/bin/bash

devices=2

CUDA_LAUNCH_BLOCKING=1 python -u main_eval.py \
--dataset cwq \
--seed 2021 \
--test_data ./data/ComplexWebQuestions/annotated_test.pkl \
--wo_vocab ./vocab/word_vocab_cwq.pkl \
--not_kb_constraint \
--not_save_result \
--context_mode attention \
--not_matching_score \
--not_mention_feature \
--not_matching_feature \
--d_h 256 \
--d_emb 300 \
--d_f 32 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 5 \
--max_num_op 45 \
--cpt ./runs/cwq/1653202047/checkpoints/best_snapshot_epoch_39_val_aqg_acc_76.268271711092_val_acc_49.46976210948696_model.pt \
--result_name result.pkl \
--kb_endpoint http://10.201.102.90:8890//sparql
