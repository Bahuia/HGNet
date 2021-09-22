#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../train_seq2seq.py \
--model_name_or_path facebook/bart-base \
--do_train \
--do_eval \
--do_predict \
--train_file ../../data/LC-QuAD/seq2seq_train.json \
--validation_file ../../data/LC-QuAD/seq2seq_dev.json \
--test_file ../../data/LC-QuAD/seq2seq_test.json \
--output_dir ./runs/bart_cpt/cpt0/ \
--num_train_epochs 100 \
--text_column text \
--summary_column summary \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--overwrite_output_dir \
--predict_with_generate

