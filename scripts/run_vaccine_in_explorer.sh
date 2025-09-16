#!/bin/bash
module load cuda/12.3.0

python train.py --data_dir ./dataset/vaccine_pathogen_docred \
--transformer_type bert \
--model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 8 \
--test_batch_size 16 \
--gradient_accumulation_steps 1 \
--num_labels 2 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 10.0 \
--seed 66 \
--num_class 2 \
--device cuda
