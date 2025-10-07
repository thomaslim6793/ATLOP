#!/bin/bash
#SBATCH --job-name=atlop_vaccine_pathogen_finetune_docred_with_entity_masking
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=atlop_vaccine_pathogen_finetune_docred_with_entity_masking_%j.out
#SBATCH --error=atlop_vaccine_pathogen_finetune_docred_with_entity_masking_%j.err

# Load modules
module load cuda/12.3.0
module load anaconda3/2024.06

# Set up environment
cd /home/hy.lim/ATLOP
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# Activate conda
eval "$(conda shell.bash hook)"
conda activate atlop

python train.py --data_dir ./dataset/vaccine_pathogen_docred \
--transformer_type bert \
--base_model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 16 \
--test_batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 20.0 \
--seed 66 \
--device cuda \
--load_checkpoint ./best_docred_model_with_entity_masking.pth \
--fine_tune \
--save_checkpoint ./best_vaccine_model_with_entity_masking.pth \
--save_cache \
--use_cache \
--entity_masking
