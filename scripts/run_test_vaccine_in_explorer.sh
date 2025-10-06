#!/bin/bash
#SBATCH --job-name=atlop_test_vaccine_finetune_docred_with_entity_masking
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=atlop_test_vaccine_finetune_docred_with_entity_masking_%j.out
#SBATCH --error=atlop_test_vaccine_finetune_docred_with_entity_masking_%j.err

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

python test.py --data_dir ./dataset/vaccine_pathogen_docred \
--transformer_type bert \
--base_model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--dev_file dev.json \
--test_file test.json \
--test_batch_size 16 \
--num_labels 2 \
--seed 66 \
--num_class 2 \
--device cuda \
--load_checkpoint ./best_vaccine_model_with_entity_masking.pth \
--use_cache \
--display_test_examples \
--entity_masking
