#!/bin/bash

set -euo pipefail

cd src/

TRAIN_FRAC=0.5
RUN_ID=1o10

# STRATS pretrain (mimic_iii)
python main.py \
  --pretrain 1 \
  --dataset mimic_iii \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-4 \
  --max_epochs 30

# STRATS finetune from pretrained checkpoint (mimic_iii)
python main.py \
  --dataset mimic_iii \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-5 \
  --load_ckpt_path ../outputs/mimic_iii/pretrain/checkpoint_best.bin \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# TCN mimic_iii
python main.py \
  --dataset mimic_iii \
  --model_type tcn \
  --num_layers 4 \
  --hid_dim 128 \
  --kernel_size 4 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# GRUD mimic_iii
python main.py \
  --dataset mimic_iii \
  --model_type grud \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# GRU mimic_iii
python main.py \
  --dataset mimic_iii \
  --model_type gru \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# SAND mimic_iii
python main.py \
  --dataset mimic_iii \
  --model_type sand \
  --num_layers 4 \
  --r 24 \
  --M 12 \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# Interpnet mimic_iii
python main.py \
  --dataset mimic_iii \
  --model_type interpnet \
  --hid_dim 64 \
  --ref_points 96 \
  --hours_look_ahead 24 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# Export TCN from its trained checkpoint
python main.py \
  --dataset mimic_iii \
  --model_type tcn \
  --num_layers 4 \
  --hid_dim 128 \
  --kernel_size 4 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/tcn|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_tcn_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export GRUD from its trained checkpoint
python main.py \
  --dataset mimic_iii \
  --model_type grud \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/grud|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_grud_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export GRU from its trained checkpoint
python main.py \
  --dataset mimic_iii \
  --model_type gru \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/gru|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_gru_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export SAND from its trained checkpoint
python main.py \
  --dataset mimic_iii \
  --model_type sand \
  --num_layers 4 \
  --r 24 \
  --M 12 \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/sand|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_sand_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export Interpnet from its trained checkpoint
python main.py \
  --dataset mimic_iii \
  --model_type interpnet \
  --hid_dim 64 \
  --ref_points 96 \
  --hours_look_ahead 24 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/interpnet|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_interpnet_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export STRATS predictions from its trained checkpoint
python main.py \
  --dataset mimic_iii \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-5 \
  --load_ckpt_path ../outputs/mimic_iii/pretrain/checkpoint_best.bin \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/strats,num_layers:2,hid_dim:64,num_heads:16,dropout:0.2,attention_dropout:0.2,lr:5e-5|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_strats_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0
