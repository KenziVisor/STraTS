#!/bin/bash

set -euo pipefail

cd src/

TRAIN_FRAC=0.5
RUN_ID=1o10

# TCN physionet
python main.py \
  --dataset physionet_2012 \
  --model_type tcn \
  --num_layers 6 \
  --hid_dim 64 \
  --kernel_size 4 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# GRUD physionet
python main.py \
  --dataset physionet_2012 \
  --model_type grud \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# GRU physionet
python main.py \
  --dataset physionet_2012 \
  --model_type gru \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# SAND physionet
python main.py \
  --dataset physionet_2012 \
  --model_type sand \
  --num_layers 4 \
  --r 24 \
  --M 12 \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# Interpnet physionet
python main.py \
  --dataset physionet_2012 \
  --model_type interpnet \
  --hid_dim 64 \
  --ref_points 192 \
  --hours_look_ahead 48 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID}

# Export TCN from its trained checkpoint
python main.py \
  --dataset physionet_2012 \
  --model_type tcn \
  --num_layers 6 \
  --hid_dim 64 \
  --kernel_size 4 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/tcn|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_tcn.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export GRUD from its trained checkpoint
python main.py \
  --dataset physionet_2012 \
  --model_type grud \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/grud|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_grud.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export GRU from its trained checkpoint
python main.py \
  --dataset physionet_2012 \
  --model_type gru \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/gru|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_gru.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export SAND from its trained checkpoint
python main.py \
  --dataset physionet_2012 \
  --model_type sand \
  --num_layers 4 \
  --r 24 \
  --M 12 \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/sand|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_sand.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export Interpnet from its trained checkpoint
python main.py \
  --dataset physionet_2012 \
  --model_type interpnet \
  --hid_dim 64 \
  --ref_points 192 \
  --hours_look_ahead 48 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/interpnet|train_frac:${TRAIN_FRAC}|run:${RUN_ID}" \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_interpnet.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0