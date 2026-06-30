#!/bin/bash

set -euo pipefail

cd src/

TRAIN_FRAC=0.5
RUN_ID=1o10

########################################
# PHYSIONET TRAINING
########################################
if [[ "${DATASET_SCOPE:-both}" == *physionet* ]]; then

# STRATS pretrain (physionet_2012)
${PYTHON:-python} main.py \
  --pretrain 1 \
  --dataset physionet_2012 \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-4 \
  --output_dir "../outputs/physionet_2012/strats_pretrain" \
  --max_epochs 100

# STRATS finetune (physionet_2012)
${PYTHON:-python} main.py \
  --dataset physionet_2012 \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-5 \
  --load_ckpt_path ../outputs/physionet_2012/strats_pretrain/checkpoint_best.bin \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/strats_finetune"

# GRU (physionet_2012)
${PYTHON:-python} main.py \
  --dataset physionet_2012 \
  --model_type gru \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/gru"

# GRUD (physionet_2012)
${PYTHON:-python} main.py \
  --dataset physionet_2012 \
  --model_type grud \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/grud"

# TCN (physionet_2012)
${PYTHON:-python} main.py \
  --dataset physionet_2012 \
  --model_type tcn \
  --num_layers 6 \
  --hid_dim 64 \
  --kernel_size 4 \
  --dropout 0.2 \
  --lr 5e-4 \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/tcn"

# SAND (physionet_2012)
${PYTHON:-python} main.py \
  --dataset physionet_2012 \
  --model_type sand \
  --num_layers 4 \
  --r 24 \
  --M 12 \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/sand"

########################################
# PHYSIONET EXPORTS
########################################

# Export STRATS predictions (physionet_2012)
${PYTHON:-python} main.py \
  --dataset physionet_2012 \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-5 \
  --load_ckpt_path ../outputs/physionet_2012/strats_finetune/checkpoint_best.bin \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/strats_finetune" \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_physionet_latent_tags_strats.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export GRU predictions (physionet_2012)
${PYTHON:-python} main.py \
  --dataset physionet_2012 \
  --model_type gru \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/gru" \
  --load_ckpt_path ../outputs/physionet_2012/gru/checkpoint_best.bin \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_physionet_latent_tags_gru.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export GRUD predictions (physionet_2012)
${PYTHON:-python} main.py \
  --dataset physionet_2012 \
  --model_type grud \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/grud" \
  --load_ckpt_path ../outputs/physionet_2012/grud/checkpoint_best.bin \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_physionet_latent_tags_grud.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export TCN predictions (physionet_2012)
${PYTHON:-python} main.py \
  --dataset physionet_2012 \
  --model_type tcn \
  --num_layers 6 \
  --hid_dim 64 \
  --kernel_size 4 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/physionet_2012/tcn" \
  --load_ckpt_path ../outputs/physionet_2012/tcn/checkpoint_best.bin \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_physionet_latent_tags_tcn.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export SAND predictions (physionet_2012)
${PYTHON:-python} main.py \
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
  --output_dir "../outputs/physionet_2012/sand" \
  --load_ckpt_path ../outputs/physionet_2012/sand/checkpoint_best.bin \
  --latent_csv_path ../data/physionet_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_physionet_latent_tags_sand.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

########################################
# MIMIC TRAINING
########################################
if [[ "${DATASET_SCOPE:-both}" == *mimic* ]]; then

# STRATS pretrain (mimic_iii)
${PYTHON:-python} main.py \
  --pretrain 1 \
  --dataset mimic_iii \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-4 \
  --output_dir "../outputs/mimic_iii/strats_pretrain" \
  --max_epochs 30

# STRATS finetune (mimic_iii)
${PYTHON:-python} main.py \
  --dataset mimic_iii \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-5 \
  --load_ckpt_path ../outputs/mimic_iii/strats_pretrain/checkpoint_best.bin \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/strats_finetune"

# GRU (mimic_iii)
${PYTHON:-python} main.py \
  --dataset mimic_iii \
  --model_type gru \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/gru"

# GRUD (mimic_iii)
${PYTHON:-python} main.py \
  --dataset mimic_iii \
  --model_type grud \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/grud"

# TCN (mimic_iii)
${PYTHON:-python} main.py \
  --dataset mimic_iii \
  --model_type tcn \
  --num_layers 4 \
  --hid_dim 128 \
  --kernel_size 4 \
  --dropout 0.2 \
  --lr 5e-4 \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/tcn"

# SAND (mimic_iii)
${PYTHON:-python} main.py \
  --dataset mimic_iii \
  --model_type sand \
  --num_layers 4 \
  --r 24 \
  --M 12 \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/sand"

########################################
# MIMIC EXPORTS
########################################

# Export STRATS predictions (mimic_iii)
${PYTHON:-python} main.py \
  --dataset mimic_iii \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-5 \
  --load_ckpt_path ../outputs/mimic_iii/strats_finetune/checkpoint_best.bin \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/strats_finetune" \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_strats_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export GRU predictions (mimic_iii)
${PYTHON:-python} main.py \
  --dataset mimic_iii \
  --model_type gru \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/gru" \
  --load_ckpt_path ../outputs/mimic_iii/gru/checkpoint_best.bin \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_gru_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export GRUD predictions (mimic_iii)
${PYTHON:-python} main.py \
  --dataset mimic_iii \
  --model_type grud \
  --hid_dim 64 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/grud" \
  --load_ckpt_path ../outputs/mimic_iii/grud/checkpoint_best.bin \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_grud_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export TCN predictions (mimic_iii)
${PYTHON:-python} main.py \
  --dataset mimic_iii \
  --model_type tcn \
  --num_layers 4 \
  --hid_dim 128 \
  --kernel_size 4 \
  --dropout 0.2 \
  --lr 5e-4 \
  --train_frac ${TRAIN_FRAC} \
  --run ${RUN_ID} \
  --output_dir "../outputs/mimic_iii/tcn" \
  --load_ckpt_path ../outputs/mimic_iii/tcn/checkpoint_best.bin \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_tcn_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

# Export SAND predictions (mimic_iii)
${PYTHON:-python} main.py \
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
  --output_dir "../outputs/mimic_iii/sand" \
  --load_ckpt_path ../outputs/mimic_iii/sand/checkpoint_best.bin \
  --latent_csv_path ../data/mimic_latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags_sand_mimic.csv \
  --predict_split all \
  --max_epochs 0 \
  --validate_after 0

fi