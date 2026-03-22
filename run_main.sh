#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

run_commands(){
    eval template="$1"
    for train_frac in 0.5 0.4 0.3 0.2 0.1; do
        for ((i=1; i<=10; i++)); do
            run_param="${i}o10"
            eval "$1 --run $run_param --train_frac $train_frac"
        done
    done
}

cd src/

# 1) Pretrain once
python main.py --pretrain 1 --dataset physionet_2012 --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 100

# 2) Finetune once
python main.py \
  --dataset physionet_2012 \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-5 \
  --load_ckpt_path ../outputs/physionet_2012/pretrain/checkpoint_best.bin \
  --latent_csv_path ../data/latent_tags.csv \
  --train_frac 0.5 \
  --run 1o10

# 3) Export once
python main.py \
  --dataset physionet_2012 \
  --model_type strats \
  --hid_dim 64 \
  --num_layers 2 \
  --num_heads 16 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 5e-5 \
  --load_ckpt_path ../outputs/physionet_2012/pretrain/checkpoint_best.bin \
  --latent_csv_path ../data/latent_tags.csv \
  --save_pred_csv_path ../outputs/predicted_latent_tags.csv \
  --predict_split all
