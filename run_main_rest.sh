#!/bin/bash

set -euo pipefail


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

# TCN physionet
template="python main.py --dataset physionet_2012 --model_type tcn --num_layers 6 --hid_dim 64 --kernel_size 4 --dropout 0.2 --lr 5e-4"
run_commands "\${template}"

# GRUD physionet
template="python main.py --dataset physionet_2012 --model_type grud --hid_dim 64 --dropout 0.2 --lr 5e-4"
run_commands "\${template}"

# GRU physionet
template="python main.py --dataset physionet_2012 --model_type gru --hid_dim 64 --dropout 0.2 --lr 5e-4"
run_commands "\${template}"

# SAND physionet
template="python main.py --dataset physionet_2012 --model_type sand --num_layers 4 --r 24 --M 12 --hid_dim 64 --dropout 0.2 --lr 5e-4"
run_commands "\${template}"

# Interpnet physionet
template="python main.py --dataset physionet_2012 --model_type interpnet --hid_dim 64 --ref_points 192 --hours_look_ahead 48 --dropout 0.2 --lr 5e-4"
run_commands "\${template}"

# Export TCN from its trained checkpoint (train_frac=0.5, run=1o10)
python main.py --dataset physionet_2012 --model_type tcn --num_layers 6 --hid_dim 64 --kernel_size 4 --dropout 0.2 --lr 5e-4 --train_frac 0.5 --run 1o10 --output_dir '../outputs/physionet_2012/tcn|train_frac:0.5|run:1o10' --latent_csv_path ../data/latent_tags.csv --save_pred_csv_path ../outputs/predicted_latent_tags_tcn.csv --predict_split all --max_epochs 0 --validate_after 0

# Export GRUD from its trained checkpoint (train_frac=0.5, run=1o10)
python main.py --dataset physionet_2012 --model_type grud --hid_dim 64 --dropout 0.2 --lr 5e-4 --train_frac 0.5 --run 1o10 --output_dir '../outputs/physionet_2012/grud|train_frac:0.5|run:1o10' --latent_csv_path ../data/latent_tags.csv --save_pred_csv_path ../outputs/predicted_latent_tags_grud.csv --predict_split all --max_epochs 0 --validate_after 0

# Export GRU from its trained checkpoint (train_frac=0.5, run=1o10)
python main.py --dataset physionet_2012 --model_type gru --hid_dim 64 --dropout 0.2 --lr 5e-4 --train_frac 0.5 --run 1o10 --output_dir '../outputs/physionet_2012/gru|train_frac:0.5|run:1o10' --latent_csv_path ../data/latent_tags.csv --save_pred_csv_path ../outputs/predicted_latent_tags_gru.csv --predict_split all --max_epochs 0 --validate_after 0

# Export SAND from its trained checkpoint (train_frac=0.5, run=1o10)
python main.py --dataset physionet_2012 --model_type sand --num_layers 4 --r 24 --M 12 --hid_dim 64 --dropout 0.2 --lr 5e-4 --train_frac 0.5 --run 1o10 --output_dir '../outputs/physionet_2012/sand|train_frac:0.5|run:1o10' --latent_csv_path ../data/latent_tags.csv --save_pred_csv_path ../outputs/predicted_latent_tags_sand.csv --predict_split all --max_epochs 0 --validate_after 0

# Export Interpnet from its trained checkpoint (train_frac=0.5, run=1o10)
python main.py --dataset physionet_2012 --model_type interpnet --hid_dim 64 --ref_points 192 --hours_look_ahead 48 --dropout 0.2 --lr 5e-4 --train_frac 0.5 --run 1o10 --output_dir '../outputs/physionet_2012/interpnet|train_frac:0.5|run:1o10' --latent_csv_path ../data/latent_tags.csv --save_pred_csv_path ../outputs/predicted_latent_tags_interpnet.csv --predict_split all --max_epochs 0 --validate_after 0
