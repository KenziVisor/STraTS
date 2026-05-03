# Runnable Scripts

This file lists repository entry-point scripts only. Internal model, dataset, evaluator, and helper modules are intentionally omitted.

## `src/main.py`

- Purpose: Main training, validation, checkpointing, and prediction-export CLI for STraTS, iSTraTS, and baseline supervised models. Also runs self-supervised pretraining with `--pretrain 1`.
- Inputs: Processed pickle at `../data/processed/{dataset}.pkl`; supervised runs need `--latent_csv_path`; optional `--load_ckpt_path`, model hyperparameters, split controls, and export path.
- Outputs: `output_dir` with `checkpoint_best.bin`, `log.txt`, learning-curve artifacts, optional `training_summary.txt`, and optional prediction CSV from `--save_pred_csv_path`.
- Example: `(cd src && python main.py --dataset physionet_2012 --model_type strats --latent_csv_path ../data/physionet_latent_tags.csv --train_frac 0.5 --run 1o10)`

## `src/preprocess_physionet_2012.py`

- Purpose: Build the split-aware PhysioNet 2012 processed pickle consumed by `src/main.py`.
- Inputs: Raw PhysioNet folders and outcomes under the hardcoded `../../physionet2012` path, relative to `src/`.
- Outputs: `../data/processed/physionet_2012.pkl`.
- Example: `(cd src && python preprocess_physionet_2012.py)`

## `src/preprocess_mimic_iii_large.py`

- Purpose: Build the split-aware MIMIC-III processed pickle consumed by `src/main.py`.
- Inputs: Raw MIMIC-III CSV files under the hardcoded `../mimiciii` path, relative to `src/`.
- Outputs: `../data/processed/mimic_iii.pkl`.
- Example: `(cd src && python preprocess_mimic_iii_large.py)`

## `run_main.sh`

- Purpose: Compact PhysioNet workflow: pretrain STraTS, fine-tune on latent tags, then export predictions.
- Inputs: `data/processed/physionet_2012.pkl`, `data/latent_tags.csv`, and the pretraining checkpoint created by the first step.
- Outputs: PhysioNet output directories under `outputs/physionet_2012/` and `outputs/predicted_latent_tags.csv`.
- Example: `bash run_main.sh`

## `run_main_rest.sh`

- Purpose: PhysioNet baseline workflow for TCN, GRUD, GRU, SAND, and InterpNet training plus checkpoint-based prediction export.
- Inputs: `data/processed/physionet_2012.pkl`, `data/latent_tags.csv`, and existing or newly created baseline checkpoints for `TRAIN_FRAC=0.5`, `RUN_ID=1o10`.
- Outputs: Baseline output directories under `outputs/physionet_2012/` and per-model prediction CSVs under `outputs/`.
- Example: `bash run_main_rest.sh`

## `run_main_mimic.sh`

- Purpose: MIMIC workflow for STraTS pretraining/fine-tuning, baseline training, and checkpoint-based prediction export.
- Inputs: `data/processed/mimic_iii.pkl`, `data/latent_tags.csv`, and the checkpoints created by the training steps.
- Outputs: MIMIC output directories under `outputs/mimic_iii/` and per-model prediction CSVs under `outputs/`.
- Example: `bash run_main_mimic.sh`

## `run_full_main.sh`

- Purpose: Combined PhysioNet and MIMIC workflow for STraTS and baseline training plus prediction export.
- Inputs: `data/processed/physionet_2012.pkl`, `data/processed/mimic_iii.pkl`, `data/physionet_latent_tags.csv`, `data/mimic_latent_tags.csv`, and checkpoints created earlier in the script.
- Outputs: Dataset-specific output directories under `outputs/physionet_2012/` and `outputs/mimic_iii/`, plus exported prediction CSVs under `outputs/`.
- Example: `bash run_full_main.sh`
