# STraTS Project Details

## One-screen summary

- This is a compact PyTorch research repo for irregular ICU time-series.
- There are two actual run modes:
  1. `--pretrain 1`: self-supervised forecasting of near-future measurements.
  2. default supervised mode: multi-label latent-tag prediction from `--latent_csv_path`.
- `src/main.py` is the only real entry point.
- `run_main.sh` shows the intended sequence: pretrain STraTS -> fine-tune on latent tags -> export predictions.
- In this fork, the end-to-end path is centered on `strats` / `istrats`. Other backends still contain original single-output assumptions; see `Known issues / limitations`.
- `README.md` is empty. Use this file as the main repo map.

## Files that matter

- `src/main.py`: CLI, output-dir naming, model selection, training loop, validation, checkpoint save/load, prediction export.
- `src/dataset_pretrain.py`: self-supervised pretraining batches and saved variable stats.
- `src/dataset.py`: supervised latent-tag loading, split subsampling, normalization, and model-specific tensorization.
- `src/models.py`: shared head logic. Pretrain, fine-tune, and scratch-supervised use different heads.
- `src/modeling_strats.py`: STraTS / iSTraTS implementation.
- `src/modeling_*.py`: baseline backends.
- `src/evaluator.py`: per-target AUROC/AUPRC/minRP averaging for supervised runs.
- `src/evaluator_pretrain.py`: negative forecasting loss for pretraining.
- `src/preprocess_physionet_2012.py`, `src/preprocess_mimic_iii_large.py`: build the processed pickle files expected by the loaders.

## Artifact contract

### Processed pickle

Expected at `../data/processed/{dataset}.pkl` and loaded as:

`events, oc, train_ids, val_ids, test_ids = pickle.load(...)`

Current training code expects `events` to contain at least:

- `ts_id`
- `minute`
- `variable`
- `value`

Notes:

- `oc` is mostly leftover lineage from the original repo. Current supervised training does not use it for targets.
- MIMIC preprocessing also stores a `TABLE` column, but training ignores it.
- `physionet_2012` preprocessing converts `ICUType` into `ICUType_1` to `ICUType_4` before saving.

### Latent CSV

Required for supervised mode via `--latent_csv_path`.

Expected shape:

- one `ts_id` column
- every other column is treated as a binary label

Loader behavior:

- casts CSV `ts_id` to string
- keeps only rows matching the supervised split ids
- raises on missing labels or duplicate `ts_id`
- sets `args.num_targets`, `args.target_columns`, and `args.pos_class_weight`

### Pretrain sidecar

Pretraining writes `pt_saved_variables.pkl` into the output dir.

Contents:

- `variables`
- `means_stds`
- `max_minute`

Supervised fine-tuning with `--load_ckpt_path` tries to reuse this file from the checkpoint directory.

## Actual runtime flows

### 1. Pretraining

Triggered by `--pretrain 1`.

Path:

- `PretrainDataset(args)`
- STraTS-style irregular batches
- `PretrainEvaluator`

Behavior:

- test split is removed entirely
- variables are restricted to those seen in the pretraining train split
- static features are pulled out into `demo`
- values are normalized from train data only
- each sample picks a forecast anchor `t1` from observed timestamps except the last timestamp, and only if `t1 >= 12h`
- model input is history up to `t1`
- target is the last observed value per variable inside the next 2 hours `(t1, t1 + 120 min]`
- PhysioNet uses up to 48h for time normalization
- MIMIC keeps up to 5 days of raw events, but each sampled context is clipped to the last 24h window before `t1`

Pretraining evaluation nuance:

- `PretrainEvaluator` materializes and caches 3 random forecast windows per evaluation chunk the first time each split is evaluated
- metric is `loss_neg = -forecast_mse`

### 2. Supervised latent-tag training

Default mode (`--pretrain 0`).

Path:

- `Dataset(args)`
- selected backend from `model_type`
- `Evaluator`

Behavior:

- train/val splits can be downsampled with `--train_frac` and shifted across repeated runs via `--run` such as `1o10`
- only variables seen in the supervised train split are kept
- MIMIC is truncated to the first 24h and ages above 200 are clamped to 91.4
- static variables are removed from the event stream and normalized separately into `demo`
- PhysioNet static demo = `Age`, `Gender`, `Height`, `ICUType_1..4`, plus `Gender_missing` and `Height_missing`
- MIMIC static demo = `Age`, `Gender`
- evaluator computes per-target AUROC, AUPRC, and max min(precision, recall), then averages across non-degenerate label columns
- `eval_train` is only the first 2000 train examples

### 3. Prediction export

If `--save_pred_csv_path` is set, export happens after the training loop.

Output columns:

- `ts_id`
- one `<label>_prob` column per target
- one thresholded binary column per target

Important behavior:

- export uses model probabilities at threshold `0.5`
- export does not skip training
- if `checkpoint_best.bin` exists in the current output dir, export reloads it before writing the CSV

## Representation by model type

- `strats`, `istrats`
  Irregular triplets: `values`, normalized `times` in `[-1, 1]`, `varis`, `obs_mask`, `demo`, `labels`.
- `gru`, `tcn`, `sand`
  Hourly dense tensor with concatenated `values`, observation mask, and delta-since-last-observation, plus `demo`.
- `grud`
  Irregular dense tensors: `x_t`, `m_t`, `delta_t`, `seq_len`, `demo`, `labels`.
- `interpnet`
  Irregular dense tensors: `t`, `x`, `m`, holdout mask `h`, `demo`, `labels`.

Practical truth:

- only STraTS / iSTraTS are wired for pretraining
- only STraTS / iSTraTS appear fully adapted to this fork's multi-label latent-tag task

## STraTS internals

- each observation token is `time_cve + value_cve + variable_embedding`
- a custom transformer contextualizes the observation tokens
- fusion attention pools the token set into a single time-series embedding
- `strats` pools the contextualized tokens; `istrats` pools the raw triplet embeddings
- static demo features are concatenated before the final head(s)

## Shared model logic

`src/models.py` controls the head layout:

- pretrain: `forecast_head(ts_demo_emb -> V)`
- scratch supervised: `binary_head(ts_demo_emb -> K)`
- fine-tune (`load_ckpt_path` set): `forecast_head(ts_demo_emb -> V)` then `binary_head(V -> K)`

This means "fine-tune from pretraining" is architecturally designed around models that explicitly route through `forecast_head` before classification. In the current code, that is only true for `Strats.forward()`.

## Training and checkpointing

- optimizer: `AdamW`
- gradient clipping: `0.3`
- seed: `args.seed + int(args.run.split('o')[0])`
- validation runs before any training when `validate_after < 0` (default)
- best checkpoint metric:
  - pretrain: `loss_neg`
  - supervised: `auprc + auroc`
- best checkpoint path: `checkpoint_best.bin`
- log file: `log.txt`

Auto output-dir naming:

- pretrain: `../outputs/<dataset>/<prefix>pretrain/`
- supervised: `../outputs/<dataset>/<prefix><model_type>...|train_frac:<x>|run:<y>`
- when `load_ckpt_path` is set, `finetune_` is prepended to the output prefix

## Known issues / limitations (by code inspection)

1. Warm-start loading in `src/main.py` is likely broken.
   The code loads `args.load_ckpt_path`, merges overlapping keys into `curr_state_dict`, then ignores it and calls `model.load_state_dict(torch.load(model_path_best, ...))`. This likely defeats the intended checkpoint initialization.

2. Multi-label support is not consistently ported beyond STraTS / iSTraTS.
   `GRU_TS`, `TCN_TS`, `SAND`, `GRUD_TS`, and `InterpNet` all squeeze logits with `[:,0]`, while the dataset and evaluator now expect `K` outputs. These backends still look like single-target code.

3. Fine-tuning from a pretrained checkpoint looks STraTS-only.
   In finetune mode, `TimeSeriesModel` makes `binary_head` expect input size `V`, but the non-STraTS backends still feed `ts_demo_emb` directly into `binary_head`. That is a shape mismatch.

4. Pretraining is not enforced to be STraTS-only, but the code path is.
   `PretrainDataset.get_batch()` returns STraTS-style irregular inputs plus forecast targets. Non-STraTS backends do not accept that signature.

5. Prediction export is STraTS-specific.
   `export_latent_predictions()` hardcodes `values/times/varis/obs_mask/demo` and calls the model with the STraTS signature.

6. Supervised MIMIC latent-tag loading may break on numeric `ts_id`.
   `Dataset` stringifies `sup_ts_ids` for latent-label alignment, but later builds `self.splits` with raw `train_ids/val_ids/test_ids` against a string-keyed mapping. That looks unsafe for MIMIC, where `ts_id` is numeric in preprocessing.

7. `requirements.txt` is incomplete for current imports.
   At minimum, `transformers` and `pytz` are imported by runtime code but not listed.

## Fast re-entry

If a future Codex session needs context quickly:

1. Read this file.
2. Read `src/main.py`.
3. Read `src/dataset_pretrain.py` or `src/dataset.py` depending on mode.
4. Read only the model file you are editing.
5. If the task uses `load_ckpt_path`, check the warm-start bug first.
6. If the task uses a non-STraTS backend, verify shape assumptions before changing anything.
