#!/bin/bash
#SBATCH --job-name=thesis-strats
#SBATCH --output=strats_%j.out
#SBATCH --error=strats_%j.err
#SBATCH --partition=shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00

set -e

cd /truenas/home/kenzikob/STraTS

apptainer exec --nv \
  --bind /truenas/home/kenzikob/STraTS:/workspace \
  /truenas/sif_images/pytorch_cuda12.6_ngc_conda_vscode.sif \
  bash -lc '
    cd /workspace
    source /opt/conda/etc/profile.d/conda.sh
    conda activate strats310
    ./run_full_main.sh
  '
