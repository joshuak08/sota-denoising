#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=5-00:00:00
#SBATCH --mem=15GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name=restormer_volcano
#SBATCH --account=coms031144
#SBATCH --output=train_restormer.out

. ~/initConda.sh

conda activate restormer

python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt training_options.yml --launcher pytorch
