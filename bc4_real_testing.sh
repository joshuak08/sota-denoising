#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_veryshort
#SBATCH --job-name=psnr_compare_mod_testing
#SBATCH --account=coms031144
#SBATCH --output=psnr_compare_mod_testing.out

. ~/initConda.sh

conda activate restormer

python real_testing.py
--yaml_file=training_options.yml  
--outputDir="/output" 
--outputImages=True 
--testingImagesDir="path/to/testingImages/dir" # replace with directory to testing images
--savedState="net_g_latest.pth" # replace with wanted saved model state 