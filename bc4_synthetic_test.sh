#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_veryshort
#SBATCH --job-name=synthetic_testing
#SBATCH --account=coms031144
#SBATCH --output=synthetic_testing.out

. ~/initConda.sh

conda activate restormer

python synthetic_testing.py 
--yaml_file=training_options.yml 
--outputDir='/output' 
--outputImages=True 
--testingImagesDir="path/to/testingImages/dir" # change to relevant