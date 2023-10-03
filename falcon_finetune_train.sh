#!/bin/bash
#SBATCH --job-name=falconfinetune
#SBATCH --output=/ocean/projects/cis230007p/palavall/DataContaminationResearch/falconfinetunetrain.out
#SBATCH --err=/ocean/projects/cis230007p/palavall/DataContaminationResearch/falconfinetunetrain.err 
#SBATCH --time=8:00:00
#SBATCH	--mem=16Gb
#SBATCH --gpus=v100-16:1
#SBATCH --partition=GPU-shared

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpalaval-research
cd /ocean/projects/cis230007p/palavall/DataContaminationResearch

python3 falcon_finetune_train.py