#!/bin/bash
#SBATCH --job-name=mpalavalexperiment2
#SBATCH --output=/ocean/projects/cis230007p/palavall/DataContaminationResearch/exp2.out
#SBATCH --err=/ocean/projects/cis230007p/palavall/DataContaminationResearch/exp2.err 
#SBATCH --time=8:00:00
#SBATCH --mem=16Gb
#SBATCH --gpus=v100-16:1
#SBATCH --partition=GPU-shared

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpalaval-research
cd /ocean/projects/cis230007p/palavall/DataContaminationResearch

python3 experiment2.py
