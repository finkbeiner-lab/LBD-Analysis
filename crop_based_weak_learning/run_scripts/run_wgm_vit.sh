#!/bin/bash

#SBATCH --job-name=training      ## Name of the job
#SBATCH --output=train.out    ## Output file
#SBATCH --time=07:59:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8     ## The number of threads the code will use
#SBATCH --mem=100G     ## Real memory(MB) per CPU required by the job.
#SBATCH --gres=gpu:1
#SBATCH --partition=kif-extended

## Load the python interpreters

source /gladstone/finkbeiner/home/mahirwar/miniforge3/etc/profile.d/conda.sh
#module load cuda/12.4
conda activate gigapath2
module load cuda/12.4

cd /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/crop_based

python3 wgm_vit_training.py
