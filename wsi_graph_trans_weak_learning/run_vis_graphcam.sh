#!/bin/bash

#SBATCH --job-name=vis      ## Name of the job
#SBATCH --output=graphcvis.out    ## Output file
#SBATCH --time=07:59:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8     ## The number of threads the code will use
#SBATCH --mem=200G     ## Real memory(MB) per CPU required by the job.
#SBATCH --gres=gpu:1
#SBATCH --partition=kif-multi-node

## Load the python interpreters

source /gladstone/finkbeiner/home/mahirwar/miniforge3/etc/profile.d/conda.sh
#module load cuda/12.4
conda activate gigapath2
module load cuda/12.4
cd /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022
python3 src/vis_graphcam.py --path_file /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/all_train.txt --path_patches /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/simclr_features1/simclr_files --path_WSI /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/recovered --path_graph /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/simclr_features1/simclr_files
