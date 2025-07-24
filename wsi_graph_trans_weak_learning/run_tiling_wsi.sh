#!/bin/bash

#SBATCH --job-name=tiling      ## Name of the job
#SBATCH --output=tiling_wsi.out    ## Output file
#SBATCH --time=07:59:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8     ## The number of threads the code will use
#SBATCH --mem=100G     ## Real memory(MB) per CPU required by the job.
#SBATCH --gres=gpu:1


## Load the python interpreters

source /gladstone/finkbeiner/home/mahirwar/miniforge3/etc/profile.d/conda.sh
#module load cuda/12.4
conda activate gigapath2
module load cuda/12.4

cd /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022


python src/tile_WSI.py -s 512 -e 0 -j 32 -B 50 -M 20 -o /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/tiled_slide  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/test_slide/*.svs"

