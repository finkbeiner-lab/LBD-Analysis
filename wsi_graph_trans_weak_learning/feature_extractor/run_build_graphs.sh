#!/bin/bash

#SBATCH --job-name=graph      ## Name of the job
#SBATCH --output=build_graph.out    ## Output file
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

cd /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/feature_extractor

#python3 build_graphs.py --weights "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/feature_extractor/runs/May22_22-38-49_kif-gh200-02.gladstone.internal/checkpoints/model.pth" --dataset "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/feature_extractor/all_patches.csv" --output "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/simclr_features"


#python3 build_graphs.py --backbone "resnet50" --weights "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/feature_extractor/runs/May30_10-28-40_kif-gh200-04.gladstone.internal//checkpoints/model.pth" --dataset "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/tiled_slide/*/20.0" --output "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/simclr_features2"


python3 build_graphs.py --backbone "resnet50" --weights "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/feature_extractor/runs/May23_19-38-08_kif-gh200-02.gladstone.internal/checkpoints/model.pth" --dataset "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/tiled_slide/*/20.0" --output "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/simclr_features1"
