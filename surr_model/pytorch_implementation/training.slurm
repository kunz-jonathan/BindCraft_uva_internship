#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --output=second_iteration_model_training_%A.log
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --partition=gpuRTX-4th,gpuRTX-1st,gpuRTX-2nd,gpuRTX-3rd
#SBATCH --gpus=1


CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindCraft_modified
export LD_LIBRARY_PATH=${CONDA_BASE}/lib


# train model
python training_one_shot_mmpbsa_one_set_second_iteration.py