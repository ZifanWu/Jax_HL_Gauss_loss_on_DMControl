#!/bin/bash

#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2    # <- match to OMP_NUM_THREADS
#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=esrl
#SBATCH --time=12:00:00      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --output=./output/%x_%A_%a.out
#SBATCH --error=./output/%x_%A_%a.err
### GPU options ###
##SBATCH --gpus-per-node=1
#SBATCH --gres=gpu

#SBATCH --array=0-2


# module load deeplearning/2023.3;
source activate /uufs/chpc.utah.edu/common/home/u1520755/miniconda3/envs/jaxrl
# SCRDIR=/scratch/general/<file-system>/$USER/$SLURM_JOB_ID
# mkdir -p $SCRDIR
# cd $SCRDIR

# copy data to scratch (if needed)
# cp <input-files> $SCRDIR

# python train.py --save_dir= /scratch/general/nfs1/$USER/  --env_name hopper-hop --updates_per_step 32 --track \
#                                       --index=${SLURM_ARRAY_TASK_ID}

MUJOCO_GL=egl python train_pixels.py --save_dir= /scratch/general/nfs1/$USER/ --env_name hopper-hop --track \
                                     --updates_per_step 1 --index=${SLURM_ARRAY_TASK_ID}

# copy data from scratch (if needed)
# srun -M notchpeak --account=dbrown-gpu-np --partition=dbrown-gpu-np --nodes=1 --ntasks=1 --cpus-per-task 4 --mem=64g --gres=gpu -t 200:00:00 --pty bash