#!/bin/bash

#SBATCH --time=1-12:00:00
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --nodes=1
#SBATCH --constraint=A100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=VAE_GMNIST
#SBATCH --output=./logs/%x.%A_%a.out
#SBATCH --error=./logs/%x.%A_%a.err
#SBATCH --array=1-2
#SBATCH --exclusive

DELAY=$(( 5*${SLURM_ARRAY_TASK_ID}))
sleep ${DELAY}s
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SWEEP_ID='mb010/vae_GalaxyMNIST/nlvufkah'
source /share/nas2/mbowles/AstroAugmentations/venv/bin/activate
wandb agent --count 1 $SWEEP_ID
