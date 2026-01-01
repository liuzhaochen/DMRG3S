#!/bin/bash
#SBATCH --job-name=dmrg_square
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=6:00:00
#SBATCH --output=output2.log
#SBATCH --open-mode=append
#SBATCH --partition=small_cpu
echo "===== Run at $(date) ====="
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
julia  square.jl
