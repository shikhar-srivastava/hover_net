#!/bin/bash
#
#SBATCH --job-name=init_job
#SBATCH --time=23:00:00
#
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --gpus=1
#SBATCH --mem-per-cpu 4G
#SBATCH --output=/l/users/shikhar.srivastava/workspace/hover_net/logs/slurm/%j.out

/home/shikhar.srivastava/miniconda3/envs/hovernet_11/bin/python /l/users/shikhar.srivastava/workspace/hover_net/run_train.py --gpu='0'
