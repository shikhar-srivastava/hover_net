#!/bin/bash
#
#SBATCH --job-name=init_job
#SBATCH --time=23:00:00
#
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=4
#SBATCH --output=/l/users/shikhar.srivastava/workspace/hover_net/logs/slurm/%j.out

/home/shikhar.srivastava/miniconda3/envs/hovernet_11/bin/python /l/users/shikhar.srivastava/workspace/hover_net/run_train.py --gpu='0,1,2,3'
