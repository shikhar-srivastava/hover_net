#!/bin/sh

#SBATCH --cpus-per-task       128
#SBATCH --gres                gpu:4
#SBATCH --job-name            Breast_top9
#SBATCH --mem                 100G
#SBATCH --nodes               1
#SBATCH --ntasks              1
#SBATCH --output              /l/users/shikhar.srivastava/workspace/hover_net/logs/top9/slurm/%j.out
#SBATCH --time                23:00:00

ulimit -u 10000
/home/shikhar.srivastava/miniconda3/envs/hovernet_11/bin/python /l/users/shikhar.srivastava/workspace/hover_net/run_train.py    --gpu 0,1,2,3    --bucket_step_string top9    --organ Breast