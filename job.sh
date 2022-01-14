#!/bin/bash
#
#SBATCH --job-name=init_job
#SBATCH --time=23:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=128
#SBATCH --mem=100G
#SBATCH --gres=gpu:4
#SBATCH --output=/l/users/shikhar.srivastava/workspace/hover_net/logs/%j.out

ulimit -u 10000
/home/shikhar.srivastava/miniconda3/envs/hovernet_11/bin/python /l/users/shikhar.srivastava/workspace/hover_net/run_train.py --gpu='0,1,2,3'