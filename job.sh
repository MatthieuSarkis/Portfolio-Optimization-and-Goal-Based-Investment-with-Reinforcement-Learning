#!/bin/bash

#SBATCH --job-name="SAC"
#SBATCH --output="SAC.out" # job standard output file (%j replaced by job id)
#SBATCH --error="SAC.err" # job standard error file (%j replaced by job id)

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 8 processor core(s) per node 
#SBATCH --mem=5G   # maximum memory per node
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu    # gpu node(s)

#========================================================

#srun --unbuffered 
python src/main.py \
--initial_cash 100000 \
--buy_rate 0.01 \
--sell_rate 0.01 \
--bank_rate 0.0 \
--sac_temperature 2.0 \
--limit_n_stocks 500 \
--lr_Q 0.0003 \
--lr_pi 0.0003 \
--gamma 0.99 \
--tau 0.005 \
--batch_size 256 \
--layer1_size 256 \
--layer1_size 256 \
--n_episodes 500 \
--seed 42 \
--mode train \
--memory_size 1000000 \
--initial_date 2015-01-01 \
--final_date 2020-12-31 \
--gpu_devices 0 1 2 3 \
--buy_rule most_first\
--auto_temperature \
