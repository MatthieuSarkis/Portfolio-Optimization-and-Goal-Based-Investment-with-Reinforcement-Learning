#!/bin/bash

#SBATCH --job-name="SAC_agent"
#SBATCH --output="%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="%j.err" # job standard error file (%j replaced by job id)

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 8 processor core(s) per node 
#SBATCH --mem=5G   # maximum memory per node
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu    # gpu node(s)

#========================================================

#srun --unbuffered 
python src/main.py \
--mode train \
--n_episodes 2 \
--assets_to_trade portfolios_and_tickers/tickers_S\&P500_dummy.txt \
--initial_portfolio portfolios_and_tickers/initial_portfolio_subset.json \
--plot \
--use_corr_matrix \
#--checkpoint_directory saved_outputs/2021.07.26.21.49.56 \


