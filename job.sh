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

# Simple command line: In case you trust the default values I gave to the various params/hyperparams, cf. main.py

#srun --unbuffered 
python src/main.py \
--mode train \
--n_episodes 2 \
--assets_to_trade portfolios_and_tickers/tickers_S\&P500_subset.txt \
--initial_portfolio portfolios_and_tickers/initial_portfolio_subset.json \
#--checkpoint_directory saved_outputs/2021.07.23.18.38.10 \