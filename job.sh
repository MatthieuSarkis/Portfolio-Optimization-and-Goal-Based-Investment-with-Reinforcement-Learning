#!/bin/bash

#SBATCH --job-name="net2"
#SBATCH --output="%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="%j.err" # job standard error file (%j replaced by job id)

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 8 processor core(s) per node 
#SBATCH --mem=5G   # maximum memory per node
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu    # gpu node(s)

#========================================================

python main.py \
--data_path './data.pickle' \
--pid 360 \
--odir 'saved_files' \
--random_state 42 \
--test_size 0.2 \
--patience 10 \
--epochs 100 \
--batch_size 32 \
--n_layers 5 \
--dropout_rate 0.3 \
--n_gpus 4 \
--specific_stock \
--dumped_data \
--preprocess \
# --only_rev_vol \
# --specific_stock \
# --for_RNN
