#!/bin/bash
#SBATCH --job-name=spectrum_LSTM   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniele_noto@yahoo.it     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --gres=gpu:1 			#Run on GPU
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=10 
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=./log/LSTM_for_N12_lbfgs_10_50_%j.log   # Standard output and error log
pwd; hostname; date

for n in {10..50}
	do
	srun python test.py --run $n --epochs 300 --training FOR --nshells 12 --model LSTM --optim lbfgs
	echo "finished run " $n
	done
date
