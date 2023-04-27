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
#SBATCH --output=./log/LSTM_for_N19_lbfgs_50k_train_50_80_new_dataset_%j.log   # Standard output and error log
pwd; hostname; date

for n in {50..80}
	do
	srun python test.py --run $n --epochs 300 --training FOR --nshells 19 --model LSTM --optim lbfgs --batch-size 1000 --lr 0.001 --patience 200
	echo "finished run " $n
	done
date
