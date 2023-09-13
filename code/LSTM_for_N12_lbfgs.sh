#!/bin/bash
#SBATCH --job-name=LSTMfnoise   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniele_noto@yahoo.it     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --gres=gpu:1 			#Run on GPU
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=10 
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=./log/LSTM_for_N12_lbfgs_N12_1_20_noise1e_05_%j.log   # Standard output and error log
pwd; hostname; date

for n in {1..20}
	do
	srun python test.py --run $n --epochs 300 --training FOR --nshells 12 --model LSTM --optim lbfgs --noise 1e-5 --lr 0.001 
	echo "finished run " $n
	done
date
