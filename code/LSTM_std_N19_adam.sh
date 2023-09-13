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
#SBATCH --output=./log/LSTM_std_N19_1_20_noise0_01_%j.log   # Standard output and error log
pwd; hostname; date

for n in {1..20}
	do
	srun python test.py --run $n --epochs 300 --training STD --nshells 19 --model LSTM --optim adam --batch-size 1000 --lr 0.005 --noise 0.01
	echo "finished run " $n
	done
date
