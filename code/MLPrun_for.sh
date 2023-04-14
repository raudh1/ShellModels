#!/bin/bash
#SBATCH --job-name=spectrum_MLP   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniele_noto@yahoo.it     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --gres=gpu:1 			#Run on GPU
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=10 
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --output=./log/MLP_for_12_%j.log   # Standard output and error log
pwd; hostname; date

for n in {1..10}
	do
	srun python test.py --run $n --epochs 1000 --train FOR --lr 0.002
	echo "finished run " $n
	done
date
