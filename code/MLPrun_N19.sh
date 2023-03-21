#!/bin/bash
#SBATCH --job-name=sp_MLP_19   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniele_noto@yahoo.it     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --gres=gpu:1 			#Run on GPU
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=10 
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --output=./log/MLP_std_19_%j.log   # Standard output and error log
pwd; hostname; date

for n in {1..10}
	do
	srun python test.py --run $n --epochs 200 --nshells 19
	echo "finished run " $n
	done
date
