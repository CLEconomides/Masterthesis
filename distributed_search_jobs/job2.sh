#!/bin/bash
#SBATCH --job-name=C6_2                      # Job name
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.economides@campus.lmu.de     # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks=1                           # Run on a multi CPU
#SBATCH --mem=4gb                            # Job memory request
#SBATCH --time=100:59:00                      # Time limit hrs:min:sec
#SBATCH --partition=All
#SBATCH --output=Anomaly2.log                 # Standard output and error log
pwd; hostname; date
echo "running"

# n_qbts, training_lengths, lrs, hyperparameters, seed, n_loss_param, rr, job
python3 ../main_cip1.py [3,4,5,6,7,8,9,12,15] [1,4,8] [0.05,0.2] [0,0.95] 5 1 7 2

echo "all scripts executed"
