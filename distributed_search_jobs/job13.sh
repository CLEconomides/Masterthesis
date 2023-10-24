#!/bin/bash
#SBATCH --job-name=C6_13                  # Job name
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.economides@campus.lmu.de     # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks=1                           # Run on a multi CPU
#SBATCH --mem=4gb                            # Job memory request
#SBATCH --time=144:59:00                      # Time limit hrs:min:sec
#SBATCH --partition=All
#SBATCH --output=Anomaly13.log                 # Standard output and error log
pwd; hostname; date
echo "running"

# n_qbts, training_lengths, lrs, hyperparameters, seed, n_loss_param, rr, job
python3 ../main_cip1.py [2,3,4] [1,4,8] [0.05,0.2] [0,0.95] 1 4 5 13






echo "all scripts executed"
