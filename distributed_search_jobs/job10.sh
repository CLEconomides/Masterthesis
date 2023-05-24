#!/bin/bash
#SBATCH --job-name=C6_10                  # Job name
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.economides@campus.lmu.de     # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks=1                           # Run on a multi CPU
#SBATCH --mem=4gb                            # Job memory request
#SBATCH --time=144:59:00                      # Time limit hrs:min:sec
#SBATCH --partition=All
#SBATCH --output=Anomaly10.log                 # Standard output and error log
pwd; hostname; date
echo "running"

python3 ./main_cip1.py [3,4,5] [1,4,8] [0.05,0.2] [0,0.95] 3 3 7 10

echo "all scripts executed"
