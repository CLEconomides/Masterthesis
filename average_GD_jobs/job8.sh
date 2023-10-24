#!/bin/bash
#SBATCH --job-name=avrg_8                     # Job name
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.economides@campus.lmu.de     # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks=1                           # Run on a multi CPU
#SBATCH --mem=4gb                            # Job memory request
#SBATCH --time=100:59:00                      # Time limit hrs:min:sec
#SBATCH --partition=NvidiaAll
#SBATCH --output=avrg_GD_8.log                 # Standard output and error log
pwd; hostname; date
echo "running"

# n_count_qubits, r, entangling_block_layers, n_param, seed, L*0.01
python3 ../main_cip_avrg_GD.py 8 8 4 2 16 4




echo "all scripts executed"
