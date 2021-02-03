#!/bin/bash
#SBATCH --job-name=Project7
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=yongkang.long@kaust.edu.sa
#SBATCH --mail-type=END
#SBATCH --output=LOG/log.%J
#SBATCH --time=3:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
##SBATCH -a 0-39
##SBATCH --dependency afterok:6686802_[1-100] 

#python different_optimizer.py 
#python different_optimizer5.py
#python different_optimizer3.py
python different_optimizer7.py
