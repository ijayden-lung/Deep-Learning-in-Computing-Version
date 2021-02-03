#!/bin/bash
#SBATCH --job-name=FastTransfer
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=yongkang.long@kaust.edu.sa
#SBATCH --mail-type=END
#SBATCH --output=LOG/log.%J
#SBATCH --time=12:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:v100:4
##SBATCH --constraint=[rtx2080ti]

#source activate ML

echo "This is job #${SLURM_ARRAY_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}"


python project_style-transfer_task5_6.2.py
