#!/bin/bash
#SBATCH --job-name=Style_Transfer
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=yongkang.long@kaust.edu.sa
#SBATCH --mail-type=END
#SBATCH --output=LOG/log.%J
#SBATCH --time=2:00:00
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH -a 0-244
##SBATCH --dependency afterok:6686802_[1-100] 

#source activate ML

echo "This is job #${SLURM_ARRAY_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}"


STEPS=(2000 3000 4000 5000 6000)   ####5
STYLE_WEIGHTS=(1e2 1e4 1e6 1e8 1e10 1e12 1e14) ##7
LRS=(0.1 0.01 0.007 0.005 0.003  0.001 0.0005) ##7

index1=$(( $SLURM_ARRAY_TASK_ID % 5 ))
index2=$(( $SLURM_ARRAY_TASK_ID / 5 % 7))
index3=$(( $SLURM_ARRAY_TASK_ID / 35 ))

step=${STEPS[$index1]}
style_weight=${STYLE_WEIGHTS[$index2]}
lr=${LRS[$index3]}
echo $step
echo $style_weight
echo $lr

python project_style-transfer_task1_4.py --style_weight $style_weight --steps $step --lr $lr
