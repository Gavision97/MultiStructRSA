#!/bin/bash
#
#SBATCH --job-name sinteractive
#SBATCH --partition main
#SBATCH --qos normal

#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gpus=0
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH -o /home/gavrilev/Thesis/src/interactive_%J.out

#export NODE_TLS_REJECT_UNAUTHORIZED='0'

echo Sat 09 May 2026 17:23:25 IDT
launch='sleep 600'
echo " STARTING JOB WITH THE COMMAND:  $launch "
while true; do
        eval $launch
done
