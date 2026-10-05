#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N exp_2
#$ -cwd
#$ -l h_rt=00:10:00
#$ -pe sharedmem 4
#$ -l h_vmem=16G
#$ -o outputs
#$ -e errors

. /etc/profile.d/modules.sh

module load anaconda/2024.02

conda activate wasp

python3 experiment_07.py $1
