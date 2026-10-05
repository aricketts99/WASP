#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -cwd
#$ -l h_vmem=20G
#$ -l h_rt=00:15:00
#$ -pe sharedmem 4
#$ -l h_rss=20G
#$ -o outputs
#$ -e errors

sh ./jobscript.sh $SGE_TASK_ID
