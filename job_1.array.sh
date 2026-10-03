#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -cwd
#$ -l h_vmem=20G

sh ./jobscript_1.sh $SGE_TASK_ID
