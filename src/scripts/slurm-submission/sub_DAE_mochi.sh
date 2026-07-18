#!/bin/bash
#SBATCH --nodes=1                  # Request 2 nodes
#SBATCH -t 100:00:00                # Request 12 hours in walltime.
#SBATCH -J DAE_P6_M1           # Job name
#SBATCH -o log_%x.o%j              # output file
#SBATCH -e log_%x.e%j              # error file
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          # raise only if your code is multithreaded and have free cores
#SBATCH --gres=mps:8              # ~8% of the GPU; ~12 jobs share it
#SBATCH --mem=2500M
#SBATCH --gres=shard:1
#SBATCH --export=ALL

#command="srun train_rruff_raman.sh $SLURM_JOB_NAME "
#command="srun train_rruff_raman.sh $1"
#command="srun ../train_rruff_ir.sh $(basename $(pwd))"

command="DataML_DAE -a $1"

echo $command

$command
