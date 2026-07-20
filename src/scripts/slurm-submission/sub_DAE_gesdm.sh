#!/bin/bash
#SBATCH --nodes=1                  # Request 2 nodes
#SBATCH -t 100:00:00                # Request 12 hours in walltime.
#SBATCH -J DAE_P6_M1           # Job name
#SBATCH -o log_%x.o%j              # output file
#SBATCH -e log_%x.e%j              # error file
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2          # raise only if your code is multithreaded
#SBATCH --gres=mps:25              # 100/25 = 4 GPU slices; GPU is not the bottleneck
#SBATCH --mem=2500M
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Keras 3 backend selection. Change to "tensorflow", "torch", or "jax".
# Overridable from the command line, e.g.:  KERAS_BACKEND=torch sbatch sub_DAE_mochi.sh <file>
# The DAE script's configureDevices() is backend-agnostic and will honor the
# single GPU that Slurm's shard + ConstrainDevices exposes to this job.
# ---------------------------------------------------------------------------
export KERAS_BACKEND="${KERAS_BACKEND:-tensorflow}"

# Optional diagnostics: confirm which card this job was given.
echo "Backend              : $KERAS_BACKEND"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<unset, cgroup-constrained>}"
nvidia-smi -L 2>/dev/null || true


#command="srun train_rruff_raman.sh $SLURM_JOB_NAME "
#command="srun train_rruff_raman.sh $1"
#command="srun ../train_rruff_ir.sh $(basename $(pwd))"

command="DataML_DAE -a $1"

echo $command

$command
