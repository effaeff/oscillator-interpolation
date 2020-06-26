#!/bin/bash -l

#SBATCH --time=02:00:00
#SBATCH -N 1 -c 48 -C cquad01
#SBATCH --partition=short

#SBATCH --mem=200000

#SBATCH --mail-user=felix.finkeldey@tu-dortmund.de
#SBATCH --mail-type=ALL
#SBATCH -o /work/smfefink/oscillator-interpolation/out.txt

cd /work/smfefink/oscillator-interpolation
export PYTHONPATH=$PYTHONPATH:/work/smfefink/pylib

#module purge
#module load gcc/7.3.0
echo "sbatch: START SLURM_JOB_ID $SLURM_JOB_ID (SLURM_TASK_PID $SLURM_TASK_PID) on $SLURMD_NODENAME"
echo "sbatch: SLURM_JOB_NODELIST $SLURM_JOB_NODELIST"
echo "sbatch: SLURM_JOB_ACCOUNT $SLURM_JOB_ACCOUNT"
/work/smfefink/anaconda3/bin/python scripts/interpolate_frf.py 

