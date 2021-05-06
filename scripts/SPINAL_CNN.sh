#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu 12500MB
#SBATCH --ntasks 8
#SBATCH --nodes=1
#SBATCH --output="spinal.slurm"
#SBATCH --time 36:00:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export ENV_PATH="/lustre/scratch/grp/fslg_internn/env/internn"
export PATH="$ENV_PATH:$PATH"
eval "$(conda shell.bash hook)"
conda activate $ENV_PATH

cd "/lustre/scratch/grp/fslg_internn/internn"
which python
python -u main_both.py
