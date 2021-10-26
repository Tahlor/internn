#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu 12500MB
#SBATCH --ntasks 8
#SBATCH --nodes=1
#SBATCH --output="/lustre/scratch/grp/fslg_internn/internn/lm/slurm/scripts/variants/log_00_master_vgg_embeddings_default.slurm"
#SBATCH --time 72:00:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="/lustre/scratch/grp/fslg_internn/env/internn:$PATH"
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/grp/fslg_internn/env/internn

cd "/lustre/scratch/grp/fslg_internn/internn/lm"
which python
python -u /lustre/scratch/grp/fslg_internn/internn/lm/train_BERT.py --config '/lustre/scratch/grp/fslg_internn/internn/lm/configs/variants/00_master_vgg_embeddings_default.yaml'
    