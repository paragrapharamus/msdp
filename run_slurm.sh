#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va4317
export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
conda activate msdp
source /vol/cuda/11.1.0-cudnn8.0.4.30/setup.sh
TERM=vt100 # or TERM=xterm

/usr/bin/nvidia-smi
uptime

nohup python -u run.py > out/exp.log 2>&1

