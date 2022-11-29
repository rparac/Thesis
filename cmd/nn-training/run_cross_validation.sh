#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=rp218 # required to send email notifcations - please replace <your_username> with your college login name or email address

# venv setup
export PATH=/vol/bitbucket/${USER}/thesis-venv/bin/:$PATH
source activate

# setup working environment
export PYTHONPATH=${PYTHONPATH}:/vol/bitbucket/${USER}/luke-for-roko
source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi

# Run the actual model
python cross_val.py

# How long did the script run for
uptime

