#!/bin/bash
#SBATCH --time=0-03:00              # Time limit (D-HH:MM)
#SBATCH --gpus-per-node=v100:2     # Request one A100 GPU (change type/number as needed)
#SBATCH --mem=3200M                 # Memory per node (4GB is a starting point)
#SBATCH --job-name=my_gpu_python
#SBATCH --output=%x-%j.out          # Standard output file

# 1. Load the modules used to create the virtual environment
module load python/3.10 scipy-stack

# 2. Activate the virtual environment
source $SCRATCH/mizanu/myspace/phiTS/myenv/bin/activate

# 3. Execute your Python script
python train.py

# The environment deactivates automatically when the script finishes.