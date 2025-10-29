#!/bin/bash
#SBATCH --job-name=fir_h100_job      # Job name
#SBATCH --time=0-6:00               # Max run time (D-HH:MM). Max is 7 days (168h).
#SBATCH --output=%x-%j-benchmark.out           # Output file
#SBATCH --error=%x-%j-benchmark.err            # Error file
#SBATCH --gpus=h100:4                # Request one full H100 (80GB)
#SBATCH --cpus-per-task=12           # Request 12 CPU cores (Recommended ratio: 12 cores per H100)
#SBATCH --mem=128G                    # Request 64 GB of system RAM (approx 1/4 of node memory for 1 GPU)

# --- Environment Setup ---
# cd $SLURM_SUBMIT_DIR                 # Change to submission directory

module --force purge                        # Clear all modules
module load StdEnv/2023              # Load a modern standard environment
module load cuda/12.2                # Load an appropriate CUDA version for H100

source $SCRATCH/myenv/bin/activate

# --- Job Execution ---
echo "Starting H100 GPU job on Fir node $HOSTNAME"
# or for a Python job
python3 train.py

echo "Job finished."