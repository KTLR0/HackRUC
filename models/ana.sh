#!/bin/bash

#SBATCH --job-name=enthusiasm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/slurm.%N.%j.%x.out
#SBATCH --error=slurm_logs/slurm.%N.%j.%x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amb755@scarletmail.rutgers.edu

echo "Starting Enthusiasm Detector job..."

# Create directories for logs and models
mkdir -p slurm_logs/
mkdir -p saved_models/

# -------------------------------
# Activate the virtual environment
# -------------------------------
source $HOME/enthusiasm_env/bin/activate
echo "Using Python: $(which python)"


# -------------------------------
# Run the Python script
# -------------------------------
srun -N1 -n1 $HOME/enthusiasm_env/bin/python enthusiasm_detector.py --save_model_path saved_models/enthusiasm_model.pt

echo "Finished Enthusiasm Detector job."
