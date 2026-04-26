#!/bin/bash
#SBATCH --job-name=dast
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=dast_%j.log
#SBATCH --error=dast_%j.err

echo "1. JOB STARTED"

cd ~/DaST || exit
echo "2. IN DIRECTORY: $(pwd)"

source dast-env/bin/activate || echo "FAILED ENV ACTIVATION"

echo "3. PYTHON PATH:"
which python

echo "4. CHECKING GPU"
nvidia-smi

echo "5. STARTING PYTHON"

python -u run_dast.py --dataset=mnist --cuda --workers=1 --niter=50