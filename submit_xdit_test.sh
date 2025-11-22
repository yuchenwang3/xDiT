#!/bin/bash
#SBATCH --job-name=xdit_test
#SBATCH --output=xdit_test_%j.out
#SBATCH --error=xdit_test_%j.err
#SBATCH --account=bcrn-delta-gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Load necessary modules
module load anaconda3_gpu

# Set environment
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

# Activate xDiT conda environment
conda activate xdit

# Set HuggingFace cache to scratch directory
export HF_HOME="/scratch/bcrn/yuchen87/huggingface_cache"
mkdir -p $HF_HOME

# Print job information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Print GPU information
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Run the xDiT test
echo "=== Running xDiT Test ==="
python test_xdit.py

echo ""
echo "=== Job Completed ==="
echo "End time: $(date)"