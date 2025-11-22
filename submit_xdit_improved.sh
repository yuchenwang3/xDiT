#!/bin/bash
#SBATCH --job-name=xdit_test
#SBATCH --output=xdit_test_%j.out
#SBATCH --error=xdit_test_%j.err
#SBATCH --account=bejc-delta-gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Load necessary modules (as per guide)
module load anaconda3_gpu

# Set environment using existing miniconda (based on error message)
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate xdit

# Set HuggingFace cache to local directory (fallback due to permission issues)
export HF_HOME="$HOME/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Print job information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""

# Print system information
echo "=== System Information ==="
echo "Available modules:"
module list 2>&1
echo ""

# Print GPU information
echo "=== GPU Information ==="
nvidia-smi
echo ""
echo "GPU topology:"
nvidia-smi topo -m
echo ""

# Print environment information
echo "=== Environment Information ==="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA version: $(nvcc --version | grep release | head -1)"
echo ""

# Check xDiT installation
echo "=== xDiT Installation Check ==="
python -c "import xdit; print(f'xDiT version: {xdit.__version__}')" 2>/dev/null || echo "xDiT not found or import failed"
echo ""

# Check torch and CUDA availability
echo "=== PyTorch and CUDA Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "PyTorch check failed"
echo ""

# Run the xDiT test
echo "=== Running xDiT Test ==="
echo "Command: python test_xdit.py"
echo "=========================="
python test_xdit.py

# Check exit code
EXIT_CODE=$?
echo ""
echo "=== Test Results ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully"
else
    echo "✗ Test failed with exit code: $EXIT_CODE"
fi

echo ""
echo "=== Job Completed ==="
echo "End time: $(date)"
echo "Duration: $SECONDS seconds"
exit $EXIT_CODE