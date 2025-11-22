(base) [yuchen87@dt-login01 xDiT]$ accounts

Project Summary for User 'yuchen87':

Account                        Balance(Hours)   Deposited(Hours)  Project
----------------------------  ----------------  ----------------  ----------------------
bche-delta-gpu                             635             11173  applied parallel pr...
bejc-delta-gpu                            1965              5902  machine learning sy...

use bejc for this workspace



Basic


Pre-set environment
Delta has some prepared environments, you can use them via
module load anaconda3_cpu/gpu/mi100
You may always want to do this, either interactively or not, before running any code;
You can also create your own environment based on it

Disk
You will be arranged to a certain disk (/u/<your ACCESS id>), you can see it via
Quota
Note: the root directory assigned to you is rather limited, the better option may be using /scratch/bcrn/<your ACCESS id> (and remember to put huggingface cache and maybe environments there)

Running in a node (interactively)
Before running in a node, you would be in a login node where there are no gpus
For us, the accountname would be bcrn-delta-gpu
Before running, you may want to export environment variables, via
export PATH=...
Typing them in the shell or put them into .bashrc
You can get a interactive GPU shell via
srun -A accountname --time=00:10:00 --nodes=1 --ntasks-per-node=16 --partition=gpuA40x4 --gpus=4 --mem=32g --pty /bin/bash
Interactive means that after you are granted the node, you can do anything with the command line with the gpu (however interactive nodes would have a 1-hour time limit)

Running with job submission
You can submit a job with (script.sh) via
sbatch --job-name=$job_name --output=$log_path -A accountname --time=01:00:00 --nodes=1 --ntasks-per-node=16 --partition=gpuA40x4 --gpus=4 --mem=32g \
./scripts.sh $some-input-variable-to-the-scripts

You can submit a job with directly in the command line with something like this
sbatch --job-name=$job_name --output=$log_path -A accountname --time=01:00:00 --nodes=1 --ntasks-per-node=16 --partition=gpuA40x4 --gpus=4 --mem=32g \
wrap=”python helloworld.py”

You can cancel a job via
scancel #jobid
scancel –user=$USER
You can watch what’s going on on the account queue via
watch squeue -A accountname



































Specific Usages
vLLM
Installation:
Module load cuda/12.3.0
conda create -n vllm_env python=3.8 -y
conda activate vllm_env
pip install vllm
ToT
Installation:
conda activate vllm_env (assume you finish vLLM installation)
pip install tomli
pip install flash-attn –no-build-solation
pip install absl-py wrapt gast astunparse
git clone https://github.com/princeton-nlp/tree-of-thought-llm
pip install -r requirements.txt
pip install -e .

Huggingface models
Installation:
module load anaconda3_gpu
install appropriate transformer and corresponding packages
install vllm, flash_attn and so on if your model needs it
Note: remember to change the HF_HOME to /scratch/bcrn/<your ACCESS id>
Multi-node GPU parallelism
Create a job script with the proper number of GPU you need
#!/bin/bash
#SBATCH --job-name=multinode-example

#SBATCH --nodes=2

#SBATCH --partition=gpuA100x4

#SBATCH --ntasks=2

#SBATCH --gpus-per-task=1

#SBATCH --cpus-per-task=4


module load anaconda3_gpu


