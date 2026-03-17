#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-72:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name research_project			### name of the job
#SBATCH --output jobs/job-%J.out			### output log for running job - %J for job number
#SBATCH --error jobs/job-%J.err
#SBATCH --gpus=rtx_6000:1				### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=1

# Note: the following 4 lines are commented out

#SBATCH --mail-user=gavrilev@post.bgu.ac.il	### user's email for sending job status messages
#SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=24G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
module load cuda/11.8
source activate chemprop				### activate a conda environment, replace my_env with your conda environme
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export CUBLAS_WORKSPACE_CONFIG=:4096:8


export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface 

# use when running through vscode -> export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Check CUDA version
nvcc --version

# Check nvidia-smi
nvidia-smi
					### this command executes jupyter lab – replace with your own command
python main.py --exp_name "ex6" --res_file_name "cv_all_1403" --plot_name "1403_exp6" --log_msg "stratified_cv_1403_exp6 - MultiStructRSA - w 3D struct, w seq & with cross-attention fusion (wd=1e-6)" --log_file_name "stratified_cv_1403_exp6" --eval_method 'cv' --k_fold 5 --device 'cuda'
#python main.py --log_msg "cold_1802_exp1 - model w ft pt gRNAde & cross attn" --log_file_name "cold_1802_exp1" --eval_method 'cold_blind_cv' --cold_type 'rna' --device 'cuda'
#jupyter lab
