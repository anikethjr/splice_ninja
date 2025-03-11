#!/bin/bash
# Job name:
#SBATCH --job-name=splice_ninja
#
# Account:
#SBATCH --account=co_nilah
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# QoS:
#SBATCH --qos=savio_lowprio
#
# Number of tasks (one for each GPU desired for use case):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=16
#
# Request one GPU:
#SBATCH --gres=gpu:A40:2
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
#save output and error messages
#SBATCH --output=/global/scratch/users/aniketh/slurm_logs/slurm_job_%j.out
#SBATCH --error=/global/scratch/users/aniketh/slurm_logs/slurm_job_%j.err
#
# send email when job begins
#SBATCH --mail-type=begin  
# send email when job ends      
#SBATCH --mail-type=end  
# send email if job fails        
#SBATCH --mail-type=fail         
#SBATCH --mail-user=aniketh@berkeley.edu
#
## Command(s) to run:
#

source /global/home/users/aniketh/.bashrc
mamba activate /clusterfs/nilah/aniketh/mamba_envs/promoter_models

cd /global/home/users/aniketh/splice_ninja
