#! /bin/bash
#SBATCH --job-name=SG
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=zencloud
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=./slurm_logs/slurm-%J.out
#SBATCH --error=./slurm_logs/slurm-%J.err

# Environment activation
source /mnt/projects/sne/Gohar/2.PROJECT_Enhanced_Sampling_Using_Autoencoder_CVs/betavae/.venv/bin/activate
