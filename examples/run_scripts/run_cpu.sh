#! /bin/bash
#SBATCH -J SG
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p zencloud
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o ./slurm_logs/slurm-%J.out
#SBATCH -e ./slurm_logs/slurm-%J.err

source run.sh