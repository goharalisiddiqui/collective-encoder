#!/bin/bash
#SBATCH -J CE_Training  
### number of cores
#SBATCH -c 16
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p gpucloud
##SBATCH -B 1:8:2
#SBATCH --mem=30G
##SBATCH --gpus=1
#SBATCH --gres=shard:4
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o /home/ge45daw/slurm_logs/slurm-%J.out
#SBATCH -e /home/ge45daw/slurm_logs/slurm-%J.err




####### CLEANING THE ENV #######
module purge
spack env deactivate
spack unload --all
################################

####### LOADING REQUIRED MODULES #######
if [ "$SLURM_JOB_PARTITION" == "gpucloud" ] ; then
    ## GPUCLOUD
    watch -n 1 "nvidia-smi -q -d UTILIZATION >> /home/ge45daw/slurm_logs/slurm-$SLURM_JOB_ID.gpu" &> /dev/null &
    module load spack_skylake_avx512
    spack env activate ml_skylake_avx512
elif [ "$SLURM_JOB_PARTITION" == "carlos" ] ; then
    ## CARLOS
    watch -n 1 "nvidia-smi -q -d UTILIZATION >> /home/ge45daw/slurm_logs/slurm-$SLURM_JOB_ID.gpu" &> /dev/null &
    module load spack_x86-64-2
    spack env activate python-pytorch_haswell
elif [ "$SLURM_JOB_PARTITION" == "zencloud" ] ; then
    ## ZENCLOUD
    echo "No ML env for spack implemented for zencloud"
fi
########################################





srun python engine.py    \
                    --inputfile $DATA_DIR/20221016_COLLLECTIVE_ENCODER_TRAINING_DATA_OAH/INPUTS \
                    --outpath ./saved_runs \
                    --modelpath . \
                    --save_model \
                    --nepochs 5000 \
                    --labels "dist_hg.z" \
                    --output_to_file \
                    --gpu \
                    --network "1000,500,100,50,20" \
                    --normalize \
                    --beta=1.5 \
                    --lrate=0.00001

