#!/bin/bash
#SBATCH -J CE_Training
### number of cores
#SBATCH -c 4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p wom
##SBATCH -B 1:8:2
#SBATCH --mem=20G
#SBATCH --gpus=1
##SBATCH --gres=shard:4
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
    module load spack_x86_64_v3
    module load python/3.9-torch2-cuda12
elif [ "$SLURM_JOB_PARTITION" == "wom" ] ; then
    ## WOM
    watch -n 1 "nvidia-smi -q -d UTILIZATION >> /home/ge45daw/slurm_logs/slurm-$SLURM_JOB_ID.gpu" &> /dev/null &
    module load spack_x86_64_v3
    module load python/3.9-torch2-cuda12
elif [ "$SLURM_JOB_PARTITION" == "carlos" ] ; then
    ## CARLOS
    watch -n 1 "nvidia-smi -q -d UTILIZATION >> /home/ge45daw/slurm_logs/slurm-$SLURM_JOB_ID.gpu" &> /dev/null &
    module load spack_x86_64_v3
    module load python/3.9-torch2-cuda12
elif [ "$SLURM_JOB_PARTITION" == "microcloud" ] ; then
    ## CARLOS
    module load spack_x86_64_v3
    module load python/3.9-torch2-cuda12
elif [ "$SLURM_JOB_PARTITION" == "zencloud" ] ; then
    ## ZENCLOUD
    module load spack_x86_64_v3
    module load python/3.9-torch2-cuda12
fi
########################################




srun python engine.py    \
                    --inputfile $PWD/../enhanced_md/INPUTS \
                    --outpath . \
                    --modelpath . \
                    --save_model \
                    --save_checkpoint \
                    --nepochs 10 \
                    --labels "" \
                    --output_to_file \
                    --networktype 'VAESimple' \
                    --normalize \
                    --beta=1.0 \
                    --lrate=0.01 \
                    --gpu \
                    # --outfolder 'VAESimple_Short_l9_5000' \
                    # --network "200,9" \

