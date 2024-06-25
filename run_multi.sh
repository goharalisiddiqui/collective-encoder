#!/bin/bash
#SBATCH -J CE_Training
### number of cores
#SBATCH -N 1
#SBATCH --ntasks 8
#SBATCH --cpus-per-task 8
#SBATCH -p wom
##SBATCH -B 1:8:2
#SBATCH --mem=64G
#SBATCH --gpus=3
##SBATCH --gres=shard:4
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o /home/ge45daw/slurm_logs/slurm-%J.out
#SBATCH -e /home/ge45daw/slurm_logs/slurm-%J.err




####### CLEANING THE ENV #######
module purge
################################

####### LOADING REQUIRED MODULES #######
if [ "$SLURM_JOB_PARTITION" == "gpucloud" ] ; then
    ## GPUCLOUD
    # watch -n 1 "nvidia-smi -q -d UTILIZATION >> /home/ge45daw/slurm_logs/slurm-$SLURM_JOB_ID.gpu" &> /dev/null &
    module load spack_x86_64_v3
    module load python/3.9-torch2-cuda12
elif [ "$SLURM_JOB_PARTITION" == "wom" ] ; then
    ## WOM
    # watch -n 1 "nvidia-smi -q -d UTILIZATION >> /home/ge45daw/slurm_logs/slurm-$SLURM_JOB_ID.gpu" &> /dev/null &
    module load spack_x86_64_v3
    module load python/3.9-torch2-cuda12
elif [ "$SLURM_JOB_PARTITION" == "carlos" ] ; then
    ## CARLOS
    # watch -n 1 "nvidia-smi -q -d UTILIZATION >> /home/ge45daw/slurm_logs/slurm-$SLURM_JOB_ID.gpu" &> /dev/null &
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
# iter=1
# beta=115.0
# while getopts i:b: flag
# do
#     case "${flag}" in
#         i) iter=${OPTARG};;
#         b) beta=${OPTARG};;
#     esac
# done

source .venv/bin/activate



i=1
for beta in `seq 0.0 5.0 5.0`; do

    (srun --ntasks 1 --overlap --cpus-per-task 8 --mem=7G python engine.py \
                    --inputfile ../datasets/INPUTS_heavy \
                    --outpath ./ \
                    --nepochs 5 \
                    --tblogger \
                    --nexp $i \
                    --labels phi psi \
                    --output_to_file \
                    --network "200,2" \
                    --networktype 'VAE' \
                    --beta=$beta \
                    --lrate=0.001 \
                    --normalize \
                    --plot_every 100 \
                    --gpu \
                    --save_checkpoint \
                    --overwrite) &

                    # --modelfile "./beta_search_500_norm/ce_training_10/VAE_mse_checkpoint" \
                    # --load_model \
                    # --outfolder 'ce_training' \

    i=$((i+1))

done

wait