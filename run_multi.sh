#!/bin/bash
#SBATCH -J CE_Training
### number of cores
#SBATCH -N 1
#SBATCH --ntasks 6
#SBATCH --cpus-per-task 8
#SBATCH -p wom
##SBATCH -B 1:8:2
#SBATCH --mem=20G
##SBATCH --gpus=3
#SBATCH --gres=shard:8
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


if test -d .venv; then
  source .venv/bin/activate
fi



i=1
<<<<<<< HEAD
for beta in `seq 0.0 5.0 500.0`; do

    (srun --ntasks 1 --exact --cpus-per-task 2 --mem=2G --gres=shard:1 python engine.py      \
                    --inputfile $DATA_DIR/20221201_COLLECTIVE_ENCODER_TRAINING_DATA_CKIT/INPUTS \
                    --outpath ./beta-sweep_VAE_1000 \
                    --modelpath . \
                    --tblogger \
                    --nepochs 1000 \
=======
# for beta in `seq 0.0 5.0 5.0`; do
for beta in 0.0001 0.001 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.5 0.7 1.0 1.2 1.3 1.5 1.7 2.0 ; do

    (srun --ntasks 1 --cpus-per-task 8 --mem=7G --gpus 1 python engine.py \
                    --inputfile ../datasets/INPUTS_heavy \
                    --outpath ./beta_search_ld6_lr0p001_long \
                    --nepochs 100 \
>>>>>>> c8b56d7 (Add avg.sigma printout in vae_net)
                    --nexp $i \
                    --labels "dist_Au-K1" \
                    --output_to_file \
<<<<<<< HEAD
=======
                    --network "200,100,50,6" \
>>>>>>> c8b56d7 (Add avg.sigma printout in vae_net)
                    --networktype 'VAE' \
                    --network "1500,1000,2" \
                    --normalize \
<<<<<<< HEAD
                    --beta=$beta \
=======
                    --plot_every 50 \
>>>>>>> c8b56d7 (Add avg.sigma printout in vae_net)
                    --save_checkpoint \
                    --plot_every 100 \
                    --lrate=0.0001) &

<<<<<<< HEAD
                    # --outfolder 'CKIT_VAE' \
=======
                    # --tblogger \
>>>>>>> c8b56d7 (Add avg.sigma printout in vae_net)
                    # --modelfile "./beta_search_500_norm/ce_training_10/VAE_mse_checkpoint" \
                    # --load_model \
                    # --outfolder 'ce_training' \

    i=$((i+1))

done

wait