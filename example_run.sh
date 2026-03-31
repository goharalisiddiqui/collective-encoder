#! /bin/bash
#SBATCH -J CollectiveEncoder
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p ###PARTITION###
#SBATCH --mem=4G
#SBATCH --gres=####GPU_RESOURCES###
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o ./slurm_logs/slurm-%J.out
#SBATCH -e ./slurm_logs/slurm-%J.err

unset $pref
if [ ! -z "${SLURM_JOB_ID}" ]; then
    echo "Running on compute node"
    pref='srun'
    mkdir -p slurm_logs
else 
    echo "Running on local machine"
    pref=''
fi

if test ! -d .venv; then
    echo "Creating local virtual environment"
    python -m venv .venv
    source .venv/bin/activate
    pip install .
else
    echo "Activating local virtual environment in .venv"
    source .venv/bin/activate
fi

#Read command line arguments
while getopts d flag
do
    case "${flag}" in
        d) debug=1;;
    esac
done

if [ "$debug" == 1 ]; then
    $pref collective-encoder-train --config config.yaml --debug
else
    $pref collective-encoder-train --config config.yaml
fi