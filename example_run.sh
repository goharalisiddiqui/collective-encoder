#!/bin/sh

####### PREPARE ENV #######
if [ ! -d ".venv" ]; then
    read -p "No virtual environment found. Create one? (y/n) " answer
    if [ "$answer" != "y" ]; then
        echo "Exiting. Please create a virtual environment and install dependencies before running."
        exit 1
    fi
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi
########################################

unset $pref
if [ ! -z "${SLURM_JOB_ID}" ]; then
    echo "Running on compute node"
    pref='srun'
    mkdir -p slurm_logs

else 
    echo "Running on local machine"
    pref=''
fi

#Read command line arguments
config_path="config.yaml"
extra_args=""
command="collective-encoder-train"
while getopts dc:pt flag
do
    case "${flag}" in
        d) extra_args="--debug";;
        c) config_path="${OPTARG}";;
        p) command="collective-encoder-prepare";;
        t) command="collective-encoder-test";;
    esac
done

$pref $command --config $config_path $extra_args