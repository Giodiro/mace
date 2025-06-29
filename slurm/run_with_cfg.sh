#!/bin/bash

set -eux

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment.yaml>"
    exit 1
fi
INPUT_YAML="$1"

EXP_DIR="/leonardo_work/IIT24_AtomSim/franken/rebuttal/mace_finetune/"
if [ ! -d "$EXP_DIR" ]; then
    echo "ERROR: Experiment directory expected at '$EXP_DIR' does not exist. Please create it, or edit this file to point to an existing directory for experiments."
    exit 2
fi

# Extract exp_name
EXP_NAME=`awk -F': ' '/^name:/ {print $2}' "$INPUT_YAML"`
if [ -z "$EXP_NAME" ]; then
    echo "ERROR: Could not extract experiment name (key 'exp_name') from input file at '$INPUT_YAML'."
    exit 3
fi

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="${EXP_NAME}"
#SBATCH --output="${EXP_DIR}/${EXP_NAME}__%j.out"
#SBATCH --error="${EXP_DIR}/${EXP_NAME}__%j.out"
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --account=IscrC_ERLO
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=10

conda activate /leonardo/home/userexternal/gmeanti0/micromamba/envs/franken-dev
source activate /leonardo/home/userexternal/gmeanti0/micromamba/envs/franken-dev

PYTHONPATH='.' python -m mace.cli.run_train --config ${INPUT_YAML}

EOT