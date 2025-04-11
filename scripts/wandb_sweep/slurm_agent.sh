#!/bin/bash
################################
#        SLURM options         #
################################
# uncomment to run multiple agents in parallel.
# --array=1-X where X is number agents.
##SBATCH --array=1-X
#SBATCH --ntasks-per-node=1  # same as trainer.devices
#SBATCH --nodes=1  # same as trainer.num_nodes
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err

################################
#   Python environment setup   #
################################
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate finetune

################################
#Required wandb sweep settings #
################################
export WANDB_PROJECT="autotune-test"
SWEEP_ID=""

################################
#   No change required below   #
################################
{
    IFS=$'\n' read -r -d '' AGENT_DETAILS;
    IFS=$'\n' read -r -d '' AGENT_COMMAND;
} < <((printf '\0%s\0' "$(timeout 30 srun --ntasks=1 wandb agent --count 1 $SWEEP_ID)" 1>&2) 2>&1)
RUN_ID=$(echo $AGENT_DETAILS | sed -e "s/.*\[\([^]]*\)\].*/\1/g" -e "s/[\'\']//g")
if [[ -z "$RUN_ID" ]]; then
   echo wandb agent timed out. >&2
   exit 1
fi
AGENT_COMMAND="${AGENT_COMMAND} --trainer.logger.version ${RUN_ID}"
echo Training command: $AGENT_COMMAND

wait
srun $AGENT_COMMAND
