# How to use W&B Sweeps with ModelGenerator for hyperparameter tuning

## Caveats
W&B agents cannot launch multi-node training jobs, which causes great difficulties integrating W&B Sweeps with ModelGenerator. This guide is based on a hacky workaround that introduces many limitations.

### The workaround
An agent is configured to exit immediately after retrieving the next set of hyperparamenters and outputing the complete training command to stdout. This command is then executed on each node without being monitored by an active agent.

### Limitations
1. All agent functionalities are lost. It is not possible to use agent to start/stop/resume/update training runs. Users must manually terminate training runs or implement early-stopping mechanisms.
2. Failed runs have to be re-run manually using your own sbatch scripts. The command for that run is availale in stdout of the failed run.
3. Parameter importance plots use wrong parameters by default, it can be manually fixed by selecting the right parameter names in your mgen config.

>**NOTE**: Before proceeding, please make sure that your training job uses **WandbLogger**.
## SLURM
### Step 1: create a wandb sweep
The default `slurm_sweep.yaml` creates a wandb sweep with the training command `mgen fit --config .local/test.yaml` under the project `autotune-test`. Please modify it to suit your experiments. Key values to change are **project**, **command** and **parameters**.

Run the following command to create a wandb sweep:
```bash
wandb sweep scripts/wandb_sweep/slurm_sweep.yaml
```
Take a note of your sweep ID for step 2. It looks like `<entity>/<project>/<id>` and is found in the output: `wandb: Run sweep agent with: wandb agent`
### Step 2: submit the next training job to SLURM
Similar to step 1, you need to edit `slurm_agent.sh` for your experiment. The most important changes are **WANDB_PROJECT** and **SWEEP_ID**.

The following command creates one sweep agent that runs training with the next set of hyperparamenters.
```bash
sbatch scripts/wandb_sweep/slurm_agent.sh
```

>**TIPS**: To queue your other sweep runs, use `sbatch --dependency`. To launch your other sweep runs in parallel, use `sbatch --array=1-X` where `X` is the number of parallel runs.
