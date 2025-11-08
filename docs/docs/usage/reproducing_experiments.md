# Reproducing Experiments

Experiments run through AIDO.ModelGenerator generate `config.yaml` and `<model>.ckpt` artifacts which can be used to reproduce training and evaluation, or extend experiments and improve models.

Many pre-packaged experiments are available in the [experiments](https://github.com/genbio-ai/ModelGenerator/blob/main/experiments) directory with instructions to run inference, evaluate, and reproduce training runs.

## Experiment Configs

When you start a run, AIDO.ModelGenerator saves a full `config.yaml` file for every experiment run which includes all user-specified and default args.
To reproduce a training run

```
mgen fit --config config.yaml
```

The location of the saved `config.yaml` varies by the logger used, but by default this will be under `logs/lightning_logs/version_X/config.yaml`.

Checkpoints can be used immediately to reproduce test metrics or run inference on new data.

## Experiment Checkpoints

If you use checkpointing callbacks (enabled by default), AIDO.ModelGenerator will save `<model>.ckpt` files within your logging directory.
These can be used with the saved `config.yaml` to run evaluation or prediction.

```
# Reproduce test metrics
mgen test --config config.yaml \
  --ckpt_path logs/lightning_logs/version_X/checkpoints/<my_best_val>.ckpt

# Reproduce predictions
mgen predict --config config.yaml \
  --ckpt_path logs/lightning_logs/version_X/checkpoints/<my_best_val>.ckpt \
  --config configs/examples/save_predictions.yaml
```

## Pre-packaged Experiments

Experiment configs and scripts from studies using AIDO.ModelGenerator are available in the [experiments](https://github.com/genbio-ai/ModelGenerator/blob/main/experiments) directory with instructions to run inference, evaluate, and reproduce training runs.

Below are some general guidelines for reproducing experiments, or adding new experiments to the repository.
For more details on each study, see the [Studies](../studies/index.md) page.

### Finetuning Experiments

To reproduce a training run with a saved config, simply run
```
mgen fit --config config.yaml
```

> Note: If you are using a full frozen config, some machine-specific parameters like `trainer.num_nodes` may need to be adjusted.

To evaluate the results, take the `best_val` checkpoint and run
```
mgen test --config config.yaml --ckpt_path logs/lightning_logs/version_X/checkpoints/<my_best_val>.ckpt
```

In some cases, experiments will be more complicated than a single config (e.g. for cross-validation)
In these cases, look for a README, a bash script, or contact the study authors.

### Zero-shot Experiments

If an experiment is zero-shot and only requires inference, you can test the model without any training
```
mgen test --config config.yaml
```

### Inference Experiments

If an experiment uses predictions for a downstream task, you can run inference on new data
```
mgen predict --config config.yaml --config configs/examples/save_predictions.yaml
```
and use any associated post-processing scripts with the experiment to compile and plot the results.
