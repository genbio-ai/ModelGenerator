# Tasks

Tasks define data and model usage.
They provide a simple interface for swapping backbones, adapters, and data without any code changes, enabling rapid and reproducible experimentation.
They are specified with the `--model` argument in the CLI or in the `model` section of a configuration file.

Tasks automatically configure [backbones](./backbones) and [adapters](./adapters) for training with `mgen fit`, evaluation with `mgen test/validate`, and inference with `mgen predict`.
They cover a range of use-cases for information extraction, domain adaptation, supervised prediction, generative modeling, and zero-shot applications.

This reference overviews the available no-code tasks for finetuning and inference.
If you would like to develop new tasks, see [Experiment Design](../../experiment_design).

```yaml
# Example Task Configuration
model:
  class_path: SequenceClassification
  init_args:
    backbone:
      class_path: aido_dna_7b
      init_args:
        use_peft: true
        lora_r: 16
        lora_alpha: 32
        lora_dropout: 0.1  
    adapter:
      class_path: modelgenerator.adapters.MLPPoolAdapter
      init_args:
        pooling: mean_pooling
        hidden_sizes: 
        - 512
        - 256
        bias: true
        dropout: 0.1
        dropout_in_middle: false
    optimizer:
      class_path: torch.optim.AdamW
      init_args:
        lr: 1e-4
    lr_scheduler:
      class_path: torch.optim.lr_scheduler.StepLR
      init_args:
        step_size: 1
        gamma: 0.1
data:
  ...
trainer:
  ...
```

> Note: Adapters and Backbones are typed as [`Callables`](https://jsonargparse.readthedocs.io/en/stable/index.html#callable-type), since some args are reserved to be automatically configured within the task.
As a general rule, positional arguments are reserved while keyword arguments are free to use.
For example, the backbone, adapter, optimizer, and lr_scheduler can be configured as

## Extract

::: modelgenerator.tasks.Embed

::: modelgenerator.tasks.Inference

## Adapt

::: modelgenerator.tasks.MLM

::: modelgenerator.tasks.ConditionalMLM

## Predict

::: modelgenerator.tasks.SequenceClassification

::: modelgenerator.tasks.TokenClassification

::: modelgenerator.tasks.PairwiseTokenClassification

::: modelgenerator.tasks.MMSequenceRegression

## Generate

::: modelgenerator.tasks.Diffusion

::: modelgenerator.tasks.ConditionalDiffusion

## Zero-Shot

::: modelgenerator.tasks.ZeroshotPredictionDiff

::: modelgenerator.tasks.ZeroshotPredictionDistance