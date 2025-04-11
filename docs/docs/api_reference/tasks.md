# Tasks

> Note: Adapters and Backbones are typed as [`Callables`](https://jsonargparse.readthedocs.io/en/stable/index.html#callable-type), since some args are reserved to be automatically configured within the task.
As a general rule, positional arguments are reserved while keyword arguments are free to use.
For example, the backbone, adapter, optimizer, and lr_scheduler can be configured as

```yaml
# My SequenceClassification task
model:
  class_path: SequenceClassification
  init_args:
    backbone:
      class_path: aido_dna_dummy
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
```

::: modelgenerator.tasks.SequenceClassification
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false

::: modelgenerator.tasks.TokenClassification
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false

::: modelgenerator.tasks.PairwiseTokenClassification
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false

::: modelgenerator.tasks.Diffusion
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false

::: modelgenerator.tasks.ConditionalDiffusion
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false

::: modelgenerator.tasks.ConditionalMLM
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false

::: modelgenerator.tasks.Embed
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false

::: modelgenerator.tasks.Inference
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false

::: modelgenerator.tasks.MLM
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false

::: modelgenerator.tasks.MMSequenceRegression
    handler: python
    options:
      members: false
      show_root_heading: true
      show_source: false