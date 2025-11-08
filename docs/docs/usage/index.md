# Basic Usage

AIDO.ModelGenerator orchestrates experiments with [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) to make runs modular, composable, and reproducible.

Training, validation, testing, and prediction are separated into independent CLI calls to `mgen fit/validate/test/predict`.
For researchers developing new backbones, heads, or tasks, a typical workflow might be
```
# My model development workflow
1. mgen fit --config config.yaml
2. mgen validate --config config.yaml --ckpt_path path/to/model.ckpt
# Make any config or code changes, repeat 1,2 until satisfied
3. mgen test --config config.yaml --ckpt_path path/to/model.ckpt
# Run inference for plotting, analysis, or deployment
4. mgen predict --config config.yaml --ckpt_path path/to/model.ckpt
```

For researchers or practicioners using pre-trained models, a typical workflow might be
```
# My inference and evaluation workflow
1. git clone https://huggingface.co/genbio-ai/<finetuned_model>
  # finetuned_model/
  # ├── config.yaml
  # ├── model.ckpt
2. Compile your data to predict or test on
3. Copy and override the `config.yaml` with your own data paths and splits
4. mgen test --config my_config.yaml --ckpt_path finetuned_model/model.ckpt
4. mgen predict --config my_config.yaml --ckpt_path finetuned_model/model.ckpt \
    --config configs/examples/save_predictions.yaml
```

For more details on designing and running experiments, see the [Experiment Design](../experiment_design/index.md) pocket guide.

## Options

To explore what options are configurable, `mgen` provides a `--help` flag at all levels.

```
mgen --help
mgen <fit/validate/test/predict> --help
mgen fit --model.help <Task>
# e.g. mgen fit --model.help ConditionalDiffusion
mgen fit --data.help <Dataset>
# e.g. mgen fit --data.help PromoterExpressionRegression
mgen fit --model.help <Task> --model.<arg>.help <arg_object>
# e.g. mgen fit --model.help ConditionalDiffusion --model.backbone.help aido_dna_dummy
```

## Using Configs

### Config-CLI duality
For reproducibility and fine-grained control, all CLI calls can be organized into a `config.yaml` file.
The command
```
mgen fit --model ConditionalDiffusion --model.backbone aido_dna_dummy \
  --data ConditionalDiffusion --data.path "genbio-ai/100m-random-promoters"
```

is equivalent to
`mgen fit --config my_config.yaml` with

```
# my_config.yaml
model:
  class_path: ConditionalDiffusion
  backbone: aido_dna_dummy
data:
  class_path: ConditionalDiffusion
  init_args:
    path: "genbio-ai/100m-random-promoters"
```

### Composability

Configs are also composable and allow multiple `--config` flags to be passed.
Runs always use only the LAST value for each argument.
Combining these two configs

```
# my_config.yaml
model:
  class_path: ConditionalDiffusion
  backbone: aido_dna_dummy
  adapter:
    class_path: modelgenerator.adapters.ConditionalLMAdapter
    init_args:
        n_head: 6
        dim_feedforward: int = 1024,
        num_layers: int = 2,

# my_new_config.yaml
model:
  class_path: ConditionalDiffusion
  backbone: aido_dna_7b
  adapter:
    class_path: modelgenerator.adapters.ConditionalLMAdapter
    init_args:
        num_layers: int = 8,
```
as `mgen fit --config my_config.yaml --config my_new_config.yaml` results in

```
model:
  class_path: ConditionalDiffusion
  backbone: aido_dna_7b
  adapter:
    class_path: modelgenerator.adapters.ConditionalLMAdapter
    init_args:
        n_head: 6
        dim_feedforward: int = 1024,
        num_layers: int = 8,
```

We provide some useful tools in `configs/examples` for logging, development, LoRA finetuning, and prediction writing that can be composed with your own configs.

### Reproducibility

The full configuration including all defaults and user-specified arguments will always be saved for each run.
This file changes location depending on logger, but will be in `logs/lightning_logs/your-experiment/config.yaml` by default, or if using wandb `logs/config.yaml`.
Even if AIDO.ModelGenerator defaults change, simply using `mgen fit --config your/logged/config.yaml` will always reproduce the experiment.

### Example

Below is a full example `config.yaml` saved from the simple command
```
mgen fit --model SequenceRegression --data PromoterExpressionRegression
```
See the pre-packaged experiments in the `experiments` directory for more examples.

```yaml
# lightning.pytorch==2.4.0
seed_everything: 0
model:
  class_path: modelgenerator.tasks.SequenceRegression
  init_args:
    backbone:
      class_path: modelgenerator.backbones.aido_dna_dummy
      init_args:
        from_scratch: false
        max_length: null
        use_peft: false
        frozen: false
        save_peft_only: true
        lora_r: 16
        lora_alpha: 32
        lora_dropout: 0.1
        lora_target_modules:
        - query
        - value
        config_overwrites: null
        model_init_args: null
    adapter:
      class_path: modelgenerator.adapters.LinearCLSAdapter
    num_outputs: 1
    optimizer:
      class_path: torch.optim.AdamW
      init_args:
        lr: 0.001
        betas:
        - 0.9
        - 0.999
        eps: 1.0e-08
        weight_decay: 0.01
        amsgrad: false
        maximize: false
        foreach: null
        capturable: false
        differentiable: false
        fused: null
    lr_scheduler: null
    use_legacy_adapter: false
    strict_loading: true
    reset_optimizer_states: false
data:
  class_path: modelgenerator.data.PromoterExpressionRegression
  init_args:
    path: genbio-ai/100m-random-promoters
    normalize: true
    config_name: null
    train_split_name: train
    test_split_name: test
    valid_split_name: null
    train_split_files: null
    test_split_files: null
    valid_split_files: null
    test_split_size: 0.2
    batch_size: 128
    shuffle: true
    sampler: null
    num_workers: 0
    pin_memory: true
    persistent_workers: false
    cv_num_folds: 1
    cv_test_fold_id: 0
    cv_enable_val_fold: true
    cv_fold_id_col: null
    cv_val_offset: 1
trainer:
  accelerator: auto
  strategy:
    class_path: lightning.pytorch.strategies.DDPStrategy
    init_args:
      accelerator: null
      parallel_devices: null
      cluster_environment: null
      checkpoint_io: null
      precision_plugin: null
      ddp_comm_state: null
      ddp_comm_hook: null
      ddp_comm_wrapper: null
      model_averaging_period: null
      process_group_backend: null
      timeout: 0:30:00
      start_method: popen
      output_device: null
      dim: 0
      broadcast_buffers: true
      process_group: null
      bucket_cap_mb: null
      find_unused_parameters: false
      check_reduction: false
      gradient_as_bucket_view: false
      static_graph: false
      delay_all_reduce_named_params: null
      param_to_hook_all_reduce: null
      mixed_precision: null
      device_mesh: null
  devices: auto
  num_nodes: 1
  precision: 32
  logger: null
  callbacks:
  - class_path: lightning.pytorch.callbacks.LearningRateMonitor
    init_args:
      logging_interval: step
      log_momentum: false
      log_weight_decay: false
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      dirpath: null
      filename: best_val:{step}-{val_loss:.3f}-{train_loss:.3f}
      monitor: val_loss
      verbose: false
      save_last: null
      save_top_k: 1
      save_weights_only: false
      mode: min
      auto_insert_metric_name: true
      every_n_train_steps: null
      train_time_interval: null
      every_n_epochs: null
      save_on_train_epoch_end: null
      enable_version_counter: true
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      dirpath: null
      filename: best_train:{step}-{val_loss:.3f}-train:{train_loss:.3f}
      monitor: train_loss
      verbose: false
      save_last: null
      save_top_k: 1
      save_weights_only: false
      mode: min
      auto_insert_metric_name: true
      every_n_train_steps: null
      train_time_interval: null
      every_n_epochs: null
      save_on_train_epoch_end: null
      enable_version_counter: true
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      dirpath: null
      filename: last:{step}-{val_loss:.3f}-{train_loss:.3f}
      monitor: null
      verbose: false
      save_last: null
      save_top_k: 1
      save_weights_only: false
      mode: min
      auto_insert_metric_name: true
      every_n_train_steps: null
      train_time_interval: null
      every_n_epochs: null
      save_on_train_epoch_end: null
      enable_version_counter: true
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      dirpath: null
      filename: step:{step}-{val_loss:.3f}-{train_loss:.3f}
      monitor: null
      verbose: false
      save_last: null
      save_top_k: -1
      save_weights_only: false
      mode: min
      auto_insert_metric_name: true
      every_n_train_steps: 1000
      train_time_interval: null
      every_n_epochs: null
      save_on_train_epoch_end: null
      enable_version_counter: true
  fast_dev_run: false
  max_epochs: -1
  min_epochs: null
  max_steps: -1
  min_steps: null
  max_time: null
  limit_train_batches: null
  limit_val_batches: null
  limit_test_batches: null
  limit_predict_batches: null
  overfit_batches: 0.0
  val_check_interval: null
  check_val_every_n_epoch: 1
  num_sanity_val_steps: null
  log_every_n_steps: 50
  enable_checkpointing: null
  enable_progress_bar: null
  enable_model_summary: null
  accumulate_grad_batches: 1
  gradient_clip_val: 1
  gradient_clip_algorithm: null
  deterministic: null
  benchmark: null
  inference_mode: true
  use_distributed_sampler: true
  profiler:
    class_path: lightning.pytorch.profilers.PyTorchProfiler
    init_args:
      dirpath: null
      filename: null
      group_by_input_shapes: false
      emit_nvtx: false
      export_to_chrome: true
      row_limit: 20
      sort_by_key: null
      record_module_names: true
      table_kwargs: null
      record_shapes: false
    dict_kwargs:
      profile_memory: true
  detect_anomaly: false
  barebones: false
  plugins: null
  sync_batchnorm: false
  reload_dataloaders_every_n_epochs: 0
  default_root_dir: logs
ckpt_path: null
```
