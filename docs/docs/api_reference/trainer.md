# Trainer

AIDO.ModelGenerator uses the LightningCLI for configuring runs with the PyTorch Lightning Trainer.
The entrypoint for the CLI is `mgen`, which can be used with the `fit`, `test`, `validate`, and `predict` commands and the `--model`, `--data`, and `--trainer` arguments and their sub-arguments.
```bash
mgen fit --model ConditionalDiffusion --model.backbone aido_dna_300m \
  --data ConditionalDiffusionDataModule --data.path "genbio-ai/100m-random-promoters" \
  --trainer.max_epochs 1 --trainer.accelerator auto --trainer.devices auto
```

For detailed information about the LightningCLI, see the [LightningCLI documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).

```yaml
# Example Trainer Configuration
trainer:
  accelerator: auto
  strategy: lightning.pytorch.strategies.DDPStrategy
  devices: auto
  num_nodes: 1
  precision: bf16-mixed
  logger: null
  callbacks:
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      filename: best_val:{step}-{val_loss:.3f}-{train_loss:.3f}
      monitor: val_loss
      save_top_k: 1
  fast_dev_run: false
  max_epochs: 100
  limit_val_batches: null
  val_check_interval: null
  check_val_every_n_epoch: 1
  log_every_n_steps: 50
  accumulate_grad_batches: 1
  gradient_clip_val: 1
  gradient_clip_algorithm: null
  detect_anomaly: false
  default_root_dir: logs
model:
  ...
data:
  ...
```
