# Saving Outputs

AIDO.ModelGenerator provides a unified and hardware-adaptive interface for inference, embedding, and prediction with pre-trained models.

This page covers how to use AIDO.ModelGenerator to get embeddings and predictions from pre-trained backbones as well as finetuned models, and how to save and manage outputs for downstream analysis.

## Pre-trained Backbones

Backbones in AIDO.ModelGenerator are pre-trained foundation models.

A full list of available backbones is in the [Backbone API reference](../api_reference/backbones.md).
For each data modality, we suggest using

- `aido_dna_7b` for DNA sequences
- `aido_protein_16b` for protein sequences
- `aido_rna_1b600m` for RNA sequences
- `aido_cell_650m` for gene expression
- `aido_protein2structoken_16b` for translating protein sequence to structure tokens
- `aido_dna_dummy` and `aido_protein_dummy` for debugging
- `dna_onehot` and `protein_onehot` for non-FM baselines

## Backbone Embedding and Inference

To get embeddings, use `mgen predict` with the `Embed` task.

> Note: Predictions will always be taken from the test set.
> To get predictions from another dataset, set it as the test set using the `--data.test_split_files`

> Note: Distributed inference with DDP is enabled by default.
> Predictions need post-processing to be compiled into a single output.
> See below for details on distributed inference.

For example, to get embeddings from the `dummy` model on a small number of sequences in the `genbio-ai/100m-random-promoters` dataset and save to a `predictions` directory, use the following command:
```
# mgen predict --config config.yaml
# config.yaml:
model:
  class_path: Embed
  init_args:
    backbone: aido_dna_dummy
data:
  class_path: SequencesDataModule
  init_args:
    path: genbio-ai/100m-random-promoters
    x_col: sequence
    id_col: sequence  # No real ID in this dataset, so just use input sequence
    test_split_size: 0.0001
trainer:
  callbacks:
  - class_path: modelgenerator.callbacks.PredictionWriter
    dict_kwargs:
      output_dir: predictions
      filetype: pt
```

To get token probabilities, use `mgen predict` with the `Inference` task.
```
# mgen predict --config config.yaml
# config.yaml:
model:
  class_path: Inference
  init_args:
    backbone: aido_dna_dummy
data:
  class_path: SequencesDataModule
  init_args:
    path: genbio-ai/100m-random-promoters
    x_col: sequence
    id_col: sequence  # No real ID in this dataset, so just use input sequence
    test_split_size: 0.0001
trainer:
  callbacks:
  - class_path: modelgenerator.callbacks.PredictionWriter
    dict_kwargs:
      output_dir: predictions
      filetype: pt
```

## Finetuned Models

Finetuned model weights and configs from studies using AIDO.ModelGenerator are available for download on [Hugging Face](https://huggingface.co/genbio-ai).

To get predictions from a finetuned model, use `mgen predict` with the model's config file and checkpoint.
```
# Download the model and config from Hugging Face
# or use a local config.yaml and model.ckpt
mgen predict --config config.yaml --ckpt_path model.ckpt \
    --config configs/examples/save_predictions.yaml
```

Predicting, testing, or training on new data is also straightforward, and in most cases only requires matching the format of the original dataset and overriding filepaths.
See [Data Experiment Design](../experiment_design/data.md) for more details.

## Distributed Inference

Models and datasets are often too large to fit in memory on a single device.

AIDO.ModelGenerator supports distributed training and inference on multiple devices by sharding models and data with [FSDP](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html).
For example, to split `aido_protein_16b` across multiple nodes and multiple GPUs, add the following to your config:
```
trainer:
  num_nodes: X  # 1 by default, but not automatic. Must be set correctly for multi-node.
  devices: auto
  strategy:
    class_path: lightning.pytorch.strategies.FSDPStrategy
    init_args:
      sharding_strategy: FULL_SHARD
      auto_wrap_policy: modelgenerator.distributed.fsdp.wrap.AutoWrapPolicy
```

The `auto_wrap_policy` is necessary to shard the model in FSDP.


By default, PredictionWriter will save separate files for each batch and each device.
For customization options, see the [Callbacks API reference](../api_reference/callbacks.md).

We recommend using batch-level writing in most cases to avoid out-of-memory issues, and compiling and filtering predictions using a simple post-processing script.
```python
import torch
import os

# Load all predictions
predictions = []
for file in os.listdir("predictions"):
    if file.endswith(".pt"):
        prediction_device_batch = torch.load(os.path.join("predictions", file))
        prediction_clean = # Do any key filtering or transformations necessary here
        predictions.append(prediction_clean)

# Combine, convert, make a DataFrame, etc, and save for the next pipeline step.
```
