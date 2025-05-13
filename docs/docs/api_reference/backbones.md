# Backbones

Backbones are pretrained foundation models.
They are specified with the `--model.backbone` argument in the CLI or in the `model.backbone` section of a configuration file.

AIDO.ModelGenerator wraps messy foundation models in a standardized interface, allowing them to be applied to finetuning and inference [tasks](../tasks) without any code, and even fused for multi-modal tasks.
Backbones are also interchangeable, making it simple to run benchmarks and create leaderboards so you can find the best model for your task.

Many backbones come with options for parameter-efficient finetuning (PEFT) methods, low-memory checkpointing, and small-scale debugging models to assist with developing on large-scale foundation models.

This reference overviews the available no-code backbones.
If you would like to integrate new backbones, see [Experiment Design](../../experiment_design).

```yaml
# Example Backbone Configuration
model:
  class_path: modelgenerator.tasks.SequenceRegression
  init_args:
    backbone:
      class_path: modelgenerator.backbones.aido_rna_1b600m_cds
      init_args:
        max_length: 1024
        use_peft: true
        save_peft_only: true
        lora_r: 32
        lora_alpha: 64
        lora_dropout: 0.1
        lora_target_modules:
        - query
        - value
        config_overwrites:
          hidden_dropout_prob: 0.1
          attention_probs_dropout_prob: 0.1
        model_init_args: null
data:
  ...
trainer:
  ...
```

## DNA

::: modelgenerator.backbones.aido_dna_7b

::: modelgenerator.backbones.aido_dna_300m

::: modelgenerator.backbones.enformer

::: modelgenerator.backbones.borzoi

::: modelgenerator.backbones.flashzoi

## RNA

::: modelgenerator.backbones.aido_rna_1b600m

::: modelgenerator.backbones.aido_rna_1b600m_cds

::: modelgenerator.backbones.aido_rna_650m

::: modelgenerator.backbones.aido_rna_650m_cds

::: modelgenerator.backbones.aido_rna_300m_mars

::: modelgenerator.backbones.aido_rna_25m_mars

::: modelgenerator.backbones.aido_rna_1m_mars

## Protein

::: modelgenerator.backbones.aido_protein_16b

::: modelgenerator.backbones.aido_protein_16b_v1

::: modelgenerator.backbones.esm2_15b

::: modelgenerator.backbones.esm2_3b

::: modelgenerator.backbones.esm2_650m

::: modelgenerator.backbones.esm2_150m

::: modelgenerator.backbones.esm2_35m

::: modelgenerator.backbones.esm2_8m

## Structure

::: modelgenerator.backbones.aido_protein2structoken_16b

::: modelgenerator.backbones.aido_protein_rag_16b

::: modelgenerator.backbones.aido_protein_rag_3b

## Cell

::: modelgenerator.backbones.aido_cell_100m

::: modelgenerator.backbones.aido_cell_10m

::: modelgenerator.backbones.aido_cell_3m

::: modelgenerator.backbones.scfoundation

::: modelgenerator.backbones.geneformer

## Tissue

::: modelgenerator.backbones.aido_tissue_3m

::: modelgenerator.backbones.aido_tissue_60m

## Integrations

::: modelgenerator.backbones.Huggingface

## Debug

::: modelgenerator.backbones.Onehot

::: modelgenerator.backbones.dna_onehot

::: modelgenerator.backbones.protein_onehot

::: modelgenerator.backbones.aido_dna_debug

::: modelgenerator.backbones.aido_protein_debug

::: modelgenerator.backbones.aido_dna_dummy

## Base Classes

::: modelgenerator.backbones.SequenceBackboneInterface

::: modelgenerator.backbones.HFSequenceBackbone

::: modelgenerator.backbones.GenBioBERT

::: modelgenerator.backbones.GenBioFM

::: modelgenerator.backbones.GenBioCellFoundation

::: modelgenerator.backbones.GenBioCellSpatialFoundation
