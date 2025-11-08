# Quick Start

## Installation
```
git clone https://github.com/genbio-ai/ModelGenerator.git
cd ModelGenerator
pip install -e .
```
Source installation is necessary to add new backbones, finetuning tasks, and data transformations, as well as use convenience configs and scripts. If you only need to run inference, reproduce published experiments, or finetune on new data, you can use
```
pip install modelgenerator
pip install git+https://github.com/genbio-ai/openfold.git@c4aa2fd0d920c06d3fd80b177284a22573528442
pip install git+https://github.com/NVIDIA/dllogger.git@0540a43971f4a8a16693a9de9de73c1072020769
```

## Quick Start
### Get embeddings from a pre-trained model
```
mgen predict --model Embed --model.backbone aido_dna_dummy \
  --data SequencesDataModule --data.path genbio-ai/100m-random-promoters \
  --data.x_col sequence --data.id_col sequence --data.test_split_size 0.0001 \
  --config configs/examples/save_predictions.yaml
```

### Get token probabilities from a pre-trained model
```
mgen predict --model Inference --model.backbone aido_dna_dummy \
  --data SequencesDataModule --data.path genbio-ai/100m-random-promoters \
  --data.x_col sequence --data.id_col sequence --data.test_split_size 0.0001 \
  --config configs/examples/save_predictions.yaml
```

### Finetune a model
```
mgen fit --model ConditionalDiffusion --model.backbone aido_dna_dummy \
  --data ConditionalDiffusionDataModule --data.path "genbio-ai/100m-random-promoters" \
  --data.x_col sequence --data.y_col label --data.rename_cols "{sequence: sequences}"
```

### Evaluate a model checkpoint
```
mgen test --model ConditionalDiffusion --model.backbone aido_dna_dummy \
  --data ConditionalDiffusionDataModule --data.path "genbio-ai/100m-random-promoters" \
  --data.x_col sequence --data.y_col label --data.rename_cols "{sequence: sequences}" \
  --ckpt_path logs/lightning_logs/version_X/checkpoints/<your_model>.ckpt
```

### Save predictions
```
mgen predict --model ConditionalDiffusion --model.backbone aido_dna_dummy \
  --data ConditionalDiffusionDataModule --data.path "genbio-ai/100m-random-promoters" \
  --data.x_col sequence --data.y_col label --data.rename_cols "{sequence: sequences}" \
  --ckpt_path logs/lightning_logs/version_X/checkpoints/<your_model>.ckpt \
  --config configs/examples/save_predictions.yaml
```

## Configify your experiment
This command
```
mgen fit --model ConditionalDiffusion --model.backbone aido_dna_dummy \
  --data ConditionalDiffusionDataModule --data.path "genbio-ai/100m-random-promoters" \
  --data.x_col sequence --data.y_col label --data.rename_cols "{sequence: sequences}"
```

is equivalent to
`mgen fit --config my_config.yaml` with

```
# my_config.yaml
model:
  class_path: ConditionalDiffusion
  init_args:
    backbone: aido_dna_dummy
data:
  class_path: ConditionalDiffusionDataModule
  init_args:
    path: "genbio-ai/100m-random-promoters"
    x_col: sequence
    y_col: label
    rename_cols:
      sequence: sequences
```

## Use composable configs to customize workflows
```
mgen fit --model SequenceRegression --data PromoterExpressionRegression \
  --config configs/defaults.yaml \
  --config configs/examples/lora_backbone.yaml \
  --config configs/examples/wandb.yaml
```

We provide some useful examples in `configs/examples`.
Configs use the LAST value for each attribute.
Check the full configuration logged with each experiment in `logs/lightning_logs/your-experiment/config.yaml`, or if using wandb `logs/config.yaml`.

## Use LoRA for parameter-efficient finetuning
This also avoids saving the full model, only the LoRA weights are saved.
```
mgen fit --data PromoterExpressionRegression \
  --model SequenceRegression --model.backbone.use_peft true \
  --model.backbone.lora_r 16 \
  --model.backbone.lora_alpha 32 \
  --model.backbone.lora_dropout 0.1
```

## Use continued pretraining for finetuning domain adaptation
First run pretraining objective on finetuning data
```
# https://arxiv.org/pdf/2310.02980
mgen fit --model MLM --model.backbone aido_dna_dummy \
  --data MLMDataModule --data.path leannmlindsey/GUE \
  --data.x_col sequence --data.y_col label --data.rename_cols "{sequence: sequences}" \
  --data.config_name prom_core_notata
```

Then finetune using the adapted model
```
mgen fit --model SequenceClassification --model.strict_loading false \
  --data SequenceClassificationDataModule --data.path leannmlindsey/GUE \
  --data.x_col sequence --data.y_col label --data.rename_cols "{sequence: sequences}" \
  --data.config_name prom_core_notata \
  --ckpt_path logs/lightning_logs/version_X/checkpoints/<your_adapted_model>.ckpt
```
Make sure to turn off `strict_loading` to replace the adapter!

## Use the head/adapter/decoder that comes with the backbone
```
mgen fit --model SequenceClassification --data GUEClassification \
  --model.use_legacy_adapter true
```
