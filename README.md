[![GenBio AI](docs/docs/assets/images/genbio_header.png)](https://genbio.ai/)

# [AIDO](https://github.com/genbio-ai/AIDO).ModelGenerator


[![License](https://img.shields.io/badge/license-GenBio_AI_Community_License-orange)](https://github.com/genbio-ai/ModelGenerator/blob/main/LICENSE)
![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
[![PyPI version](https://badge.fury.io/py/modelgenerator.svg)](https://badge.fury.io/py/modelgenerator)
![Maintenance](https://img.shields.io/maintenance/yes/2025?style=flat-square)
[![Downloads](https://pepy.tech/badge/modelgenerator)](https://pepy.tech/project/modelgenerator)
[![DOI](https://img.shields.io/badge/DOI-10.1101/2025.06.30.662437-brightgreen)](https://doi.org/10.1101/2025.06.30.662437)

AIDO.ModelGenerator is a software stack powering the development of an AI-driven Digital Organism by enabling researchers to adapt pretrained models and generate finetuned models for downstream tasks.
To read more about AIDO.ModelGenerator's integral role in building the world's first AI-driven Digital Organism, see [AIDO](https://github.com/genbio-ai/AIDO).

AIDO.ModelGenerator is open-sourced as an opinionated plug-and-play research framework for cross-disciplinary teams in ML & Bio.
It is designed to enable rapid and reproducible prototyping with four kinds of experiments in mind:

1. Applying pre-trained foundation models to new data
2. Developing new finetuning and inference tasks for foundation models
3. Benchmarking foundation models and creating leaderboards
4. Testing new architectures for finetuning performance

while also scaling with hardware and integrating with larger data pipelines or research workflows.

AIDO.ModelGenerator is built on PyTorch, HuggingFace, and Lightning, and works seamlessly with these ecosystems.

See the [AIDO.ModelGenerator documentation](https://genbio-ai.github.io/ModelGenerator) for installation, usage, tutorials, and API reference.

## Who uses ModelGenerator?

### 🧬 Biologists
* Intuitive one-command CLIs for in silico experiments
* Pre-trained model zoo
* Broad data compatibility
* Pipeline-oriented workflows

### 🤖 ML Researchers
* Reproducible-by-design experiments
* Architecture A/B testing
* Automatic hardware scaling
* Integration with PyTorch, Lightning, HuggingFace, and WandB

### ☕ Software Engineers
* Extensible and modular models, tasks, and data
* Strict typing and documentation
* Fail-fast interface design
* Continuous integration and testing

### 🤝 Everyone benefits from
* A collaborative hub and focal point for multidisciplinary work on experiments, models, software, and data
* Community-driven development
* Permissive license for academic and non-commercial use

## Projects using AIDO.ModelGenerator

- [Accurate and General DNA Representations Emerge from Genome Foundation Models at Scale](https://doi.org/10.1101/2024.12.01.625444)
- [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://doi.org/10.1101/2024.11.28.625345)
- [Mixture of Experts Enable Efficient and Effective Protein Understanding and Design](https://doi.org/10.1101/2024.11.29.625425)
- [Scaling Dense Representations for Single Cell with Transcriptome-Scale Context](https://doi.org/10.1101/2024.11.28.625303)
- [Balancing Locality and Reconstruction in Protein Structure Tokenizer](https://doi.org/10.1101/2024.12.02.626366)

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
