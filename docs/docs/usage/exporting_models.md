# Exporting Models

While AIDO.ModelGenerator is CLI-driven, models created with AIDO.ModelGenerator can also be loaded in Python scripts and exported to HuggingFace.

## Exporting and Loading with CLI

As an example of a finetuned model export, see some of the many checkpoints available on [Huggingface](https://huggingface.co/genbio-ai) for immediate inference.
The only requirements are the `config.yaml` and `<model>.ckpt` files to be run again from AIDO.ModelGenerator.

```
# Download from HF
git clone https://huggingface.co/genbio-ai/dummy-ckpt

# Evaluate
mgen test --config dummy-ckpt/config.yaml \
  --ckpt_path dummy-ckpt/best_val:step=742-val_loss=0.404-train_loss=0.464.ckpt 

# Predict
mgen predict --config dummy-ckpt/config.yaml \
  --ckpt_path dummy-ckpt/best_val:step=742-val_loss=0.404-train_loss=0.464.ckpt \
  --config configs/examples/save_predictions.yaml
```

## Exporting and Loading in Python

Model checkpoints can also be used in Python scripts, notebooks, or other codebases with PyTorch Lightning.

```
# my_notebook.ipynb

########################################################################################
# Download the data
from huggingface_hub import snapshot_download
from pathlib import Path

my_models_path = Path.home().joinpath('my_models', 'dummy-ckpt')
my_models_path.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="genbio-ai/dummy-ckpt", local_dir=my_models_path)

########################################################################################
# Run the model
import torch

# Check the config.yaml for the model class
from modelgenerator.tasks import SequenceClassification

ckpt_path = my_models_path.joinpath('best_val:step=742-val_loss=0.404-train_loss=0.464.ckpt')
model = SequenceClassification.load_from_checkpoint(ckpt_path)

collated_batch = model.transform({"sequences": ["ACGT", "ACGT"]})
logits = model(collated_batch)
print(logits)
print(torch.argmax(logits, dim=-1))
########################################################################################
```

### Exporting and Loading in Hugging Face

Checkpoints can also be converted to HF format.

```
# Download from HF
git clone https://huggingface.co/genbio-ai/dummy-ckpt

# Convert to HF
mgen-hf convert --config conversion_config.yaml

# conversion_config.yaml:
task_class: modelgenerator.tasks.SequenceClassification
ckpt_path: dummy-ckpt/best_val:step=742-val_loss=0.404-train_loss=0.464.ckpt
dest_dir: dummy-ckpt-hf
push_to_hub: false
repo_id: genbio-ai/dummy-ckpt-hf
```

Then load the model in Hugging Face format
```
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("./dummy-ckpt-hf")
# Or if pushed to hub
# model = AutoModel.from_pretrained("genbio-ai/dummy-ckpt-hf", trust_remote_code=True)

collated_batch = model.genbio_model.transform({"sequences": ["ACGT", "ACGT"]})
logits = model(collated_batch)
print(logits)
print(torch.argmax(logits, dim=-1))
```