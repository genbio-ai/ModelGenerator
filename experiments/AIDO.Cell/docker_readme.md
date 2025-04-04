# Data setup

Data for cell classification tasks can be found in [cell-downstream-tasks](https://huggingface.co/datasets/genbio-ai/cell-downstream-tasks) on Hugging Face.

To download all cell downstream tasks:
```
cd /path/to/ModelGenerator/modelgenerator
git clone git@hf.co:datasets/genbio-ai/cell-downstream-tasks
```

You should only need to do this once.

# Building the Docker image

```bash
cd /path/to/ModelGenerator
docker build -t finetune -f Dockerfile .
```

You should only need to do this once. 

# Hugging Face authentication

If you need to access private or gated models/data:

```bash
huggingface-cli login
```

# Fine-tuning a model

```bash
cd /path/to/ModelGenerator
docker run --rm --runtime=nvidia \
-v /home/user/ModelGenerator/configs:/workspace/configs \
-v /home/user/mgen_data:/workspace/mgen_data \
-v /home/user/ModelGenerator/modelgenerator:/workspace/modelgenerator \
-v /home/user/ModelGenerator/experiments:/workspace/experiments \
-v /home/user/.cache/huggingface:/root/.cache/huggingface \
finetune bash -c "mgen fit --config experiments/AIDO.Cell/cell_type_classification.yaml"
```

# Evaluating a checkpoint

```bash
cd /path/to/ModelGenerator
docker run --rm --runtime=nvidia \
-v /home/user/ModelGenerator/configs:/workspace/configs \
-v /home/user/mgen_data:/workspace/mgen_data \
-v /home/user/ModelGenerator/modelgenerator:/workspace/modelgenerator \
-v /home/user/ModelGenerator/experiments:/workspace/experiments \
-v /home/user/.cache/huggingface:/root/.cache/huggingface \
finetune bash -c "mgen test --config experiments/AIDO.Cell/cell_type_classification.yaml --ckpt_path /workspace/lightning_logs/version_X/checkpoints/my.ckpt"
```
