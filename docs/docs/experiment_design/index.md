# Experiment Design

AIDO.ModelGenerator is designed to enable rapid and reproducible prototyping with four kinds of experiments in mind:

1. Applying pre-trained foundation models to new data
2. Developing new finetuning and inference tasks for foundation models
3. Benchmarking foundation models and creating leaderboards
4. Testing new architectures for finetuning performance

while also scaling with hardware and integrating with larger data pipelines or research workflows.

This section is a pocket guide on developing each of these types of experiments, outlining key interfaces and the minimal code required to get new experiments, models, or data up and running.

## Experiment Types

AIDO.ModelGenerator interfaces hide boilerplate and standardize training, evaluation, and prediction to enable a few common development goals.
If you want to

3. Use a new dataset for finetuning or inference.
    1. [Load your data](datafiles.md): If you want to apply an existing model or task to train, evaluate, or predict with your data (locally or from Hugging Face), this usually requires *no code*.
    2. [Or add a dataset](data.md): If you have unusual data types, require more than simple loading, or want to develop a new loading pattern for a new task, implement a `DataInterface`, **2** methods usually <10 lines.
1. Benchmark foundation models against each other and create leaderboards for your use-case.
    1. [Add a backbone](backbones.md): Implement a `BackboneInterface`, mostly one-liners.
2. Develop new finetuning or inference tasks that make use of foundation models (e.g. inverse folding, multi-modal fusion, diffusion).
    1. [Add a task](tasks.md): Implement a `TaskInterface`, **5** methods usually <10 lines each
3. Test new architectures for finetuning performance.
    2. [Add an adapter](tasks/#adding-adapters): Implement a `nn.Module` for use with a finetuning task, **2** methods.

## Codebase Structure

AIDO.ModelGenerator is built on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/levels/core_skills.html) for training and testing, [Huggingface](https://huggingface.co/docs/datasets/index) for model and data management, and [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for experiment configuration and organization.

Most development will focus on implementing a simple interface in `backbones` `adapters` `tasks` or `data`

```python
pyproject.toml  # Installation and packaging
Dockerfile      # Containerization
configs/        # Useful config files
experiments/    # Configs and scripts to reproduce manuscripts
docs/           # Website
modelgenerator/
	# Main codebase
*	backbones/  # Backbone models
*	adapters/   # Finetuning adapter heads
*	tasks/      # Model finetuning and usage
*	data/       # Data loading and formatting
    main.py     # mgen CLI entrypoint
    # Supplementary files
    callbacks.py      # Useful callbacks for saving inferences, etc
    lr_schedulers.py  # Custom LR schedulers
    metrics.py        # Custom metrics
    # Task-specific submodules
    rna_ss/                 # RNA secondary structure prediction
    structure_tokenizer/    # Protein structure tokenization
    rna_inv_fold/           # RNA inverse folding
    prot_inv_fold/          # Protein inverse folding
```
