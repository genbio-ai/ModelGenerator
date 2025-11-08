# Embedding Caching (Experimental)

AIDO.ModelGenerator provides seamless file-based persistent embedding caching for all freezable backbones (e.g. backbones with the option `frozen=True`). This feature aims to boost training speed and reduce overall resource consumption by skipping backbone forwarding and redundant data loading.

## Create and resume from cache

Embedding caching is enabled by setting `--model.backbone.enable_cache true`. It works for all mgen subcommands including fit, validate, test and predict.

### Examples
**Train a model and save cache at the same time**
```bash
mgen fit --config my_config.yaml --model.backbone.enable_cache true --model.backbone.file_cache_dir my/cache/folder
```
As training progresses, cached backbone output will be saved to disk and automatically used in future steps. For example, if your first epoch iterates through all the training data, cached embeddings will be utilized starting from the second epoch automatically.

**Resume training from an existing cache**
```bash
mgen fit --config my_config.yaml --model.backbone.enable_cache true --model.backbone.file_cache_dir my/cache/folder
```
No change to the command is required, just make sure `--model.backbone.file_cache_dir` points to the right folder. Cached embedding will be used from the first step.

**Create cache without training the model**
```bash
mgen predict --config my_config.yaml --model.backbone.enable_cache true --model.backbone.file_cache_dir my/cache/folder
```
The best practice in this case is to use the `Embed` task, which is minimal and contains the backbone only.
