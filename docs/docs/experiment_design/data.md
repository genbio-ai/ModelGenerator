# Adding Data Loaders

AIDO.ModelGenerator uses [Lightning DataModules](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) for dataset management and loading.
We also provide a few tools to make data management more convenient, and work with common file types out-of-the-box.

AIDO.ModelGenerator provides a `DataInterface` class that hides boilerplate, along with a `HFDatasetLoaderMixin` that combines Lightning DataModule structure and [HuggingFace Datasets](https://huggingface.co/docs/datasets) convenience together to quickly load data from HuggingFace or common file formats (e.g. tsv, csv, json, etc).
More convenient mixins and example usage are outlined below.

Many common tasks and data loaders are already implemented in AIDO.ModelGenerator, and only require setting new paths to run with new data.
See the [Data API Reference](../api_reference/data.md) for all types of available data modules.

::: modelgenerator.data.DataInterface
    handler: python
    options:
      filters:
        - "!^__"
      members:
        - setup
        - load_and_split_dataset
      show_root_heading: true
      show_source: true

## Useful Mixins

::: modelgenerator.data.HFDatasetLoaderMixin
    handler: python
    options:
      filters:
        - "!^__"
      show_root_heading: true
      show_source: true

::: modelgenerator.data.KFoldMixin
    handler: python
    options:
      filters:
        - "!^__"
      show_root_heading: true
      show_source: true

## Implementing a DataModule

To transform datasets for task-specific behaviors (e.g. masking for masked language modeling), use `torch.utils.data.Dataset` objects to implement the transformation.
Below is an example.

::: modelgenerator.data.MLMDataModule
    handler: python
    options:
      filters:
        - "!^__"
      members:
        - setup
      show_root_heading: true
      show_source: true

::: modelgenerator.data.MLMDataset
    handler: python
    options:
      filters:
        - "!^__"
      show_root_heading: true
      show_source: true
