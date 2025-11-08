# Overriding Data Files

Applying a task or pre-trained model to train, evaluate, or predict with your data (locally or from Hugging Face) is straightforward.
There are two requirements:

1. You must know which [Task](../api_reference/tasks.md) and [DataModule](../api_reference/data.md) you are using.
If you are using a finetuned model checkpoint, find this in the associated config.yaml.
2. Your data must be in a format that the Hugging Face `load_dataset` can read (e.g. tsv, csv, json, etc).

If these are satisified, congrats! You can immediately use your data with AIDO.ModelGenerator by overriding the data paths and files in the config.

## Overriding Data Paths

Suppose a model was trained with data from Hugging Face but now we want to predict or retrain with a local dataset.
```
my_dataset/
  # Columns: `id`, `dna_sequence`, `expression`
  my_train.tsv
  my_test.tsv

```
The config.yaml `data` section might look like this:
```yaml
data:
  class_path: PromoterExpressionRegression
  init_args:
    path: genbio-ai/100m-random-promoters
    train_split_files: train.tsv
    test_split_files: test.tsv
    x_col: sequence
    y_col: label
    rename_cols:
      sequence: sequences
    val_split_size: 0.1
```
To re-run with your data, just override the necessary args
```yaml
data:
  class_path: PromoterExpressionRegression
  init_args:
    path: my_dataset
    train_split_files: my_train.tsv
    test_split_files: my_test.tsv
    x_col: dna_sequence
    y_col: expression
    rename_cols:
      dna_sequences: sequences
    val_split_size: 0.1
```

If some of the overrides aren't immediately obvious from the config, you can go to the [Data API Reference](../api_reference/data.md) to find documentation for usage.
Many datasets are [convenience datasets](https://github.com/genbio-ai/ModelGenerator/tree/main/modelgenerator/data/__init__.py), and aren't explicitly documented, so you'll need to look for their parent class in the [Data API Reference](../api_reference/data.md).

For example, a quick look at the codebase shows that `PromoterExpressionRegression` is a convenience subclass of `SequenceRegressionDataModule`.
The above overrides are equivalent to
```yaml
data:
  class_path: SequenceRegressionDataModule
  init_args:
    path: my_dataset
    train_split_files: my_train.tsv
    test_split_files: my_test.tsv
    x_col: dna_sequence
    y_col: expression
    rename_cols:
      dna_sequence: sequences
    val_split_size: 0.1
```
