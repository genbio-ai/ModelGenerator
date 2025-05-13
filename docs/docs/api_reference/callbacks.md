# Callbacks

Callbacks can be used with the LightningCLI trainer to inject custom behavior into the training process.
Callbacks are configured in the `trainer` section of the YAML configuration file. 

We provide a few custom callbacks for common use cases, but many more are available in the Lightning ecosystem.
Check the [Trainer documentation](../trainer) for more details.

```yaml
# Example Callback Configuration
trainer:
  callbacks:
  - class_path: modelgenerator.callbacks.PredictionWriter
    dict_kwargs:
      output_dir: my_predictions
      filetype: tsv
      write_cols:
        - id
        - prediction
        - label
model:
  ...
data:
  ...
```

::: modelgenerator.callbacks.PredictionWriter

::: modelgenerator.callbacks.FTScheduler