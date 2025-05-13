# Adapters

Adapters work with [`Backbones`](../backbones) and [`Tasks`](../tasks) to adapt pretrained models to new objectives.
They are specified with the `--model.adapter` argument in the CLI or in the `model.adapter` section of a configuration file.

Adapters are the focal point for architecture design on top of backbones, and can be swapped with other adapters of the same type to benchmark different architectures.

This reference overviews the available no-code adapters.
If you would like to develop new adapters, see [Experiment Design](../../experiment_design).

```yaml
# Example Adapter Configuration
model:
  class_path: modelgenerator.tasks.SequenceRegression
  init_args:
    adapter:
      class_path: modelgenerator.adapters.MLPPoolAdapter
      init_args:
        pooling: mean_pooling
        hidden_sizes:
        - 512
        bias: true
        dropout: 0.1
        dropout_in_middle: false
data:
  ...
trainer:
  ...
```

## Sequence Adapters

These adapters make a single prediction for the entire input.

::: modelgenerator.adapters.MLPAdapter

::: modelgenerator.adapters.LinearCLSAdapter

::: modelgenerator.adapters.LinearMeanPoolAdapter

::: modelgenerator.adapters.LinearMaxPoolAdapter

::: modelgenerator.adapters.LinearTransformerAdapter

::: modelgenerator.adapters.ResNet2DAdapter

::: modelgenerator.adapters.ResNet1DAdapter

## Token Adapters

These adapters make one prediction per token.

::: modelgenerator.adapters.LinearAdapter

::: modelgenerator.adapters.MLPAdapter

::: modelgenerator.adapters.MLPAdapterWithoutOutConcat

## Conditional Generation Adapters

These adapters are used for conditional generation tasks.

::: modelgenerator.adapters.ConditionalLMAdapter

## Fusion Adapters

These adapters are used for multi-modal fusion to combine multiple backbones.

::: modelgenerator.adapters.MMFusionSeqAdapter

::: modelgenerator.adapters.MMFusionTokenAdapter
