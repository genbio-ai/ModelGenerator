import pytest
import torch
from modelgenerator.adapters.adapters import (
    MLPPoolAdapter,
    LinearCLSAdapter,
    LinearMeanPoolAdapter,
    LinearMaxPoolAdapter,
    LinearTransformerAdapter,
    ResNet1DAdapter,
    ResNet2DAdapter,
    MLPAdapter,
    MLPAdapterWithoutOutConcat,
    LinearAdapter,
)


@pytest.mark.parametrize(
    "adapter_class,adapter_args,input_shape,attention_mask_shape,expected_output_shape",
    [
        (
            MLPPoolAdapter,
            {"in_features": 16, "out_features": 8, "pooling": "mean_pooling"},
            (4, 10, 16),
            (4, 10),
            (4, 8),
        ),
        (
            MLPPoolAdapter,
            {"in_features": 16, "out_features": 8, "pooling": "cls_pooling"},
            (4, 10, 16),
            (4, 10),
            (4, 8),
        ),
        (
            LinearCLSAdapter,
            {"in_features": 16, "out_features": 8},
            (4, 10, 16),
            (4, 10),
            (4, 8),
        ),
        (
            LinearMeanPoolAdapter,
            {"in_features": 16, "out_features": 8},
            (4, 10, 16),
            (4, 10),
            (4, 8),
        ),
        (
            LinearMaxPoolAdapter,
            {"in_features": 16, "out_features": 8},
            (4, 10, 16),
            (4, 10),
            (4, 8),
        ),
        (
            LinearTransformerAdapter,
            {"embed_dim": 16, "out_features": 8},
            (4, 10, 16),
            (4, 10),
            (4, 8),
        ),
        (
            ResNet1DAdapter,
            {"input_channels": 16, "num_outputs": 1},
            (4, 10, 16),
            None,
            (4,),
        ),
        (ResNet2DAdapter, {"in_channels": 16}, (4, 10, 10, 16), None, (4, 10, 10)),
    ],
)
def test_sequence_adapter_output_shapes(
    adapter_class,
    adapter_args,
    input_shape,
    attention_mask_shape,
    expected_output_shape,
):
    adapter = adapter_class(**adapter_args)
    inputs = torch.randn(*input_shape)
    attention_mask = (
        torch.randint(0, 2, attention_mask_shape) if attention_mask_shape else None
    )
    output = adapter(inputs, attention_mask)
    assert output.shape == expected_output_shape
    # all adapters should be able to handle None attention masks
    output = adapter(inputs, None)
    assert output.shape == expected_output_shape


@pytest.mark.parametrize(
    "adapter_class,adapter_args,input_shape,attention_mask_shape,expected_output_shape",
    [
        (
            MLPAdapter,
            {"in_features": 16, "out_features": 8, "hidden_sizes": [32, 16]},
            (4, 10, 16),
            (4, 10),
            (4, 10, 8),
        ),
        (
            MLPAdapterWithoutOutConcat,
            {"in_features": 16, "out_features": 8, "hidden_sizes": [32, 16]},
            (4, 10, 16),
            (4, 10),
            (4, 10, 10, 8),
        ),
        (
            LinearAdapter,
            {"in_features": 16, "out_features": 8},
            (4, 10, 16),
            (4, 10),
            (4, 10, 8),
        ),
    ],
)
def test_token_adapter_output_shapes(
    adapter_class,
    adapter_args,
    input_shape,
    attention_mask_shape,
    expected_output_shape,
):
    adapter = adapter_class(**adapter_args)
    inputs = torch.randn(*input_shape)
    attention_mask = (
        torch.randint(0, 2, attention_mask_shape) if attention_mask_shape else None
    )
    output = adapter(inputs, attention_mask)
    assert output.shape == expected_output_shape
    # all adapters should be able to handle None attention masks
    output = adapter(inputs, None)
