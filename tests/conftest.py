import pytest
import torch
import torch.nn as nn
import warnings
from unittest.mock import MagicMock, patch
from modelgenerator.backbones.backbones import (
    GenBioBERT,
    GenBioFM,
    GenBioCellFoundation,
    GenBioCellSpatialFoundation,
    Enformer,
    Borzoi,
    ESM,
    Geneformer,
    LegacyAdapterType,
)


if torch.cuda.is_available():
    # Multi GPUs are causing tests to hang
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@pytest.fixture()
def flash_attn_available():
    """
    Check if flash attention is available.
    """
    try:
        import flash_attn  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def genbiobert_cls():
    class TinyModel(GenBioBERT):
        def __init__(self, *args, **kwargs):
            config_overwrites = {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "intermediate_size": 16,
                "max_position_embeddings": 128,
            }
            super().__init__(
                *args, from_scratch=True, config_overwrites=config_overwrites, **kwargs
            )

    return TinyModel


@pytest.fixture
def genbiobert(genbiobert_cls):
    return genbiobert_cls(None, None)


@pytest.fixture
def genbiofm_cls():
    class TinyModel(GenBioFM):
        def __init__(self, *args, **kwargs):
            config_overwrites = {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "intermediate_size": 16,
                "max_position_embeddings": 128,
                "num_experts": 1,
            }
            super().__init__(
                *args, from_scratch=True, config_overwrites=config_overwrites, **kwargs
            )

    return TinyModel


@pytest.fixture
def genbiofm(genbiofm_cls):
    return genbiofm_cls(None, None)


@pytest.fixture
def genbiocellfoundation(flash_attn_available):
    class TinyModel(GenBioCellFoundation):
        def __init__(self, *args, **kwargs):
            config_overwrites = {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "intermediate_size": 16,
                "max_position_embeddings": 128,
                "_use_flash_attention_2": True,
                "bf16": True,
            }
            super().__init__(
                *args, from_scratch=True, config_overwrites=config_overwrites, **kwargs
            )
    tiny_model = TinyModel(None, None)
    if not flash_attn_available:
        warnings.warn(
            "Flash attention is not available. Using a mocked encoder for CellFoundation."
        )
        tiny_model.encoder = MagicMock(spec=nn.Module)
        tiny_model.encoder.return_value = MagicMock(
            last_hidden_state=torch.randn(4, 10, 16),
            hidden_states=[torch.randn(4, 10, 16) for _ in range(2)],
        )
        tiny_model.encoder.config = MagicMock(**tiny_model.config_overwrites)
    return tiny_model


@pytest.fixture
def genbiocellspatialfoundation(flash_attn_available):
    class TinyModel(GenBioCellSpatialFoundation):
        def __init__(self, *args, **kwargs):
            config_overwrites = {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "intermediate_size": 16,
                "max_position_embeddings": 128,
                "_use_flash_attention_2": True,
                'bf16': True,
            }
            super().__init__(
                *args, from_scratch=True, config_overwrites=config_overwrites, **kwargs
            )
    tiny_model = TinyModel(None, None)
    if not flash_attn_available:
        warnings.warn(
            "Flash attention is not available. Using a mocked encoder for CellSpatialFoundation."
        )
        tiny_model.encoder = MagicMock(spec=nn.Module)
        tiny_model.encoder.return_value = MagicMock(
            last_hidden_state=torch.randn(4, 130, 16),
            hidden_states=[torch.randn(4, 130, 16) for _ in range(2)],
        )
        tiny_model.encoder.config = MagicMock(**tiny_model.config_overwrites)
    return tiny_model


@pytest.fixture
def enformer_cls():
    class TinyModel(Enformer):
        def __init__(self, *args, **kwargs):
            config_overwrites = {
                "dim": 12,
                "depth": 2,
                "heads": 1,
                "target_length": 2,
                "dim_divisible_by": 2,
                "num_downsamples": 4,
            }
            super().__init__(
                *args,
                from_scratch=True,
                config_overwrites=config_overwrites,
                max_length=128,
                **kwargs,
            )

    return TinyModel


@pytest.fixture
def enformer(enformer_cls):
    return enformer_cls(None, None)


@pytest.fixture
def borzoi_cls():
    class TinyModel(Borzoi):
        def __init__(self, *args, **kwargs):
            config_overwrites = {
                "dim": 12,
                "depth": 2,
                "heads": 1,
                "bins_to_return": 2,
                "dim_divisible_by": 2,
                "num_downsamples": 4,
            }
            super().__init__(
                *args,
                from_scratch=True,
                config_overwrites=config_overwrites,
                max_length=128,
                **kwargs,
            )

    return TinyModel

@pytest.fixture
def borzoi(borzoi_cls):
    return borzoi_cls(None, None)

@pytest.fixture
def esm():
    # TODO: Mocking encoder because this backbone does not support `from_scratch`
    mock_encoder = MagicMock(spec=nn.Module)
    mock_encoder.config = MagicMock(hidden_size=16, num_hidden_layers=2, vocab_size=33)
    mock_encoder.return_value = MagicMock(
        last_hidden_state=torch.randn(4, 10, 16),
        hidden_states=[torch.randn(4, 10, 16) for _ in range(2)],
    )
    with patch(
        "transformers.AutoModel.from_pretrained",
        return_value=(mock_encoder, {"missing_keys": []}),
    ):
        ESM.model_path = "facebook/esm2_t30_150M_UR50D"
        model = ESM(
            legacy_adapter_type=None,
            default_config=None,
            max_length=128,
        )
    return model


@pytest.fixture
def geneformer_cls():
    class TinyModel(Geneformer):
        def __init__(self, legacy_adapter_type=None, default_config=None):
            if legacy_adapter_type == LegacyAdapterType.MASKED_LM:
                config_overwrites = {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "intermediate_size": 8,
                    "max_position_embeddings": 5,
                }
                max_length = 5
            else:
                config_overwrites = {
                    "hidden_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "intermediate_size": 16,
                    "max_position_embeddings": 10,
                }
                max_length = 10

            super().__init__(
                legacy_adapter_type=legacy_adapter_type,
                default_config=default_config,
                from_scratch=True,
                max_length=max_length,
                config_overwrites=config_overwrites,
            )

    return TinyModel


@pytest.fixture
def geneformer(geneformer_cls):
    # simple (non-MLM) tiny Geneformer
    return geneformer_cls(None, None)


@pytest.fixture
def geneformer_mlm(geneformer_cls):
    # masked-LM tiny Geneformer
    return geneformer_cls(LegacyAdapterType.MASKED_LM, None)