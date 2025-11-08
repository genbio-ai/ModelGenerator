import os
import torch
import pytest
from unittest.mock import Mock

from modelgenerator.backbones.base import (
    SequenceBackboneOutput,
    SequenceBackboneInterface,
    _BackboneCache,
    _LMDBStore,
    _IndexedStore,
)


def test_sequence_backbone_output_getitem():
    """Test SequenceBackboneOutput.__getitem__ method."""
    output = SequenceBackboneOutput(
        last_hidden_state=torch.randn(4, 8, 16),
        attention_mask=torch.ones(4, 8),
        hidden_states=[torch.randn(4, 8, 16), torch.randn(4, 8, 16)],
        special_tokens_mask=torch.zeros(4, 8),
    )

    # Test integer indexing
    indexed = output[0]
    assert indexed.last_hidden_state.shape == (8, 16)
    assert indexed.attention_mask.shape == (8,)
    assert len(indexed.hidden_states) == 2
    assert indexed.hidden_states[0].shape == (8, 16)
    assert indexed.special_tokens_mask.shape == (8,)

    # Test slice indexing
    sliced = output[1:3]
    assert sliced.last_hidden_state.shape == (2, 8, 16)
    assert sliced.attention_mask.shape == (2, 8)
    assert len(sliced.hidden_states) == 2
    assert sliced.hidden_states[0].shape == (2, 8, 16)
    assert sliced.special_tokens_mask.shape == (2, 8)


def test_sequence_backbone_output_getitem_with_none_fields():
    """Test __getitem__ when optional fields are None."""
    output = SequenceBackboneOutput(
        last_hidden_state=torch.randn(4, 8, 16),
        attention_mask=None,
        hidden_states=None,
        special_tokens_mask=None,
    )

    indexed = output[0]
    assert indexed.last_hidden_state.shape == (8, 16)
    assert indexed.attention_mask is None
    assert indexed.hidden_states is None
    assert indexed.special_tokens_mask is None


def test_sequence_backbone_output_concat():
    """Test SequenceBackboneOutput.concat method with padding."""
    # Create outputs with different sequence lengths
    output1 = SequenceBackboneOutput(
        last_hidden_state=torch.randn(2, 5, 16),
        attention_mask=torch.ones(2, 5),
        hidden_states=[torch.randn(2, 5, 16), torch.randn(2, 5, 16)],
        special_tokens_mask=torch.zeros(2, 5),
    )

    output2 = SequenceBackboneOutput(
        last_hidden_state=torch.randn(3, 8, 16),
        attention_mask=torch.ones(3, 8),
        hidden_states=[torch.randn(3, 8, 16), torch.randn(3, 8, 16)],
        special_tokens_mask=torch.zeros(3, 8),
    )

    concatenated = SequenceBackboneOutput.concat([output1, output2])

    # Check concatenated shapes - should be padded to max length (8)
    assert concatenated.last_hidden_state.shape == (5, 8, 16)  # (2+3, max_len, hidden_size)
    assert concatenated.attention_mask.shape == (5, 8)
    assert len(concatenated.hidden_states) == 2  # Same number of layers
    assert concatenated.hidden_states[0].shape == (5, 8, 16)
    assert concatenated.special_tokens_mask.shape == (5, 8)

    # Check that padding was applied correctly (first 2 sequences should have padding)
    assert concatenated.attention_mask[0, 5:].sum() == 0
    assert concatenated.attention_mask[1, 5:].sum() == 0
    assert concatenated.attention_mask[2:, :].sum() == 24


def test_sequence_backbone_output_concat_no_optional_fields():
    """Test concat when optional fields are None."""
    output1 = SequenceBackboneOutput(
        last_hidden_state=torch.randn(2, 5, 16),
        attention_mask=None,
        hidden_states=None,
        special_tokens_mask=None,
    )

    output2 = SequenceBackboneOutput(
        last_hidden_state=torch.randn(3, 7, 16),
        attention_mask=None,
        hidden_states=None,
        special_tokens_mask=None,
    )

    concatenated = SequenceBackboneOutput.concat([output1, output2])

    assert concatenated.last_hidden_state.shape == (5, 7, 16)  # Padded to max length
    assert concatenated.attention_mask is None
    assert concatenated.hidden_states is None
    assert concatenated.special_tokens_mask is None


@pytest.fixture
def mock_backbone():
    """Create a mock backbone module."""
    backbone = Mock(spec=SequenceBackboneInterface)

    # Mock the forward method to return SequenceBackboneOutput
    def mock_forward(**kwargs):
        batch_size = kwargs.get("input_ids", torch.tensor([[1, 2]])).shape[0]
        return SequenceBackboneOutput(
            last_hidden_state=torch.randn(batch_size, 5, 16),
            attention_mask=torch.ones(batch_size, 5),
            hidden_states=None,
            special_tokens_mask=None,
        )

    backbone.forward = mock_forward
    backbone.process_batch = Mock(
        return_value={"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
    )
    backbone.required_data_columns = Mock(return_value=["sequences"])

    # Add parameters method for requires_grad check
    backbone.parameters = Mock(return_value=[Mock(requires_grad=False)])

    return backbone


def test_backbone_cache_forward_first_call(mock_backbone, tmpdir):
    """Test forward pass when cache is empty (first call)."""
    cache = _BackboneCache(mock_backbone, file_cache_dir=str(tmpdir), overwrite_file_cache=True)

    result = cache.forward(uid=["id1", "id2"], input_ids=torch.tensor([[1, 2], [3, 4]]))

    assert isinstance(result, SequenceBackboneOutput)
    assert "id1" in cache._store
    assert "id2" in cache._store

    assert cache._store["id1"].last_hidden_state.shape[0] == 1
    assert cache._store["id2"].last_hidden_state.shape[0] == 1


def test_backbone_cache_forward_cached_call(mock_backbone, tmpdir):
    """Test forward pass when data is already cached."""
    # Mock the original forward to track calls
    original_forward = mock_backbone.forward
    mock_backbone.forward = Mock(side_effect=original_forward)
    cache = _BackboneCache(mock_backbone, file_cache_dir=str(tmpdir), overwrite_file_cache=True)

    cache.forward(uid=["id1", "id2"], input_ids=torch.tensor([[1, 2], [3, 4]]))
    result = cache.forward(uid=["id1", "id2"], input_ids=torch.tensor([[1, 2], [3, 4]]))

    mock_backbone.forward.assert_called_once()
    assert result.last_hidden_state.shape == (2, 5, 16)


def test_backbone_cache_forward_partial_cache_hit(mock_backbone, tmpdir):
    """Test forward pass with partial cache hit."""
    # Mock the original forward to track calls
    original_forward = mock_backbone.forward
    mock_backbone.forward = Mock(side_effect=original_forward)
    cache = _BackboneCache(mock_backbone, file_cache_dir=str(tmpdir), overwrite_file_cache=True)

    cache.forward(uid=["id1", "id2"], input_ids=torch.tensor([[1, 2], [3, 4]]))
    mock_backbone.forward.assert_called_once()
    result = cache.forward(uid=["id1", "id3"], input_ids=torch.tensor([[1, 2], [3, 4]]))
    assert mock_backbone.forward.call_count == 2  # Should call original forward for id3
    assert result.last_hidden_state.shape == (2, 5, 16)


def test_backbone_cache_process_batch_with_existing_uid(mock_backbone, tmpdir):
    """Test process_batch when uid already exists in batch."""
    cache = _BackboneCache(mock_backbone, file_cache_dir=str(tmpdir), overwrite_file_cache=True)
    mock_backbone.process_batch.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "uid": ["existing_id"],
    }

    batch = {"sequences": ["ACGT"], "uid": ["batch_id"]}
    device = torch.device("cpu")

    result = cache.process_batch(batch, device)

    # Should preserve existing uid from process_batch result
    assert result["uid"] == ["existing_id"]


def test_backbone_cache_clear_cache(mock_backbone, tmpdir):
    """Test cache clearing."""
    cache = _BackboneCache(mock_backbone, file_cache_dir=str(tmpdir), overwrite_file_cache=True)

    cache.forward(uid=["id1", "id2"], input_ids=torch.tensor([[1, 2], [3, 4]]))
    assert "id1" in cache._store
    assert "id2" in cache._store

    cache.clear()
    assert "id1" not in cache._store
    assert "id2" not in cache._store


def test_backbone_enable_cache(tmpdir):
    """Test backbone cache enabling."""
    backbone = SequenceBackboneInterface(
        enable_cache=False, file_cache_dir=str(tmpdir), overwrite_file_cache=True
    )
    assert not backbone.cache_enabled
    assert backbone.cache is None
    assert os.listdir(tmpdir) == []

    backbone.enable_cache()
    assert backbone.cache_enabled
    assert backbone.cache is not None

    # Check that methods are replaced with cached versions
    assert backbone.forward == backbone.cache.forward
    assert backbone.process_batch == backbone.cache.process_batch
    assert backbone.required_data_columns == backbone.cache.required_data_columns


def test_backbone_disable_cache(tmpdir):
    """Test backbone cache disabling."""
    backbone = SequenceBackboneInterface(
        enable_cache=True, file_cache_dir=str(tmpdir), overwrite_file_cache=True
    )
    assert backbone.cache_enabled

    backbone.disable_cache()
    assert not backbone.cache_enabled

    # Check that methods are restored to original versions
    assert backbone.forward == backbone.cache._orig_fwd
    assert backbone.process_batch == backbone.cache._orig_process_batch
    assert backbone.required_data_columns == backbone.cache._orig_required_data_columns


@pytest.mark.parametrize("store_cls", [_LMDBStore, _IndexedStore])
def test_cache_store_basic_operations(tmpdir, store_cls):
    """Test basic LMDB cache operations."""
    store = store_cls(str(tmpdir))

    # Test setting and getting
    store["key1"] = "value1"
    assert store["key1"] == "value1"

    # Test contains
    assert "key1" in store
    assert "nonexistent" not in store


@pytest.mark.parametrize("store_cls", [_LMDBStore, _IndexedStore])
def test_cache_store_write_buffer(tmpdir, store_cls):
    """Test write buffer functionality."""
    store = store_cls(str(tmpdir), write_buffer_size=2)

    # Fill buffer
    store["key1"] = SequenceBackboneOutput(last_hidden_state=torch.tensor([1]))
    store["key2"] = SequenceBackboneOutput(last_hidden_state=torch.tensor([2]))
    assert len(store.write_buffer) == 2

    store["key3"] = SequenceBackboneOutput(last_hidden_state=torch.tensor([3]))
    assert store.should_flush

    # Verify all values accessible after flush
    store.flush()
    assert len(store.write_buffer) == 0
    assert store["key1"].last_hidden_state == torch.tensor([1])
    assert store["key2"].last_hidden_state == torch.tensor([2])
    assert store["key3"].last_hidden_state == torch.tensor([3])


@pytest.mark.parametrize("store_cls", [_LMDBStore, _IndexedStore])
def test_cache_store_flush_and_persistence(tmpdir, store_cls):
    """Test flushing and data persistence."""
    store = store_cls(str(tmpdir))

    # Add data to write buffer
    store["key1"] = SequenceBackboneOutput(last_hidden_state=torch.tensor([1]))
    store.flush()

    # Create new cache instance - data should persist
    store = store_cls(str(tmpdir))
    assert store["key1"].last_hidden_state == torch.tensor([1])


@pytest.mark.parametrize("store_cls", [_LMDBStore, _IndexedStore])
def test_cache_store_clear(tmpdir, store_cls):
    """Test cache clearing."""
    store = store_cls(str(tmpdir))

    # Add data to both buffers and disk
    store["key1"] = SequenceBackboneOutput(last_hidden_state=torch.tensor([1]))
    store.flush()
    store["key2"] = SequenceBackboneOutput(last_hidden_state=torch.tensor([2]))  # In write buffer
    store.write_buffer["key3"] = SequenceBackboneOutput(last_hidden_state=torch.tensor([3]))

    store.clear()

    # All should be cleared
    assert "key1" not in store
    assert len(store.write_buffer) == 0
