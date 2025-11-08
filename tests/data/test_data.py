import pytest
from modelgenerator.data.data import *  # noqa: F403
from unittest.mock import patch
from datasets import Dataset


def test_any_dataset():
    # Create sample datasets
    seq_data = ["ACGT", "TGCA", "GCTA"]
    label_data = [0, 1, 0]

    # Create AnyDataset with multiple inputs
    dataset = AnyDataset(sequences=seq_data, labels=label_data)

    # Test length
    assert len(dataset) == 3

    # Test item retrieval
    item = dataset[0]
    assert item["sequences"] == "ACGT"
    assert item["labels"] == 0

    # Test all keys are present
    assert set(item.keys()) == {"sequences", "labels"}


def test_any_dataset_with_mismatched_lengths():
    # Test that mismatched lengths raise an assertion error
    with pytest.raises(AssertionError):
        AnyDataset(sequences=["ACGT", "TGCA"], labels=[0, 1, 0])


def test_any_dataset_add_dataset():
    dataset = AnyDataset(sequences=["ACGT", "TGCA"], labels=[0, 1])
    with pytest.raises(ValueError):
        dataset.add_dataset(key="sequences", dataset=["ACGT", "TGCA"])
    with pytest.raises(ValueError):
        dataset.add_dataset(key="new", dataset=[0, 1, 0])


def test_any_dataset_generate_uid():
    # Test UID generation
    dataset = AnyDataset(sequences=["ACGT"] * 100, labels=[0] * 100, generate_uid=True)
    assert dataset[:]["uid"].tolist() == [i for i in range(100)]


def test_replace_characters_at_indices():
    # Test character replacement function
    input_str = "ACGTACGT"
    indices = [0, 3, 7]
    replacement = "M"

    result = replace_characters_at_indices(input_str, indices, replacement)
    assert result == "MCGMACGM"

    # Test with empty indices
    assert replace_characters_at_indices(input_str, [], replacement) == input_str

    # Test with out-of-range indices
    result = replace_characters_at_indices(input_str, [-1, 8, 2], replacement)
    assert result == "ACMTACGT"


def test_mlm_dataset():
    # Create sample data
    sequences = ["ACGTACGT", "TGCATGCA"]
    dataset = AnyDataset(sequences=sequences)

    # Create MLM dataset with 50% masking rate
    mlm_dataset = MLMDataset(dataset=dataset, masking_rate=0.5)

    # Test length preservation
    assert len(mlm_dataset) == len(dataset)

    # Get a sample
    sample = mlm_dataset[0]

    # Check required keys
    assert "__empty__" in sample
    assert "sequences" in sample
    assert "target_sequences" in sample

    # Check that target sequence matches original
    assert sample["target_sequences"] == sequences[0]

    # Check that input sequence has correct number of masks
    input_seq = sample["sequences"]
    mask_count = input_seq.count("[MASK]")
    assert mask_count == 4


def test_mlm_data_module():
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"seq": ["ACGT", "TGCA"], "labels": [0, 1]}),
            Dataset.from_dict({"seq": ["GCTA"], "labels": [1]}),
            Dataset.from_dict({"seq": ["CGTA"], "labels": [0]}),
        ),
    ):
        module = MLMDataModule(
            path="dummy_path",
            x_col="seq",
            rename_cols={"seq": "renamed_seq"},
            masking_rate=0.5,
        )
        module.setup()
    train_sample = module.train_dataset[0]
    val_sample = module.val_dataset[0]
    test_sample = module.test_dataset[0]
    assert train_sample["renamed_seq"].count("[MASK]") == 2
    assert "seq" not in train_sample
    assert val_sample["renamed_seq"].count("[MASK]") == 2
    assert "seq" not in val_sample
    assert test_sample["renamed_seq"].count("[MASK]") == 2
    assert "seq" not in test_sample


def test_diffusion_dataset():
    # Create sample data
    sequences = ["ACGTACGT", "TGCATGCA"]
    dataset = AnyDataset(sequences=sequences)

    # Test with default settings
    diff_dataset = DiffusionDataset(
        dataset=dataset, timesteps_per_sample=3, randomize_targets=False
    )

    # Test length preservation
    assert len(diff_dataset) == len(dataset)

    # Get a sample
    sample = diff_dataset[0]

    # Check required keys
    assert "sequences" in sample
    assert "target_sequences" in sample
    assert "posterior_weights" in sample

    # Check sample contains correct number of timesteps
    assert len(sample["sequences"]) == 3
    assert len(sample["target_sequences"]) == 3
    assert len(sample["posterior_weights"]) == 3

    # Check masking increases with timesteps
    mask_counts = [seq.count("[MASK]") for seq in sample["sequences"]]
    assert mask_counts[0] <= mask_counts[1] <= mask_counts[2]

    # Check posterior weights are decreasing
    weights = sample["posterior_weights"]
    assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))


def test_diffusion_data_module():
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"seq": ["ACGT", "TGCA"]}),
            Dataset.from_dict({"seq": ["GCTA"]}),
            Dataset.from_dict({"seq": ["CGTA"]}),
        ),
    ):
        module = DiffusionDataModule(
            path="dummy_path",
            x_col="seq",
            rename_cols={"seq": "renamed_seq"},
            timesteps_per_sample=3,
        )
        module.setup()
    train_sample = module.train_dataset[0]
    val_sample = module.val_dataset[0]
    test_sample = module.test_dataset[0]
    assert len(train_sample["renamed_seq"]) == 3
    assert "seq" not in train_sample
    assert len(val_sample["renamed_seq"]) == 1
    assert "seq" not in val_sample
    assert len(test_sample["renamed_seq"]) == 1
    assert "seq" not in test_sample


def test_class_diffusion_data_module():
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"seq": ["ACGT", "TGCA"], "labels": [0, 1]}),
            Dataset.from_dict({"seq": ["GCTA"], "labels": [1]}),
            Dataset.from_dict({"seq": ["CGTA"], "labels": [0]}),
        ),
    ):
        module = ClassDiffusionDataModule(
            path="dummy_path",
            x_col="seq",
            rename_cols={"seq": "renamed_seq"},
            timesteps_per_sample=3,
        )
        module.setup()
    train_sample = module.train_dataset[0]
    val_sample = module.val_dataset[0]
    test_sample = module.test_dataset[0]
    assert len(train_sample["renamed_seq"]) == 3
    assert "seq" not in train_sample
    assert len(val_sample["renamed_seq"]) == 1
    assert "seq" not in val_sample
    assert len(test_sample["renamed_seq"]) == 1
    assert "seq" not in test_sample


def test_conditional_diffusion_data_module():
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"seq": ["ACGT", "TGCA"], "labels": [1.0, 2.0]}),
            Dataset.from_dict({"seq": ["GCTA"], "labels": [1.0]}),
            Dataset.from_dict({"seq": ["CGTA"], "labels": [0.0]}),
        ),
    ):
        module = ConditionalDiffusionDataModule(
            path="dummy_path",
            x_col="seq",
            rename_cols={"seq": "renamed_seq"},
            timesteps_per_sample=3,
        )
        module.setup()
    train_sample = module.train_dataset[0]
    val_sample = module.val_dataset[0]
    test_sample = module.test_dataset[0]
    assert len(train_sample["renamed_seq"]) == 3
    assert "seq" not in train_sample
    assert len(val_sample["renamed_seq"]) == 1
    assert "seq" not in val_sample
    assert len(test_sample["renamed_seq"]) == 1
    assert "seq" not in test_sample


def test_dependency_mapping_dataset(tmp_path):
    # Setup test data
    vocab_file = tmp_path / "vocab.json"
    # Making vocab different so that mutation is easier to track
    vocab_file.write_text("a\nc\ng\nt\n")
    sequences = ["ACGTACGT", "TGCATGCA", "GCTAGCTA"]
    ids = [0, 1, 2]
    data_samples = AnyDataset(sequences=sequences, ids=ids)
    dataset = DependencyMappingDataModule.DependencyMappingDataset(data_samples, vocab_file)

    # Number of sequences generated (mutations per sequence + wild type)
    num_seq_gen = [len(seq) * 4 + 1 for seq in sequences]
    expected_length = sum(num_seq_gen)
    assert len(dataset) == expected_length

    offset = 0
    for i, seq in enumerate(sequences):
        wt_idx = offset + num_seq_gen[i] - 1
        wt_sample = dataset[wt_idx]
        assert wt_sample["sequences"] == seq
        assert wt_sample["pos_i"] == -1
        assert wt_sample["mut_i"] == -1
        assert wt_sample["ids"] == ids[i]
        for j in range(offset, wt_idx):
            sample = dataset[j]
            # Mutation happens at exactly one position
            assert sum(1 for a, b in zip(sample["sequences"], seq) if a != b) == 1
            assert sample["pos_i"] != -1
            assert sample["mut_i"] != -1
            assert sample["ids"] == ids[i]
        offset += num_seq_gen[i]


def test_classification_dataset_splitting():
    # Mock the dataset splitting
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequence": ["seq1", "seq2"], "label": [0, 1]}),
            Dataset.from_dict({"sequence": ["seq3"], "label": [1]}),
            Dataset.from_dict({"sequence": ["seq4"], "label": [0]}),
        ),
    ):
        module = ClassificationDataModule(
            path="dummy_path",
            x_col="sequence",
            y_col="label",
            class_filter=None,
        )
        module.setup()

    # Check train dataset
    assert module.train_dataset[:]["sequence"] == ["seq1", "seq2"]
    assert module.train_dataset[:]["labels"] == [0, 1]

    # Check validation dataset
    assert module.val_dataset[:]["sequence"] == ["seq3"]
    assert module.val_dataset[:]["labels"] == [1]

    # Check test dataset
    assert module.test_dataset[:]["sequence"] == ["seq4"]
    assert module.test_dataset[:]["labels"] == [0]


def test_classification_class_filter():
    # Mock the dataset splitting
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequence": ["seq1", "seq2", "seq3"], "label": [0, 1, 0]}),
            Dataset.from_dict({"sequence": ["seq4"], "label": [1]}),
            Dataset.from_dict({"sequence": ["seq5"], "label": [0]}),
        ),
    ):
        module = ClassificationDataModule(
            path="dummy_path",
            x_col="sequence",
            y_col="label",
            class_filter=[1],
        )
        module.setup()

    # Check train dataset after filtering
    assert module.train_dataset[:]["sequence"] == ["seq2"]
    assert module.train_dataset[:]["labels"] == [1]

    # Check validation dataset after filtering
    assert module.val_dataset[:]["sequence"] == ["seq4"]
    assert module.val_dataset[:]["labels"] == [1]

    # Check test dataset after filtering
    assert len(module.test_dataset) == 0  # No samples with class 1 in test set


def test_classification_multilabel():
    # Mock the dataset splitting
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequence": ["seq1", "seq2"], "l1": [1, 0], "l2": [0, 1]}),
            Dataset.from_dict({"sequence": ["seq3"], "l1": [1], "l2": [1]}),
            Dataset.from_dict({"sequence": ["seq4"], "l1": [0], "l2": [0]}),
        ),
    ):
        module = ClassificationDataModule(
            path="dummy_path",
            x_col="sequence",
            y_col=["l1", "l2"],
            class_filter=None,
        )
        module.setup()

    # Check train dataset
    assert module.train_dataset[:]["sequence"] == ["seq1", "seq2"]
    assert module.train_dataset[:]["labels"].tolist() == [[1, 0], [0, 1]]

    # Check validation dataset
    assert module.val_dataset[:]["sequence"] == ["seq3"]
    assert module.val_dataset[:]["labels"].tolist() == [[1, 1]]

    # Check test dataset
    assert module.test_dataset[:]["sequence"] == ["seq4"]
    assert module.test_dataset[:]["labels"].tolist() == [[0, 0]]


def test_classification_multi_xcol():
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict(
                {
                    "x1": ["seq1", "seq2"],
                    "x2": ["seqA", "seqB"],
                    "label": [0, 1],
                }
            ),
            Dataset.from_dict(
                {
                    "x1": ["seq3"],
                    "x2": ["seqC"],
                    "label": [1],
                }
            ),
            Dataset.from_dict(
                {
                    "x1": ["seq4"],
                    "x2": ["seqD"],
                    "label": [0],
                }
            ),
        ),
    ):
        module = ClassificationDataModule(
            path="dummy_path",
            x_col=["x1", "x2"],
            y_col="label",
            rename_cols={"x1": "sequences1", "x2": "sequences2"},
        )
        module.setup()

    # Check train dataset
    assert module.train_dataset[:]["sequences1"] == ["seq1", "seq2"]
    assert module.train_dataset[:]["sequences2"] == ["seqA", "seqB"]
    assert module.train_dataset[:]["labels"] == [0, 1]

    # Check validation dataset
    assert module.val_dataset[:]["sequences1"] == ["seq3"]
    assert module.val_dataset[:]["sequences2"] == ["seqC"]
    assert module.val_dataset[:]["labels"] == [1]

    # Check test dataset
    assert module.test_dataset[:]["sequences1"] == ["seq4"]
    assert module.test_dataset[:]["sequences2"] == ["seqD"]
    assert module.test_dataset[:]["labels"] == [0]


def test_sequence_classification_default_args():
    """Test default behavior of SequenceClassificationDataModule."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequences": ["seq1", "seq2"], "labels": [0, 1]}),
            Dataset.from_dict({"sequences": ["seq3"], "labels": [1]}),
            Dataset.from_dict({"sequences": ["seq4"], "labels": [0]}),
        ),
    ):
        module = SequenceClassificationDataModule(path="dummy_path")
        module.setup()

    assert module.train_dataset[:]["sequences"] == ["seq1", "seq2"]


def test_collate_pad_labels():
    # Mock input batch
    batch = [
        {"sequences": "ACGT", "labels": torch.tensor([1, 2, 3])},
        {"sequences": "TGCA", "labels": torch.tensor([4, 5])},
    ]

    # Expected padded labels
    expected_padded_labels = torch.tensor([[1, 2, 3], [4, 5, -100]])

    # Call collate_pad_labels
    collate_fn = TokenClassificationDataModule("dummy_path").collate_fn
    result = collate_fn(batch)

    # Check sequences
    assert result["sequences"] == ["ACGT", "TGCA"]

    # Check padded labels
    assert torch.equal(result["labels"], expected_padded_labels)


def test_token_classification_dataset_processing():
    # Pairwise=False
    dataset = [
        {"sequences": "ACGT", "labels": [1, 0, 1, 0]},
        {"sequences": "TGCA", "labels": [0, 1, 0, 1]},
    ]
    module = TokenClassificationDataModule("dummy_path", max_length=3, pairwise=False)
    processed = module.process_dataset(dataset)
    sequences = processed["sequences"]
    labels = processed["labels"]
    assert sequences == ["AC", "TG"]
    assert len(labels) == 2
    assert labels[0].tolist() == [1, 0]
    assert labels[1].tolist() == [0, 1]

    # Pairwise=True
    dataset = [
        {"sequences": "ACGT", "labels": [(1, 0), (0, 1)]},
        {"sequences": "TGCA", "labels": [(0, 1), (1, 2)]},
    ]
    module = TokenClassificationDataModule("dummy_path", max_length=4, pairwise=True)
    processed = module.process_dataset(dataset)
    sequences = processed["sequences"]
    labels = processed["labels"]
    assert sequences == ["ACG", "TGC"]
    assert len(labels) == 2
    assert labels[0].tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert labels[1].tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]


def test_sequence_regression_data_module_setup():
    # Mock the dataset splitting
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequences": ["seq1", "seq2"], "labels": [1.0, 2.0]}),
            Dataset.from_dict({"sequences": ["seq3"], "labels": [3.0]}),
            Dataset.from_dict({"sequences": ["seq4"], "labels": [4.0]}),
        ),
    ):
        module = SequenceRegressionDataModule(
            path="dummy_path",
            normalize=False,
        )
        module.setup()

    # Check train dataset
    assert module.train_dataset[:]["sequences"] == ["seq1", "seq2"]
    assert module.train_dataset[:]["labels"].tolist() == [[1.0], [2.0]]

    # Check validation dataset
    assert module.val_dataset[:]["sequences"] == ["seq3"]
    assert module.val_dataset[:]["labels"].tolist() == [[3.0]]

    # Check test dataset
    assert module.test_dataset[:]["sequences"] == ["seq4"]
    assert module.test_dataset[:]["labels"].tolist() == [[4.0]]


def test_sequence_regression_data_module_normalization():
    # Mock the dataset splitting
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequence": ["seq1", "seq2"], "label": [1.0, 3.0]}),
            Dataset.from_dict({"sequence": ["seq3", "seq4"], "label": [3.0, 1.0]}),
            Dataset.from_dict({"sequence": ["seq5"], "label": [4.0]}),
        ),
    ):
        module = SequenceRegressionDataModule(
            path="dummy_path",
            x_col="sequence",
            y_col="label",
            normalize=True,
        )
        module.setup()

    # Check normalization
    train_labels = module.train_dataset[:]["labels"].tolist()
    val_labels = module.val_dataset[:]["labels"].tolist()
    test_labels = module.test_dataset[:]["labels"].tolist()

    assert train_labels == [[-1.0], [1.0]]  # Normalized values
    assert val_labels == [[1.0], [-1.0]]
    assert test_labels == [[2.0]]


def test_sequence_regression_multiinput_multilabel():
    # Mock the dataset splitting
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict(
                {
                    "input1": ["seq1", "seq2"],
                    "input2": ["seqA", "seqB"],
                    "label1": [1.0, 2.0],
                    "label2": [3.0, 4.0],
                }
            ),
            Dataset.from_dict(
                {
                    "input1": ["seq3"],
                    "input2": ["seqC"],
                    "label1": [5.0],
                    "label2": [6.0],
                }
            ),
            Dataset.from_dict(
                {
                    "input1": ["seq4"],
                    "input2": ["seqD"],
                    "label1": [7.0],
                    "label2": [8.0],
                }
            ),
        ),
    ):
        module = SequenceRegressionDataModule(
            path="dummy_path",
            x_col=["input1", "input2"],
            y_col=["label1", "label2"],
            rename_cols={"input1": "backbone1_sequences", "input2": "backbone2_sequences"},
            normalize=False,
        )
        module.setup()

    # Check train dataset
    assert module.train_dataset[:]["backbone1_sequences"] == ["seq1", "seq2"]
    assert module.train_dataset[:]["backbone2_sequences"] == ["seqA", "seqB"]
    assert module.train_dataset[:]["labels"].tolist() == [[1.0, 3.0], [2.0, 4.0]]

    # Check validation dataset
    assert module.val_dataset[:]["backbone1_sequences"] == ["seq3"]
    assert module.val_dataset[:]["backbone2_sequences"] == ["seqC"]
    assert module.val_dataset[:]["labels"].tolist() == [[5.0, 6.0]]

    # Check test dataset
    assert module.test_dataset[:]["backbone1_sequences"] == ["seq4"]
    assert module.test_dataset[:]["backbone2_sequences"] == ["seqD"]
    assert module.test_dataset[:]["labels"].tolist() == [[7.0, 8.0]]


def test_column_retrieval_data_module():
    # Mock the dataset splitting
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"col1": ["val1", "val2"], "col2": ["valA", "valB"]}),
            Dataset.from_dict({"col1": ["val3"], "col2": ["valC"]}),
            Dataset.from_dict({"col1": ["val4"], "col2": ["valD"]}),
        ),
    ):
        module = ColumnRetrievalDataModule(
            path="dummy_path",
            in_cols=["col1", "col2"],
            out_cols=["new_col1", "new_col2"],
        )
        module.setup()

    # Check train dataset
    assert module.train_dataset[:]["new_col1"] == ["val1", "val2"]
    assert module.train_dataset[:]["new_col2"] == ["valA", "valB"]

    # Check validation dataset
    assert module.val_dataset[:]["new_col1"] == ["val3"]
    assert module.val_dataset[:]["new_col2"] == ["valC"]

    # Check test dataset
    assert module.test_dataset[:]["new_col1"] == ["val4"]
    assert module.test_dataset[:]["new_col2"] == ["valD"]


def test_sequence_classification_class_weight():
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict(
                {
                    "sequence": ["seq1"] * 100,
                    "label": [0] * 20 + [1] * 80,
                }
            ),
            Dataset.from_dict({"sequence": [], "label": []}),
            Dataset.from_dict({"sequence": [], "label": []}),
        ),
    ):
        module = SequenceClassificationDataModule(
            path="dummy_path",
            x_col="sequence",
            y_col="label",
        )
        module.setup()

    assert module.class_weight.equal(torch.tensor([2.5, 0.625]))
