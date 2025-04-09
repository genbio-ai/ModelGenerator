import datasets
import pytest
from unittest.mock import patch, Mock
from modelgenerator.data.base import HFDatasetLoaderMixin, KFoldMixin


@pytest.fixture
def mock_dataset():
    data = {"feature": list(range(100)), "fold_id": [i % 5 for i in range(100)]}
    return datasets.Dataset.from_dict(data)


def test_generate_kfold_split():
    mixin = KFoldMixin()
    mixin.cv_num_folds = 5
    mixin.random_seed = 42
    splits = mixin.generate_kfold_split(num_samples=100, num_folds=5)
    assert len(splits) == 5
    assert sum(len(fold) for fold in splits) == 100
    assert all(len(set(fold)) == len(fold) for fold in splits)  # No duplicate indices


def test_read_kfold_split(mock_dataset):
    mixin = KFoldMixin()
    mixin.cv_num_folds = 5
    mixin.cv_fold_id_col = "fold_id"
    splits = mixin.read_kfold_split(mock_dataset)
    assert len(splits) == 5
    for i, fold in enumerate(splits):
        assert all(mock_dataset["fold_id"][idx] == i for idx in fold)


def test_get_split_by_fold_id(mock_dataset):
    mixin = KFoldMixin()
    mixin.cv_num_folds = 5
    mixin.cv_fold_id_col = "fold_id"
    mixin.cv_enable_val_fold = True
    train, val, test = mixin.get_split_by_fold_id(mock_dataset, [], [], fold_id=0)
    assert len(test) == 20  # 1/5 of the dataset
    assert len(val) == 20  # 1/5 of the dataset
    assert len(train) == 60  # Remaining samples
    assert len(set(train["feature"]) & set(val["feature"])) == 0  # No overlap
    assert len(set(train["feature"]) & set(test["feature"])) == 0  # No overlap
    assert len(set(val["feature"]) & set(test["feature"])) == 0  # No overlap


@pytest.fixture(scope="function")
def empty_hf_loader_mixin():
    loader = HFDatasetLoaderMixin()
    loader.path = "data/path"
    loader.train_split_name = None
    loader.valid_split_name = None
    loader.test_split_name = None
    loader.train_split_files = None
    loader.valid_split_files = None
    loader.test_split_files = None
    loader.valid_split_size = 0
    loader.test_split_size = 0
    loader.config_name = None
    loader.random_seed = 42
    return loader


@pytest.mark.parametrize("train_split", ["train", None])
@pytest.mark.parametrize("valid_split", ["valid", None])
@pytest.mark.parametrize("test_split", ["test", None])
def test_load_dataset_split(
    empty_hf_loader_mixin, train_split, valid_split, test_split
):
    loader = empty_hf_loader_mixin
    loader.train_split_name = train_split
    loader.valid_split_name = valid_split
    loader.test_split_name = test_split
    extra_args = {"a": "b"}
    with patch("modelgenerator.data.base.load_dataset") as mock:

        def side_effect(*args, **kwargs):
            return kwargs["split"]

        mock.side_effect = side_effect
        splits = loader.load_dataset(**extra_args)
        call_count = 0
        for split_name in (train_split, valid_split, test_split):
            if split_name is None:
                continue
            call_count += 1
            mock.assert_any_call(
                loader.path,
                name=loader.config_name,
                data_files=None,
                streaming=False,
                split=split_name,
                **extra_args,
            )
        assert mock.call_count == call_count
        assert splits == (train_split, valid_split, test_split)


@pytest.mark.parametrize("train_files", [["train", "files"], None])
@pytest.mark.parametrize("valid_files", [["valid", "files"], None])
@pytest.mark.parametrize("test_files", [["test", "files"], None])
def test_load_dataset_files(
    empty_hf_loader_mixin, train_files, valid_files, test_files
):
    loader = empty_hf_loader_mixin
    loader.train_split_name = "train"
    loader.valid_split_name = "valid"
    loader.test_split_name = "test"
    loader.train_split_files = train_files
    loader.valid_split_files = valid_files
    loader.test_split_files = test_files
    extra_args = {"a": "b"}
    splits = ("train", "valid", "test")
    with patch("modelgenerator.data.base.load_dataset") as mock:
        loader.load_dataset(**extra_args)
        data_files = {
            split: files
            for split, files in zip(splits, (train_files, valid_files, test_files))
            if files is not None
        }
        for split_name in splits:
            mock.assert_any_call(
                loader.path,
                name=loader.config_name,
                data_files=data_files or None,
                streaming=False,
                split=split_name,
                **extra_args,
            )
        assert mock.call_count == 3


def test_load_dataset_error(empty_hf_loader_mixin):
    loader = empty_hf_loader_mixin
    loader.train_split_name = "train"
    loader.valid_split_name = "valid"
    with patch("modelgenerator.data.base.load_dataset", Mock(side_effect=ValueError())):
        with pytest.warns(UserWarning):
            split = loader.load_dataset()
        assert len(split) == 3


@pytest.mark.parametrize(
    "valid_split,test_split,valid_size,test_size",
    (
        (None, None, 0.1, 0.2),
        ("valid", None, 0, 0.2),
        (None, "test", 0.1, 0),
    ),
)
def test_load_and_split_dataset_auto_split(
    empty_hf_loader_mixin, valid_split, test_split, valid_size, test_size
):
    loader = empty_hf_loader_mixin
    loader.train_split_name = "train"
    loader.valid_split_name = valid_split
    loader.test_split_name = test_split
    loader.valid_split_size = valid_size
    loader.test_split_size = test_size
    with patch("modelgenerator.data.base.load_dataset") as load_mock:
        load_mock.train_test_split = Mock()
        loader.load_and_split_dataset()
        call_count = 0
        if valid_split and valid_size:
            call_count += 1
            load_mock.train_test_split.assert_any_call(
                test_size=valid_size, random_seed=loader.random_seed
            )
        if test_split and test_size:
            call_count += 1
            load_mock.train_test_split.assert_any_call(
                test_size=test_size, random_seed=loader.random_seed
            )
        assert load_mock.train_test_split.call_count == call_count


def test_load_and_split_dataset_empty(empty_hf_loader_mixin):
    loader = empty_hf_loader_mixin
    with patch.object(loader, "load_dataset") as mock:
        # Test that an error is raised if no splits are specified
        mock.return_value = (None, None, None)
        with pytest.raises(ValueError):
            loader.load_and_split_dataset()
        dataset = datasets.Dataset.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Test that an empty split still contains all the columns
        mock.return_value = (None, dataset, dataset)
        train, _, _ = loader.load_and_split_dataset()
        assert train.column_names == ["a", "b"]
        assert len(train) == 0
        mock.return_value = (dataset, None, dataset)
        _, valid, _ = loader.load_and_split_dataset()
        assert valid.column_names == ["a", "b"]
        assert len(valid) == 0
        mock.return_value = (dataset, dataset, None)
        _, _, test = loader.load_and_split_dataset()
        assert test.column_names == ["a", "b"]
        assert len(test) == 0
