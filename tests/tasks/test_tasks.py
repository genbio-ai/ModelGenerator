import pytest
from unittest.mock import patch
from lightning import Trainer
from modelgenerator.tasks import (
    MLM,
    Inference,
    SequenceClassification,
    TokenClassification,
    PairwiseTokenClassification,
    Diffusion,
    ConditionalDiffusion,
    SequenceRegression,
    Embed,
    SequenceRegressionWithScaling,
)
from modelgenerator.data import (
    ColumnRetrievalDataModule,
    MLMDataModule,
    SequenceClassificationDataModule,
    TokenClassificationDataModule,
    DiffusionDataModule,
    ClassDiffusionDataModule,
    SequenceRegressionDataModule,
)
from datasets import Dataset


@pytest.fixture
def mock_input_only_data():
    """Mocked input only dataset."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequence": ["ACGTACGT" * 4, "TGCATGCA" * 4]}),
            Dataset.from_dict({"sequence": ["GTACGTAC" * 4, "CATGCATG" * 4]}),
            Dataset.from_dict({"sequence": ["TACGTACG" * 4, "CGTACGTA" * 4]}),
        ),
    ):
        yield ColumnRetrievalDataModule("dummy_path", in_cols=["sequence"], out_cols=["sequences"])


@pytest.fixture
def mock_mlm_data():
    """Mocked input only dataset."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequences": ["ACGTACGT" * 4, "TGCATGCA" * 4], "labels": [0, 0]}),
            Dataset.from_dict({"sequences": ["GTACGTAC" * 4, "CATGCATG" * 4], "labels": [0, 0]}),
            Dataset.from_dict({"sequences": ["TACGTACG" * 4, "CGTACGTA" * 4], "labels": [0, 0]}),
        ),
    ):
        yield MLMDataModule("dummy_path")


@pytest.fixture
def mock_sequence_classification_data():
    """Mocked SequenceClassification dataset."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequences": ["ACGTACGT" * 4, "TGCATGCA" * 4], "labels": [0, 1]}),
            Dataset.from_dict({"sequences": ["GTACGTAC" * 4, "CATGCATG" * 4], "labels": [1, 0]}),
            Dataset.from_dict({"sequences": ["TACGTACG" * 4, "CGTACGTA" * 4], "labels": [0, 1]}),
        ),
    ):
        yield SequenceClassificationDataModule("dummy_path")


@pytest.fixture
def mock_sequence_classification_data_multi_label():
    """Mocked SequenceClassification dataset."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict(
                {"sequences": ["ACGTACGT" * 4, "TGCATGCA" * 4], "l1": [0, 1], "l2": [1, 0]}
            ),
            Dataset.from_dict(
                {"sequences": ["GTACGTAC" * 4, "CATGCATG" * 4], "l1": [0, 0], "l2": [1, 0]}
            ),
            Dataset.from_dict(
                {"sequences": ["TACGTACG" * 4, "CGTACGTA" * 4], "l1": [1, 1], "l2": [0, 1]}
            ),
        ),
    ):
        yield SequenceClassificationDataModule("dummy_path", y_col=["l1", "l2"])


@pytest.fixture
def mock_sequence_regression_data():
    """Mocked SequenceRegression dataset."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict(
                {"sequences": ["ACGTACGT" * 4, "TGCATGCA" * 4], "labels": [42.0, 442.0]}
            ),
            Dataset.from_dict(
                {"sequences": ["GTACGTAC" * 4, "CATGCATG" * 4], "labels": [18.9, 77.3]}
            ),
            Dataset.from_dict(
                {"sequences": ["TACGTACG" * 4, "CGTACGTA" * 4], "labels": [99.9, 11.1]}
            ),
        ),
    ):
        yield SequenceRegressionDataModule("dummy_path")


@pytest.fixture
def mock_token_classification_data():
    """Mocked TokenClassification dataset."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict(
                {
                    "sequences": ["ACGTACGT" * 4, "TGCATGCA" * 4],
                    "labels": [[0, 1, 0, 1] * 8, [1, 0, 1, 0] * 8],
                }
            ),
            Dataset.from_dict(
                {
                    "sequences": ["GTACGTAC" * 4, "CATGCATG" * 4],
                    "labels": [[1, 0, 1, 0] * 8, [0, 1, 0, 1] * 8],
                }
            ),
            Dataset.from_dict(
                {
                    "sequences": ["TACGTACG" * 4, "CGTACGTA" * 4],
                    "labels": [[0, 1, 0, 1] * 8, [1, 0, 1, 0] * 8],
                }
            ),
        ),
    ):
        yield TokenClassificationDataModule("dummy_path")


@pytest.fixture
def mock_pairwise_token_classification_data():
    """Mocked PairwiseTokenClassification dataset."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict(
                {
                    "sequences": ["ACGTACGT" * 2, "TGCATGCA" * 2],
                    "labels": [[[0, 1], [1, 0]] * 4, [[1, 0], [0, 1]] * 4],
                }
            ),
            Dataset.from_dict(
                {
                    "sequences": ["GTACGTAC" * 2, "CATGCATG" * 2],
                    "labels": [[[1, 0], [0, 1]] * 4, [[0, 1], [1, 0]] * 4],
                }
            ),
            Dataset.from_dict(
                {
                    "sequences": ["TACGTACG" * 2, "CGTACGTA" * 2],
                    "labels": [[[0, 1], [1, 0]] * 4, [[1, 0], [0, 1]] * 4],
                }
            ),
        ),
    ):
        yield TokenClassificationDataModule("dummy_path", pairwise=True)


@pytest.fixture
def mock_diffusion_data():
    """Mocked Diffusion dataset."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict({"sequences": ["ACGTACGT" * 4, "TGCATGCA" * 4]}),
            Dataset.from_dict({"sequences": ["GTACGTAC" * 4, "CATGCATG" * 4]}),
            Dataset.from_dict({"sequences": ["TACGTACG" * 4, "CGTACGTA" * 4]}),
        ),
    ):
        yield DiffusionDataModule("dummy_path", timesteps_per_sample=10)


@pytest.fixture
def mock_class_diffusion_data():
    """Mocked Diffusion dataset."""
    with patch(
        "modelgenerator.data.base.HFDatasetLoaderMixin.load_and_split_dataset",
        return_value=(
            Dataset.from_dict(
                {"sequences": ["ACGT", "TGCA", "GCTA", "TACG"], "labels": [0, 1, 0, 1]}
            ),
            Dataset.from_dict(
                {"sequences": ["GTAC", "CATG", "ATGC", "CGTA"], "labels": [0, 1, 1, 0]}
            ),
            Dataset.from_dict(
                {"sequences": ["TACG", "CGTA", "ACGT", "TGCA"], "labels": [0, 1, 0, 1]}
            ),
        ),
    ):
        yield ClassDiffusionDataModule("dummy_path", timesteps_per_sample=10)


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls"])
def test_mlm_task(model_cls, mock_mlm_data, request):
    """Test MLM task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = MLM(backbone)
    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_mlm_data)

    # Test the model
    trainer.test(task, mock_mlm_data)

    # Predict with the model
    predictions = trainer.predict(task, mock_mlm_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls"])
def test_inference_task(model_cls, mock_input_only_data, request):
    """Test Inference task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = Inference(backbone)
    trainer = Trainer(fast_dev_run=True)

    # Predict with the model
    predictions = trainer.predict(task, mock_input_only_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls", "enformer_cls"])
def test_sequence_classification_task(model_cls, mock_sequence_classification_data, request):
    """Test SequenceClassification task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = SequenceClassification(backbone, n_classes=2)
    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_sequence_classification_data)

    # Test the model
    trainer.test(task, mock_sequence_classification_data)

    # Predict with the model
    predictions = trainer.predict(task, mock_sequence_classification_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls", "enformer_cls"])
def test_sequence_classification_task_functionality(
    model_cls, mock_sequence_classification_data, request
):
    """Test SequenceClassification task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = SequenceClassification(backbone, n_classes=2)
    task_weighted = SequenceClassification(
        backbone, n_classes=2, weighted_loss=True, data_module=mock_sequence_classification_data
    )

    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_sequence_classification_data)
    trainer.fit(task_weighted, mock_sequence_classification_data)

    # Test the model
    trainer.test(task, mock_sequence_classification_data)
    trainer.test(task_weighted, mock_sequence_classification_data)

    # Predict with the model
    predictions = trainer.predict(task, mock_sequence_classification_data)
    trainer.test(task_weighted, mock_sequence_classification_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls", "enformer_cls"])
def test_sequence_classification_task_multi_label(
    model_cls, mock_sequence_classification_data_multi_label, request
):
    """Test SequenceClassification task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = SequenceClassification(backbone, n_classes=2, multilabel=True)
    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_sequence_classification_data_multi_label)

    # Test the model
    trainer.test(task, mock_sequence_classification_data_multi_label)

    # Predict with the model
    predictions = trainer.predict(task, mock_sequence_classification_data_multi_label)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls"])
def test_token_classification_task(model_cls, mock_token_classification_data, request):
    """Test TokenClassification task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = TokenClassification(backbone, n_classes=2)
    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_token_classification_data)

    # Test the model
    trainer.test(task, mock_token_classification_data)

    # Predict with the model
    predictions = trainer.predict(task, mock_token_classification_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls"])
def test_pairwise_token_classification_task(
    model_cls, mock_pairwise_token_classification_data, request
):
    """Test PairwiseTokenClassification task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = PairwiseTokenClassification(backbone, n_classes=2)
    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_pairwise_token_classification_data)

    # Test the model
    trainer.test(task, mock_pairwise_token_classification_data)

    # Predict with the model
    predictions = trainer.predict(task, mock_pairwise_token_classification_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls"])
def test_diffusion_task(model_cls, mock_diffusion_data, request):
    """Test Diffusion task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = Diffusion(backbone, num_denoise_steps=4)
    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_diffusion_data)

    # Test the model
    trainer.test(task, mock_diffusion_data)

    # Predict with the model
    predictions = trainer.predict(task, mock_diffusion_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls"])
def test_conditional_diffusion_task(model_cls, mock_class_diffusion_data, request):
    """Test ConditionalDiffusion task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = ConditionalDiffusion(backbone, num_denoise_steps=4, condition_dim=1)
    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_class_diffusion_data)

    # Test the model
    trainer.test(task, mock_class_diffusion_data)

    # Predict with the model
    predictions = trainer.predict(task, mock_class_diffusion_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls"])
def test_sequence_regression_task(model_cls, mock_sequence_regression_data, request):
    """Test SequenceRegression task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = SequenceRegression(backbone, num_outputs=1)
    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_sequence_regression_data)

    # Test the model
    trainer.test(task, mock_sequence_regression_data)

    # Predict with the model
    predictions = trainer.predict(task, mock_sequence_regression_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls", "enformer_cls"])
def test_embed_task(model_cls, mock_input_only_data, request):
    """Test Embed task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = Embed(backbone)
    trainer = Trainer(fast_dev_run=True)

    # Predict with the model
    predictions = trainer.predict(task, mock_input_only_data)
    assert predictions is not None


@pytest.mark.parametrize("model_cls", ["genbiobert_cls", "genbiofm_cls"])
def test_sequence_regression_with_scaling_task(model_cls, mock_sequence_regression_data, request):
    """Test SequenceRegressionWithScaling task with mocked backbone and dataset."""
    backbone = request.getfixturevalue(model_cls)
    task = SequenceRegressionWithScaling(backbone, num_outputs=1)
    trainer = Trainer(fast_dev_run=True)

    # Fit the model
    trainer.fit(task, mock_sequence_regression_data)

    # Test the model
    trainer.test(task, mock_sequence_regression_data)

    # Predict with the model
    predictions = trainer.predict(task, mock_sequence_regression_data)
    assert predictions is not None
