import os
import subprocess
import warnings
from typing import Union, List
from modelgenerator.data.base import *
from modelgenerator.data.data import *
from typing import Union


class NTClassification(SequenceClassificationDataModule):
    """Handle for the default Nucleotide Transformer benchmarks https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3"""

    def __init__(
        self,
        path: str = "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        config_name: str = "enhancers",
        **kwargs,
    ):
        super().__init__(path=path, config_name=config_name, **kwargs)


class GUEClassification(SequenceClassificationDataModule):
    """Handle for the Genome Understanding Evlaution benchmarks from DNABERT 2 https://arxiv.org/abs/2306.15006"""

    def __init__(
        self, path: str = "leannmlindsey/GUE", config_name: str = "emp_H3", **kwargs
    ):
        super().__init__(path=path, config_name=config_name, **kwargs)


class ContactPredictionBinary(TokenClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/contact_prediction_binary",
        pairwise: bool = True,
        x_col: str = "seq",
        y_col: str = "label",
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            path=path,
            pairwise=pairwise,
            x_col=x_col,
            y_col=y_col,
            batch_size=batch_size,
            **kwargs,
        )


class SspQ3(TokenClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/ssp_q3",
        pairwise: bool = False,
        x_col: str = "seq",
        y_col: str = "label",
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            path=path,
            pairwise=pairwise,
            x_col=x_col,
            y_col=y_col,
            batch_size=batch_size,
            **kwargs,
        )


class FoldPrediction(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/fold_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class LocalizationPrediction(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/localization_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class MetalIonBinding(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/metal_ion_binding",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class SolubilityPrediction(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/solubility_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class AntibioticResistance(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/antibiotic_resistance",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class CloningClf(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/cloning_clf",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class MaterialProduction(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/material_production",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class TcrPmhcAffinity(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/tcr_pmhc_affinity",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class PeptideHlaMhcAffinity(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/peptide_HLA_MHC_affinity",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class TemperatureStability(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "proteinglm/temperature_stability",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class ClinvarRetrieve(ZeroshotClassificationRetrieveDataModule):
    def __init__(
        self,
        path: str = None,
        test_split_files: List[str] = ["ClinVar_Processed.tsv"],
        reference_file: str = "hg38.ml.fa",
        method: str = "Distance",
        window: int = 512,
        **kwargs,
    ):
        # Check and initialize GENBIO_DATA_DIR
        if "GENBIO_DATA_DIR" not in os.environ:
            default_dir = os.path.abspath("./genbio_data")
            warnings.warn(f"'GENBIO_DATA_DIR' not found in environment. Using default: {default_dir}")
            os.environ["GENBIO_DATA_DIR"] = default_dir

        # Default path if not explicitly passed
        if path is None:
            path = os.path.join(os.environ["GENBIO_DATA_DIR"], 'genbio_finetune', 'dna_datasets')

        # Check and download the files if they don't exist
        self.download_files(path)

        # Initialize the parent class
        super().__init__(
            path=path,
            test_split_files=test_split_files,
            reference_file=reference_file,
            method=method,
            window=window,
            y_col="effect",
            **kwargs,
        )

    def download_files(self, save_dir):
        files_to_download = ["ClinVar_Processed.tsv", "ClinVar_demo.tsv", "hg38.ml.fa"]

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        for file in files_to_download:
            file_path = os.path.join(save_dir, file)
            if not os.path.exists(file_path):
                subprocess.run(
                    ["wget", "-O", file_path, os.path.join("https://huggingface.co/datasets/genbio-ai/Clinvar/resolve/main", file)]
                )
                print(f"Downloaded {file} to {file_path}")
            else:
                print(f"{file} already exists, skipping download.")


class TranslationEfficiency(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "genbio-ai/rna-downstream-tasks",
        config_name: str = "translation_efficiency_Muscle",
        x_col="sequences",
        y_col="labels",
        normalize: bool = True,
        cv_num_folds: int = 10,
        cv_test_fold_id: int = 0,
        cv_enable_val_fold: bool = True,
        cv_fold_id_col: str = "fold_id",
        valid_split_name: str = None,
        valid_split_size: float = 0,
        test_split_name: str = None,
        test_split_size: float = 0,
        **kwargs,
    ):
        super().__init__(
            path=path,
            config_name=config_name,
            x_col=x_col,
            y_col=y_col,
            normalize=normalize,
            cv_num_folds=cv_num_folds,
            cv_test_fold_id=cv_test_fold_id,
            cv_enable_val_fold=cv_enable_val_fold,
            cv_fold_id_col=cv_fold_id_col,
            valid_split_name=valid_split_name,
            valid_split_size=valid_split_size,
            test_split_name=test_split_name,
            test_split_size=test_split_size,
            **kwargs,
        )


class ExpressionLevel(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "genbio-ai/rna-downstream-tasks",
        config_name: str = "expression_Muscle",
        x_col: str = "sequences",
        y_col: str = "labels",
        normalize: bool = True,
        cv_num_folds: int = 10,
        cv_test_fold_id: int = 0,
        cv_enable_val_fold: bool = True,
        cv_fold_id_col: str = "fold_id",
        valid_split_name: str = None,
        valid_split_size: float = 0,
        test_split_name: str = None,
        test_split_size: float = 0,
        **kwargs,
    ):
        super().__init__(
            path=path,
            config_name=config_name,
            x_col=x_col,
            y_col=y_col,
            normalize=normalize,
            cv_num_folds=cv_num_folds,
            cv_test_fold_id=cv_test_fold_id,
            cv_enable_val_fold=cv_enable_val_fold,
            cv_fold_id_col=cv_fold_id_col,
            valid_split_name=valid_split_name,
            valid_split_size=valid_split_size,
            test_split_name=test_split_name,
            test_split_size=test_split_size,
            **kwargs,
        )


class TranscriptAbundance(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "genbio-ai/rna-downstream-tasks",
        config_name: str = "transcript_abundance_athaliana",
        x_col: str = "sequences",
        y_col: str = "labels",
        normalize: bool = True,
        cv_num_folds: int = 5,
        cv_test_fold_id: int = 0,
        cv_enable_val_fold: bool = True,
        cv_fold_id_col: str = "fold_id",
        valid_split_name: str = None,
        valid_split_size: float = 0,
        test_split_name: str = None,
        test_split_size: float = 0,
        **kwargs,
    ):
        super().__init__(
            path=path,
            config_name=config_name,
            x_col=x_col,
            y_col=y_col,
            normalize=normalize,
            cv_num_folds=cv_num_folds,
            cv_test_fold_id=cv_test_fold_id,
            cv_enable_val_fold=cv_enable_val_fold,
            cv_fold_id_col=cv_fold_id_col,
            valid_split_name=valid_split_name,
            valid_split_size=valid_split_size,
            test_split_name=test_split_name,
            test_split_size=test_split_size,
            **kwargs,
        )


class ProteinAbundance(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "genbio-ai/rna-downstream-tasks",
        config_name: str = "protein_abundance_athaliana",
        x_col: str = "sequences",
        y_col: str = "labels",
        normalize: bool = True,
        cv_num_folds: int = 5,
        cv_test_fold_id: int = 0,
        cv_enable_val_fold: bool = True,
        cv_fold_id_col: str = "fold_id",
        valid_split_name: str = None,
        valid_split_size: float = 0,
        test_split_name: str = None,
        test_split_size: float = 0,
        **kwargs,
    ):
        super().__init__(
            path=path,
            config_name=config_name,
            x_col=x_col,
            y_col=y_col,
            normalize=normalize,
            cv_num_folds=cv_num_folds,
            cv_test_fold_id=cv_test_fold_id,
            cv_enable_val_fold=cv_enable_val_fold,
            cv_fold_id_col=cv_fold_id_col,
            valid_split_name=valid_split_name,
            valid_split_size=valid_split_size,
            test_split_name=test_split_name,
            test_split_size=test_split_size,
            **kwargs,
        )


class NcrnaFamilyClassification(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "genbio-ai/rna-downstream-tasks",  ## Ori
        config_name: str = "ncrna_family_bnoise0",
        x_col: str = "sequences",
        y_col: str = "labels",
        train_split_name: str = "train",
        valid_split_name: str = "validation",
        test_split_name: str = "test",
        **kwargs,
    ):
        super().__init__(
            path=path,
            config_name=config_name,
            x_col=x_col,
            y_col=y_col,
            train_split_name=train_split_name,
            valid_split_name=valid_split_name,
            test_split_name=test_split_name,
            **kwargs,
        )


class SpliceSitePrediction(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "genbio-ai/rna-downstream-tasks",
        config_name: str = "splice_site_acceptor",
        x_col: str = "sequences",
        y_col: str = "labels",
        train_split_name: str = "train",
        valid_split_name: str = "validation",
        test_split_name: str = "test_danio",
        batch_size: int = 16,
        **kwargs,
    ):
        super().__init__(
            path=path,
            config_name=config_name,
            x_col=x_col,
            y_col=y_col,
            train_split_name=train_split_name,
            valid_split_name=valid_split_name,
            test_split_name=test_split_name,
            batch_size=batch_size,
            **kwargs,
        )


class ModificationSitePrediction(SequenceClassificationDataModule):
    def __init__(
        self,
        path: str = "genbio-ai/rna-downstream-tasks",
        config_name: str = "modification_site",
        x_col: str = "sequences",
        y_col: List[str] = [f"labels_{i}" for i in range(12)],
        train_split_name: str = "train",
        valid_split_name: str = "validation",
        test_split_name: str = "test",
        **kwargs,
    ):
        super().__init__(
            path=path,
            config_name=config_name,
            x_col=x_col,
            y_col=y_col,
            train_split_name=train_split_name,
            valid_split_name=valid_split_name,
            test_split_name=test_split_name,
            **kwargs,
        )


class PromoterExpressionRegression(SequenceRegressionDataModule):
    """Data module for predicting expression from promoter sequences. Inherits from SequenceRegression.

    Args:
        x_col (str, optional): The name of the column containing the sequences. Defaults to "sequence".
        y_col (str, optional): The name of the column containing the labels. Defaults to "label".
        normalize (bool, optional): Whether to normalize the labels. Defaults to True.
    """

    def __init__(
        self,
        path: str = "genbio-ai/100M-random-promoters",
        x_col: str = "sequence",
        y_col: str = "label",
        normalize: bool = True,
        valid_split_size: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            path=path,
            x_col=x_col,
            y_col=y_col,
            normalize=normalize,
            valid_split_size=valid_split_size,
            **kwargs,
        )


class PromoterExpressionGeneration(ConditionalDiffusionDataModule):
    """Data module for generating promoters toward a given expression label. Inherits from SequenceRegression.

    Args:
        x_col (str, optional): The name of the column containing the sequences. Defaults to "sequence".
        y_col (str, optional): The name of the column containing the labels. Defaults to "label".
        normalize (bool, optional): Whether to normalize the labels. Defaults to True.
    """

    def __init__(
        self,
        path: str = "genbio-ai/100M-random-promoters",
        x_col: str = "sequence",
        y_col: str = "label",
        normalize: bool = True,
        valid_split_size: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            path=path,
            x_col=x_col,
            y_col=y_col,
            normalize=normalize,
            valid_split_size=valid_split_size,
            **kwargs,
        )


class FluorescencePrediction(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "proteinglm/fluorescence_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(
            path=path, normalize=normalize, x_col=x_col, y_col=y_col, **kwargs
        )


class FitnessPrediction(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "proteinglm/fitness_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(
            path=path, normalize=normalize, x_col=x_col, y_col=y_col, **kwargs
        )


class StabilityPrediction(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "proteinglm/stability_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(
            path=path, normalize=normalize, x_col=x_col, y_col=y_col, **kwargs
        )


class EnzymeCatalyticEfficiencyPrediction(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "proteinglm/enzyme_catalytic_efficiency",
        x_col: str = "seq",
        y_col: str = "label",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(
            path=path, normalize=normalize, x_col=x_col, y_col=y_col, **kwargs
        )


class OptimalTemperaturePrediction(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "proteinglm/optimal_temperature",
        x_col: str = "seq",
        y_col: str = "label",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(
            path=path, normalize=normalize, x_col=x_col, y_col=y_col, **kwargs
        )


class OptimalPhPrediction(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "proteinglm/optimal_ph",
        x_col: str = "seq",
        y_col: str = "label",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(
            path=path, normalize=normalize, x_col=x_col, y_col=y_col, **kwargs
        )


class DMSFitnessPrediction(SequenceRegressionDataModule):
    """Data module for predicting fitness from protein sequences. Inherits from SequenceRegression.

    Args:
        x_col (str, optional): The name of the column containing the sequences. Defaults to "sequence".
        y_col (str, optional): The name of the column containing the labels. Defaults to "label".
        normalize (bool, optional): Whether to normalize the labels. Defaults to True.
    """

    def __init__(
        self,
        path: str = "genbio-ai/ProteinGYM-DMS",
        train_split_files: list[str] = ["indels/B1LPA6_ECOSM_Russ_2020_indels.tsv"],
        x_col: str = "sequences",
        y_col: str = "labels",
        cv_num_folds: int = 5,
        cv_test_fold_id: int = 0,
        cv_enable_val_fold: bool = True,
        cv_fold_id_col: str = "fold_id",
        cv_val_offset: int = -1,
        valid_split_name: str = None,
        valid_split_size: float = 0,
        test_split_name: str = None,
        test_split_size: float = 0,
        **kwargs,
    ):
        super().__init__(
            path=path,
            train_split_files=train_split_files,
            x_col=x_col,
            y_col=y_col,
            cv_num_folds=cv_num_folds,
            cv_test_fold_id=cv_test_fold_id,
            cv_enable_val_fold=cv_enable_val_fold,
            cv_fold_id_col=cv_fold_id_col,
            cv_val_offset=cv_val_offset,
            valid_split_name=valid_split_name,
            valid_split_size=valid_split_size,
            test_split_name=test_split_name,
            test_split_size=test_split_size,
            **kwargs,
        )


class IsoformExpression(SequenceRegressionDataModule):
    """
    Data class for single/multimodal isoform expression prediction tasks
    """

    def __init__(
        self,
        path: str = "genbio-ai/transcript_isoform_expression_prediction",
        config_name: str = None,
        x_col: Union[str, list] = ["dna_seq", "rna_seq", "protein_seq"],
        valid_split_name="valid",
        train_split_files: Optional[Union[str, list[str]]] = "train_*.tsv",
        test_split_files: Optional[Union[str, list[str]]] = "test.tsv",
        valid_split_files: Optional[Union[str, list[str]]] = "validation.tsv",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(
            path=path,
            config_name=config_name,
            x_col=x_col,
            y_col=[f"labels_{i}" for i in range(30)],
            valid_split_name=valid_split_name,
            train_split_files=train_split_files,
            test_split_files=test_split_files,
            valid_split_files=valid_split_files,
            normalize=normalize,
            extra_reader_kwargs={"keep_default_na": False},
            **kwargs,
        )
