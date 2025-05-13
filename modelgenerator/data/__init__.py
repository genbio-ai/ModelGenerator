import os
import subprocess
import warnings
from typing import Union, List
from modelgenerator.data.base import *
from modelgenerator.data.data import *
from functools import partial


class NTClassification(SequenceClassificationDataModule):
    """Nucleotide Transformer benchmarks from InstaDeep.

    Note:
        - Manuscript: [The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3)
        - Data Card: [InstaDeepAI/nucleotide_transformer_downstream_tasks](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)
        - Configs:
            - `promoter_all`
            - `promoter_tata`
            - `promoter_no_tata`
            - `enhancers`
            - `enhancers_types`
            - `splice_sites_all`
            - `splice_sites_acceptor`
            - `splice_sites_donor`
            - `H3`
            - `H4`
            - `H3K9ac`
            - `H3K14ac`
            - `H4ac`
            - `H3K4me1`
            - `H3K4me2`
            - `H3K4me3`
            - `H3K36me3`
            - `H3K79me3`
    """

    def __init__(
        self,
        path: str = "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        config_name: str = "enhancers",
        **kwargs,
    ):
        super().__init__(path=path, config_name=config_name, **kwargs)


class GUEClassification(SequenceClassificationDataModule):
    """Genome Understanding Evaluation benchmarks for DNABERT-2 from the Liu Lab at Northwestern.

    Note:
        - Manuscript: [DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome](https://arxiv.org/abs/2306.15006)
        - Data Card: [leannmlindsey/GUE](https://huggingface.co/datasets/leannmlindsey/GUE)
        - Configs:
            - `emp_H3`
            - `emp_H3K14ac`
            - `emp_H3K36me3`
            - `emp_H3K4me1`
            - `emp_H3K4me2`
            - `emp_H3K4me3`
            - `emp_H3K79me3`
            - `emp_H3K9ac`
            - `emp_H4`
            - `emp_H4ac`
            - `human_tf_0`
            - `human_tf_1`
            - `human_tf_2`
            - `human_tf_3`
            - `human_tf_4`
            - `mouse_0`
            - `mouse_1`
            - `mouse_2`
            - `mouse_3`
            - `mouse_4`
            - `prom_300_all`
            - `prom_300_notata`
            - `prom_300_tata`
            - `prom_core_all`
            - `prom_core_notata`
            - `prom_core_tata`
            - `splice_reconstructed`
            - `virus_covid`
            - `virus_species_40`
            - `fungi_species_20`
            - `EPI_K562`
            - `EPI_HeLa-S3`
            - `EPI_NHEK`
            - `EPI_IMR90`
            - `EPI_HUVEC`
    """

    def __init__(
        self, path: str = "leannmlindsey/GUE", config_name: str = "emp_H3", **kwargs
    ):
        super().__init__(path=path, config_name=config_name, **kwargs)


class ContactPredictionBinary(TokenClassificationDataModule):
    """Protein contact prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/contact_prediction_binary](https://huggingface.co/datasets/proteinglm/contact_prediction_binary)
    
    Args:
        max_context_length: Maximum context length for the input sequences.
        msa_random_seed: Random seed for MSA generation.
        is_rag_dataset: Whether the dataset is a RAG dataset for AIDO.Protein-RAG. 
    """
    def __init__(
        self,
        path: str = "proteinglm/contact_prediction_binary",
        pairwise: bool = True,
        x_col: str = "seq",
        y_col: str = "label",
        batch_size: int = 1,
        max_context_length: int = 12800,
        msa_random_seed: Optional[int] = None,
        is_rag_dataset: bool = False,
        **kwargs,
    ):
        if is_rag_dataset:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            msa_seed_rng = random.Random(0) if msa_random_seed is None else random.Random(rank + msa_random_seed)
            collate_fn = partial(rag_collate_fn, max_context_length=max_context_length, rng=msa_seed_rng)
        else:
            collate_fn = None
        super().__init__(
            path=path,
            pairwise=pairwise,
            x_col=x_col,
            y_col=y_col,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **kwargs,
        )


class SspQ3(TokenClassificationDataModule):
    """Protein secondary structure prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/ssp_q3](https://huggingface.co/datasets/proteinglm/ssp_q3)
    
    Args:
        max_context_length: Maximum context length for the input sequences.
        msa_random_seed: Random seed for MSA generation.
        is_rag_dataset: Whether the dataset is a RAG dataset for AIDO.Protein-RAG.
    """
    def __init__(
        self,
        path: str = "proteinglm/ssp_q3",
        pairwise: bool = False,
        x_col: str = "seq",
        y_col: str = "label",
        batch_size: int = 1,
        max_context_length: int = 12800,
        msa_random_seed: Optional[int] = None,
        is_rag_dataset: bool = False,
        **kwargs,
    ):
        if is_rag_dataset:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            msa_seed_rng = random.Random(0) if msa_random_seed is None else random.Random(rank + msa_random_seed)
            collate_fn = partial(rag_collate_fn, max_context_length=max_context_length, rng=msa_seed_rng)
        else:
            collate_fn = None
        super().__init__(
            path=path,
            pairwise=pairwise,
            x_col=x_col,
            y_col=y_col,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **kwargs,
        )


class FoldPrediction(SequenceClassificationDataModule):
    """Protein fold prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/fold_prediction](https://huggingface.co/datasets/proteinglm/fold_prediction)
    
    Args:
        max_context_length: Maximum context length for the input sequences.
        msa_random_seed: Random seed for MSA generation.
        is_rag_dataset: Whether the dataset is a RAG dataset for AIDO.Protein-RAG.
    """
    def __init__(
        self,
        path: str = "proteinglm/fold_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        max_context_length: int = 12800,
        msa_random_seed: Optional[int] = None,
        is_rag_dataset: bool = False,
        **kwargs,
    ):
        if is_rag_dataset:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            msa_seed_rng = random.Random(0) if msa_random_seed is None else random.Random(rank + msa_random_seed)
            collate_fn = partial(rag_collate_fn, max_context_length=max_context_length, rng=msa_seed_rng)
        else:
            collate_fn = None
        super().__init__(path=path, x_col=x_col, y_col=y_col, collate_fn=collate_fn, **kwargs)


class LocalizationPrediction(SequenceClassificationDataModule):
    """Protein localization prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/localization_prediction](https://huggingface.co/datasets/proteinglm/localization_prediction)
    """
    def __init__(
        self,
        path: str = "proteinglm/localization_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class MetalIonBinding(SequenceClassificationDataModule):
    """Metal ion binding prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/metal_ion_binding](https://huggingface.co/datasets/proteinglm/metal_ion_binding)
    """
    def __init__(
        self,
        path: str = "proteinglm/metal_ion_binding",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class SolubilityPrediction(SequenceClassificationDataModule):
    """Protein solubility prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/solubility_prediction](https://huggingface.co/datasets/proteinglm/solubility_prediction)
    """
    def __init__(
        self,
        path: str = "proteinglm/solubility_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class AntibioticResistance(SequenceClassificationDataModule):
    """Antibiotic resistance prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/antibiotic_resistance](https://huggingface.co/datasets/proteinglm/antibiotic_resistance)
    """
    def __init__(
        self,
        path: str = "proteinglm/antibiotic_resistance",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class CloningClf(SequenceClassificationDataModule):
    """Cloning classification prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/cloning_clf](https://huggingface.co/datasets/proteinglm/cloning_clf)
    """
    def __init__(
        self,
        path: str = "proteinglm/cloning_clf",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class MaterialProduction(SequenceClassificationDataModule):
    """Material production prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/material_production](https://huggingface.co/datasets/proteinglm/material_production)
    """
    def __init__(
        self,
        path: str = "proteinglm/material_production",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class TcrPmhcAffinity(SequenceClassificationDataModule):
    """TCR-pMHC affinity prediction benchmarks from BioMap.

    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/tcr_pmhc_affinity](https://huggingface.co/datasets/proteinglm/tcr_pmhc_affinity)
    """
    def __init__(
        self,
        path: str = "proteinglm/tcr_pmhc_affinity",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class PeptideHlaMhcAffinity(SequenceClassificationDataModule):
    """Peptide-HLA-MHC affinity prediction benchmarks from BioMap.
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/peptide_HLA_MHC_affinity](https://huggingface.co/datasets/proteinglm/peptide_HLA_MHC_affinity)
    """
    def __init__(
        self,
        path: str = "proteinglm/peptide_HLA_MHC_affinity",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class TemperatureStability(SequenceClassificationDataModule):
    """Temperature stability prediction benchmarks from BioMap.

    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/temperature_stability](https://huggingface.co/datasets/proteinglm/temperature_stability)
    """
    def __init__(
        self,
        path: str = "proteinglm/temperature_stability",
        x_col: str = "seq",
        y_col: str = "label",
        **kwargs,
    ):
        super().__init__(path=path, x_col=x_col, y_col=y_col, **kwargs)


class ClinvarRetrieve(ZeroshotClassificationRetrieveDataModule):
    """ClinVar dataset for genomic variant effect prediction.
    
    Note:
        - Manuscript: [The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3)
        - Data Card: [genbio-ai/Clinvar](https://huggingface.co/datasets/genbio-ai/Clinvar)
    """
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
    """Translation efficiency prediction benchmarks from the Wang Lab at Princeton.
    
    Note:
        - Manuscript: [A 5′ UTR language model for decoding untranslated regions of mRNA and function predictions](https://www.nature.com/articles/s42256-024-00823-9)
        - Data Card: [genbio-ai/rna-downstream-tasks](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks)
        - Configs: 
            - `translation_efficiency_Muscle`
            - `translation_efficiency_HEK`
            - `translation_efficiency_pc3`
    """
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
    """Expression level prediction benchmarks from the Wang Lab at Princeton.

    Note:
        - Manuscript: [A 5′ UTR language model for decoding untranslated regions of mRNA and function predictions](https://www.nature.com/articles/s42256-024-00823-9)
        - Data Card: [genbio-ai/rna-downstream-tasks](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks)
        - Configs: 
            - `expression_Muscle`
            - `expression_HEK`
            - `expression_pc3`
    """
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
    """Transcript abundance prediction benchmarks from the Wang Lab at Princeton.
    
    Note:
        - Manuscript: [A 5′ UTR language model for decoding untranslated regions of mRNA and function predictions](https://www.nature.com/articles/s42256-024-00823-9)
        - Data Card: [genbio-ai/rna-downstream-tasks](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks)
        - Configs:
            - `transcript_abundance_athaliana`
            - `transcript_abundance_dmelanogaster`
            - `transcript_abundance_ecoli`
            - `transcript_abundance_hsapiens`
            - `transcript_abundance_hvolcanii`
            - `transcript_abundance_ppastoris`
            - `transcript_abundance_scerevisiae`
    """
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
    """Protein abundance prediction benchmarks from the Wang Lab at Princeton.
    
    Note:
        - Manuscript: [A 5′ UTR language model for decoding untranslated regions of mRNA and function predictions](https://www.nature.com/articles/s42256-024-00823-9)
        - Data Card: [genbio-ai/rna-downstream-tasks](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks)
        - Configs:
            - `protein_abundance_athaliana`
            - `protein_abundance_dmelanogaster`
            - `protein_abundance_ecoli`
            - `protein_abundance_hsapiens`
            - `protein_abundance_scerevisiae`
    """
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
    """Non-coding RNA family classification benchmarks from DPTechnology.
    
    Note:
        - Manuscript: [UNI-RNA: UNIVERSAL PRE-TRAINED MODELS REVOLUTIONIZE RNA RESEARCH](https://www.biorxiv.org/content/10.1101/2023.07.11.548588v1)
        - Data Card: [genbio-ai/rna-downstream-tasks](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks)
        - Configs:
            - `ncrna_family_bnoise0`
            - `ncrna_family_bnoise200`
    """
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
    """Splice site prediction benchmarks from the Thompson Lab at University of Strasbourg.
    
    Note:
        - Manuscript: [Spliceator: multi-species splice site prediction using convolutional neural networks](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04471-3)
        - Data Card: [genbio-ai/rna-downstream-tasks](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks)
        - Configs:
            - `splice_site_acceptor`
            - `splice_site_donor`
    """
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
    """Modification site prediction benchmarks from the Meng Lab at the University of Liverpool.
    
    Note:
        - Manuscript: [Attention-based multi-label neural networks for integrated prediction and interpretation of twelve widely occurring RNA modifications](https://www.nature.com/articles/s41467-021-24313-3)
        - Data Card: [genbio-ai/rna-downstream-tasks](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks)
        - Configs:
            - `modification_site`
    """
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
    """Gene expression prediction from promoter sequences from the Regev Lab at the Broad Institute.

    Note:
        - Manuscript: [Deciphering eukaryotic gene-regulatory logic with 100 million random promoters](https://www.nature.com/articles/s41587-019-0315-8)
        - Data Card: [genbio-ai/100M-random-promoters](https://huggingface.co/datasets/genbio-ai/100M-random-promoters)
    """

    def __init__(
        self,
        path: str = "genbio-ai/100M-random-promoters",
        x_col: str = "sequence",
        y_col: str = "label",
        normalize: bool = True,
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
    """Promoter generation from gene expression data from the Regev Lab at the Broad Institute.

    Note:
        - Manuscript: [Deciphering eukaryotic gene-regulatory logic with 100 million random promoters](https://www.nature.com/articles/s41587-019-0315-8)
        - Data Card: [genbio-ai/100M-random-promoters](https://huggingface.co/datasets/genbio-ai/100M-random-promoters)
    """

    def __init__(
        self,
        path: str = "genbio-ai/100M-random-promoters",
        x_col: str = "sequence",
        y_col: str = "label",
        normalize: bool = True,
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
    """Fluorescence prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/fluorescence_prediction](https://huggingface.co/datasets/proteinglm/fluorescence_prediction) 
    
    Args:
        max_context_length: Maximum context length for the input sequences.
        msa_random_seed: Random seed for MSA generation.
        is_rag_dataset: Whether the dataset is a RAG dataset for AIDO.Protein-RAG.
    """
    def __init__(
        self,
        path: str = "proteinglm/fluorescence_prediction",
        x_col: str = "seq",
        y_col: str = "label",
        normalize: bool = True,
        max_context_length: int = 12800,
        msa_random_seed: Optional[int] = None,
        is_rag_dataset: bool = False,
        **kwargs,
    ):
        if is_rag_dataset:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            msa_seed_rng = random.Random(0) if msa_random_seed is None else random.Random(rank + msa_random_seed)
            collate_fn = partial(rag_collate_fn, max_context_length=max_context_length, rng=msa_seed_rng)
        else:
            collate_fn = None
        super().__init__(
            path=path, normalize=normalize, x_col=x_col, y_col=y_col, collate_fn=collate_fn, **kwargs
        )


class FitnessPrediction(SequenceRegressionDataModule):
    """Fitness prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/fitness_prediction](https://huggingface.co/datasets/proteinglm/fitness_prediction)
    """
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
    """Stability prediction benchmarks from BioMap.

    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/stability_prediction](https://huggingface.co/datasets/proteinglm/stability_prediction)
    """
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
    """Enzyme catalytic efficiency prediction benchmarks from BioMap.

    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/enzyme_catalytic_efficiency](https://huggingface.co/datasets/proteinglm/enzyme_catalytic_efficiency)
    """
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
    """Optimal temperature prediction benchmarks from BioMap.
    
    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/optimal_temperature](https://huggingface.co/datasets/proteinglm/optimal_temperature)
    """
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
    """Optimal pH prediction benchmarks from BioMap.

    Note:
        - Manuscript: [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v5)
        - Data Card: [proteinglm/optimal_ph](https://huggingface.co/datasets/proteinglm/optimal_ph)
    """
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
    """Deep mutational scanning (DMS) fitness prediction benchmarks from the Gal Lab at Oxford and the Marks Lab at Harvard.
    
    Note:
        - Manuscript: [ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design](https://proteingym.org/)
        - Data Card: [genbio-ai/ProteinGYM-DMS](https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS)
    
    Args:
        max_context_length: Maximum context length for the input sequences.
        msa_random_seed: Random seed for MSA generation.
        is_rag_dataset: Whether the dataset is a RAG dataset for AIDO.Protein-RAG.
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
        cv_replace_val_fold_as_test_fold: bool = False,
        cv_fold_id_col: str = "fold_id",
        cv_val_offset: int = -1,
        valid_split_name: str = None,
        valid_split_size: float = 0,
        test_split_name: str = None,
        test_split_size: float = 0,
        max_context_length: int = 12800,
        msa_random_seed: Optional[int] = None,
        is_rag_dataset: bool = False,
        **kwargs,
    ):

        if is_rag_dataset:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            msa_seed_rng = random.Random(0) if msa_random_seed is None else random.Random(rank + msa_random_seed)
            collate_fn = partial(rag_collate_fn, max_context_length=max_context_length, rng=msa_seed_rng)
        else:
            collate_fn = None

        super().__init__(
            path=path,
            train_split_files=train_split_files,
            x_col=x_col,
            y_col=y_col,
            cv_num_folds=cv_num_folds,
            cv_test_fold_id=cv_test_fold_id,
            cv_enable_val_fold=cv_enable_val_fold,
            cv_replace_val_fold_as_test_fold=cv_replace_val_fold_as_test_fold,
            cv_fold_id_col=cv_fold_id_col,
            cv_val_offset=cv_val_offset,
            valid_split_name=valid_split_name,
            valid_split_size=valid_split_size,
            test_split_name=test_split_name,
            test_split_size=test_split_size,
            collate_fn=collate_fn,
            **kwargs,
        )


class IsoformExpression(SequenceRegressionDataModule):
    """Isoform expression prediction benchmarks from the 

    Note:
        - Manuscript: [Multi-modal Transfer Learning between Biological Foundation Models](https://arxiv.org/abs/2406.14150)
        - Data Card: [genbio-ai/transcript_isoform_expression_prediction](https://huggingface.co/datasets/genbio-ai/transcript_isoform_expression_prediction)
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
