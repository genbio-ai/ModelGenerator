import os
from modelgenerator.backbones.backbones import *
from modelgenerator.backbones.base import *


class aido_rna_1m_mars(GenBioBERT):
    """AIDO.RNA model with 1M parameters pretrained on 886M RNAs in the MARS dataset.

    Note: 
        - Mauscript: [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1)
        - Model Card: [AIDO.RNA-1M-MARS](https://huggingface.co/genbio-ai/AIDO.RNA-1M-MARS)
        - Weights: [genbio-ai/AIDO.RNA-1M-MARS](https://huggingface.co/genbio-ai/AIDO.RNA-1M-MARS)

    Attributes:
        model_path (str): genbio-ai/AIDO.RNA-1M-MARS
    """
    model_path = "genbio-ai/AIDO.RNA-1M-MARS"


class aido_rna_25m_mars(GenBioBERT):
    """AIDO.RNA model with 25M parameters pretrained on 886M RNAs in the MARS dataset.
    
    Note:
        - Mauscript: [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1)
        - Model Card: [AIDO.RNA-25M-MARS](https://huggingface.co/genbio-ai/AIDO.RNA-25M-MARS)
        - Weights: [genbio-ai/AIDO.RNA-25M-MARS](https://huggingface.co/genbio-ai/AIDO.RNA-25M-MARS)
        
    Attributes:
        model_path (str): genbio-ai/AIDO.RNA-25M-MARS
    """
    model_path = "genbio-ai/AIDO.RNA-25M-MARS"


class aido_rna_300m_mars(GenBioBERT):
    """AIDO.RNA model with 300M parameters pretrained on 886M RNAs in the MARS dataset.

    Note:
        - Mauscript: [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1)
        - Model Card: [AIDO.RNA-300M-MARS](https://huggingface.co/genbio-ai/AIDO.RNA-300M-MARS)
        - Weights: [genbio-ai/AIDO.RNA-300M-MARS](https://huggingface.co/genbio-ai/AIDO.RNA-300M-MARS)
    
    Attributes:
        model_path (str): genbio-ai/AIDO.RNA-300M-MARS
    """
    model_path = "genbio-ai/AIDO.RNA-300M-MARS"


class aido_rna_650m(GenBioBERT):
    """AIDO.RNA model with 650M parameters pretrained on 42M ncRNAs in the RNACentral database.

    Note:
        - Mauscript: [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1)
        - Model Card: [AIDO.RNA-650M](https://huggingface.co/genbio-ai/AIDO.RNA-650M)
        - Weights: [genbio-ai/AIDO.RNA-650M](https://huggingface.co/genbio-ai/AIDO.RNA-650M)

    Attributes:
        model_path (str): genbio-ai/AIDO.RNA-650M
    """
    model_path = "genbio-ai/AIDO.RNA-650M"


class aido_rna_650m_cds(GenBioBERT):
    """AIDO.RNA model with 650M parameters adapted from `aido_rna_650m` by continued pretrained on 9M coding sequence RNAs from organisms in ENA.

    Note:
        - Mauscript: [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1)
        - Model Card: [AIDO.RNA-650M-CDS](https://huggingface.co/genbio-ai/AIDO.RNA-650M-CDS)
        - Weights: [genbio-ai/AIDO.RNA-650M-CDS](https://huggingface.co/genbio-ai/AIDO.RNA-650M-CDS)

    Attributes:
        model_path (str): genbio-ai/AIDO.RNA-650M-CDS
    """
    model_path = "genbio-ai/AIDO.RNA-650M-CDS"

class aido_rna_1b600m(GenBioBERT):
    """SOTA AIDO.RNA model with 1.6B parameters pretrained on 42M ncRNAs in the RNACentral database.
    
    Note:
        - Mauscript: [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1)
        - Model Card: [AIDO.RNA-1.6B](https://huggingface.co/genbio-ai/AIDO.RNA-1B600M)
        - Weights: [genbio-ai/AIDO.RNA-1.6B](https://huggingface.co/genbio-ai/AIDO.RNA-1B600M)
        
    Attributes:
        model_path (str): genbio-ai/AIDO.RNA-1.6B
    """


class aido_rna_1b600m_cds(GenBioBERT):
    """SOTA AIDO.RNA model with 1.6B parameters adapted from `aido_rna_1b600m` by continued pretrained on 9M coding sequence RNAs from organisms in ENA.

    Note:
        - Mauscript: [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1)
        - Model Card: [AIDO.RNA-1.6B-CDS](https://huggingface.co/genbio-ai/AIDO.RNA-1B600M-CDS)
        - Weights: [genbio-ai/AIDO.RNA-1.6B-CDS](https://huggingface.co/genbio-ai/AIDO.RNA-1B600M-CDS)

    Attributes:
        model_path (str): genbio-ai/AIDO.RNA-1.6B-CDS
    """
    model_path = "genbio-ai/AIDO.RNA-1.6B-CDS"


class aido_dna_dummy(GenBioBERT):
    """A small dummy AIDO.DNA model created from scratch for debugging purposes only

    Note:
        - This model is not intended for any real-world applications and is only for testing purposes.
        - It has a very small number of parameters and is not trained on any data.

    Attributes:
        model_path: genbio-ai/AIDO.DNA-dummy
    """
    model_path = "genbio-ai/AIDO.DNA-dummy"


class aido_dna_300m(GenBioBERT):
    """AIDO.DNA model with 300M parameters pretrained on 10.6B nucleotides from 796 species in the NCBI RefSeq database.
    
    Note:
        - Mauscript: [Accurate and General DNA Representations Emerge from Genome Foundation Models at Scale](https://www.biorxiv.org/content/10.1101/2024.12.01.625444v2)
        - Model Card: [AIDO.DNA-300M](https://huggingface.co/genbio-ai/AIDO.DNA-300M)
        - Weights: [genbio-ai/AIDO.DNA-300M](https://huggingface.co/genbio-ai/AIDO.DNA-300M)
        
    Attributes:
        model_path (str): genbio-ai/AIDO.DNA-300M
    """
    model_path = "genbio-ai/AIDO.DNA-300M"

class aido_dna_7b(GenBioBERT):
    """AIDO.DNA model with 7B parameters pretrained on 10.6B nucleotides from 796 species in the NCBI RefSeq database.

    Note:
        - Mauscript: [Accurate and General DNA Representations Emerge from Genome Foundation Models at Scale](https://www.biorxiv.org/content/10.1101/2024.12.01.625444v2)
        - Model Card: [AIDO.DNA-7B](https://huggingface.co/genbio-ai/AIDO.DNA-7B)
        - Weights: [genbio-ai/AIDO.DNA-7B](https://huggingface.co/genbio-ai/AIDO.DNA-7B)

    Attributes:
        model_path (str): genbio-ai/AIDO.DNA-7B
    """
    model_path = "genbio-ai/AIDO.DNA-7B"


class aido_protein_16b(GenBioFM):
    """AIDO.Protein model with 16B parameters pretrained on 1.2T amino acids from UniRef90 and ColabFoldDB.

    Note:
        - Mauscript: [Mixture of Experts Enable Efficient and Effective Protein Understanding and Design](https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1)
        - Model Card: [AIDO.Protein-16B](https://huggingface.co/genbio-ai/AIDO.Protein-16B)
        - Weights: [genbio-ai/AIDO.Protein-16B](https://huggingface.co/genbio-ai/AIDO.Protein-16B)

    Attributes:
        model_path (str): genbio-ai/AIDO.Protein-16B 
    """
    model_path = "genbio-ai/AIDO.Protein-16B"


class aido_protein_16b_v1(GenBioFM):
    """AIDO.Protein model with 16B parameters adapted from `aido_protein_16b` by continued pretrained on 100B amino acids from UniRef90.

    Note:
        - Mauscript: [Mixture of Experts Enable Efficient and Effective Protein Understanding and Design](https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1)
        - Model Card: [AIDO.Protein-16B-v1](https://huggingface.co/genbio-ai/AIDO.Protein-16B-v1)
        - Weights: [genbio-ai/AIDO.Protein-16B-v1](https://huggingface.co/genbio-ai/AIDO.Protein-16B-v1)

    Attributes:
        model_path (str): genbio-ai/AIDO.Protein-16B-v1
    """ 
    model_path = "genbio-ai/AIDO.Protein-16B-v1"


class aido_protein_rag_16b(GenBioFM):
    """AIDO.Protein-RAG model with 16B parameters adapted from `aido_protein_16b` with 180B tokens of MSA and structural context from UniRef50/UniClust30 and AlphaFold Database.
    
    Note: 
        - Mauscripts: 
            - [Mixture of Experts Enable Efficient and Effective Protein Understanding and Design](https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1)
            - [Retrieval Augmented Protein Language Models for Protein Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.12.02.626519v1)
        - Model Card: [AIDO.Protein-RAG-16B](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-16B)
        - Weights: [genbio-ai/AIDO.Protein-RAG-16B](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-16B)
        
    Attributes:
        model_path (str): genbio-ai/AIDO.Protein-RAG-16B
    """ 
    model_path = "genbio-ai/AIDO.Protein-RAG-16B"


class aido_protein_rag_3b(GenBioFM):
    """AIDO.Protein-RAG model with 3B parameters adapted from a 3B version of AIDO.Protein 16B with 180B tokens of MSA and structural context from UniRef50/UniClust30 and AlphaFold Database.

    Note:
        - Mauscripts: 
            - [Mixture of Experts Enable Efficient and Effective Protein Understanding and Design](https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1)
            - [Retrieval Augmented Protein Language Models for Protein Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.12.02.626519v1)
        - Model Card: [AIDO.Protein-RAG-3B](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-3B)
        - Weights: [genbio-ai/AIDO.Protein-RAG-3B](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-3B)

    Attributes:
        model_path (str): genbio-ai/AIDO.Protein-RAG-3B
    """
    model_path = "genbio-ai/AIDO.Protein-RAG-3B"


class aido_protein2structoken_16b(GenBioFM):
    """AIDO.Protein2StructureToken model with 16B parameters adapted from `aido_protein_16b` and for structure prediction with AIDO.StructureTokenizer.
    The model is trained on 170M sequences and structures from AlphaFold Database and 0.4M sequences and structures from PDB.
    
    Note:
        - Mauscripts: 
            - [Mixture of Experts Enable Efficient and Effective Protein Understanding and Design](https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1)
            - [Balancing Locality and Reconstruction in Protein Structure Tokenizer](https://www.biorxiv.org/content/10.1101/2024.12.02.626366v2)
        - Model Card: [AIDO.Protein2StructureToken-16B](https://huggingface.co/genbio-ai/AIDO.Protein2StructureToken-16B)
        - Weights: [genbio-ai/AIDO.Protein2StructureToken-16B](https://huggingface.co/genbio-ai/AIDO.Protein2StructureToken-16B)
        
    Attributes:
        model_path (str): genbio-ai/AIDO.Protein2StructureToken-16B"""
    model_path = "genbio-ai/AIDO.Protein2StructureToken-16B"


class aido_protein_debug(GenBioFM):
    """A small protein dense transformer model created from scratch for debugging purposes only.

    Note:
        - This model is not intended for any real-world applications and is only for testing purposes.
        - It is created from scratch with a very small number of parameters and is not trained on any data.

    Args:
        *args: Positional arguments passed to the parent class.
        **kwargs: Keyword arguments passed to the parent class.
            `from_scratch=True` and `config_overwrites={'hidden_size': 64, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'intermediate_size': 128}` are always overridden.
    """

    def __init__(
            self, 
            *args, 
            **kwargs
        ):
        from_scratch = True
        config_overwrites = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "num_experts": 2,
        }
        super().__init__(
            *args, from_scratch=from_scratch, config_overwrites=config_overwrites, **kwargs
        )


class aido_dna_debug(GenBioBERT):
    """A small dna/rna dense transformer model created from scratch for debugging purposes only.
    
    Note:
        - This model is not intended for any real-world applications and is only for testing purposes.
        - It is created from scratch with a very small number of parameters and is not trained on any data.

    Args:
        *args: Positional arguments passed to the parent class.
        **kwargs: Keyword arguments passed to the parent class.
            `from_scratch=True` and `config_overwrites={'hidden_size': 64, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'intermediate_size': 128}` are always overridden.
    """

    def __init__(
            self, 
            *args, 
            **kwargs,
        ):
        from_scratch = True
        config_overwrites = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
        }
        super().__init__(
            *args, from_scratch=from_scratch, config_overwrites=config_overwrites, **kwargs
        )


class dna_onehot(Onehot):
    """One-hot encoding for DNA sequences. 
    Used for benchmarking finetuning tasks without pretrained embeddings.

    Attributes:
        vocab_file (str): Path to the vocabulary file `modelgenerator/huggingface_models/dnabert/vocab.txt`
    """
    vocab_file = os.path.join(
        Path(__file__).resolve().parent.parent.parent,
        "modelgenerator/huggingface_models/rnabert/vocab.txt",
    )

class protein_onehot(Onehot):
    """One-hot encoding for protein sequences.
    Used for benchmarking finetuning tasks without pretrained embeddings.
    
    Attributes:
        vocab_file (str): Path to the vocabulary file `modelgenerator/huggingface_models/fm4bio/vocab_protein.txt`
    """
    vocab_file = os.path.join(
        Path(__file__).resolve().parent.parent.parent,
        "modelgenerator/huggingface_models/fm4bio/vocab_protein.txt",
    )


class aido_cell_3m(GenBioCellFoundation):
    """AIDO.Cell model with 3M parameters pretrained on 50M single-cell expression profiles from diverse set of human tissues and organs.
    
    Note:
        - Mauscript: [Scaling Dense Representations for Single Cell with Transcriptome-Scale Context](https://www.biorxiv.org/content/10.1101/2024.11.28.625303v1)
        - Model Card: [AIDO.Cell-3M](https://huggingface.co/genbio-ai/AIDO.Cell-3M)
        - Weights: [genbio-ai/AIDO.Cell-3M](https://huggingface.co/genbio-ai/AIDO.Cell-3M)
        - Integrations:
            - [CZI Virtual Cell Models](https://virtualcellmodels.cziscience.com/model/01964078-54e7-7937-8817-0c53dda9c153)
        
    Attributes:
        model_path (str): genbio-ai/AIDO.Cell-3M
    """
    model_path = "genbio-ai/AIDO.Cell-3M"

class aido_cell_10m(GenBioCellFoundation):
    """AIDO.Cell model with 10M parameters pretrained on 50M single-cell expression profiles from diverse set of human tissues and organs.
    
    Note:
        - Mauscript: [Scaling Dense Representations for Single Cell with Transcriptome-Scale Context](https://www.biorxiv.org/content/10.1101/2024.11.28.625303v1)
        - Model Card: [AIDO.Cell-10M](https://huggingface.co/genbio-ai/AIDO.Cell-10M)
        - Weights: [genbio-ai/AIDO.Cell-10M](https://huggingface.co/genbio-ai/AIDO.Cell-10M)
        - Integrations:
            - [CZI Virtual Cell Models](https://virtualcellmodels.cziscience.com/model/01964078-54e7-7937-8817-0c53dda9c153)
            
    Attributes:
        model_path (str): genbio-ai/AIDO.Cell-10M
    """
    model_path = "genbio-ai/AIDO.Cell-10M"


class aido_cell_100m(GenBioCellFoundation):
    """AIDO.Cell model with 100M parameters pretrained on 50M single-cell expression profiles from diverse set of human tissues and organs.
    
    Note:
        - Mauscript: [Scaling Dense Representations for Single Cell with Transcriptome-Scale Context](https://www.biorxiv.org/content/10.1101/2024.11.28.625303v1)
        - Model Card: [AIDO.Cell-100M](https://huggingface.co/genbio-ai/AIDO.Cell-100M)
        - Weights: [genbio-ai/AIDO.Cell-100M](https://huggingface.co/genbio-ai/AIDO.Cell-100M)
        - Integrations:
            - [CZI Virtual Cell Models](https://virtualcellmodels.cziscience.com/model/01964078-54e7-7937-8817-0c53dda9c153)
            
    Attributes:
        model_path (str): genbio-ai/AIDO.Cell-100M
    """
    model_path = "genbio-ai/AIDO.Cell-100M"


class esm2_8m(ESM):
    """ESM2 8M model

    Note:
        - Mauscript: [Evolutionary-scale prediction of atomic level protein structure with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
        - GitHub: [facebookresearch/esm](https://github.com/facebookresearch/esm)
        - Model Card: [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
        - Weights: [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D)

    Attributes:
        model_path (str): facebook/esm2_t6_8M_UR50D
    """
    model_path = "facebook/esm2_t6_8M_UR50D"


class esm2_35m(ESM):
    """ESM2 35M model

    Note:
        - Mauscript: [Evolutionary-scale prediction of atomic level protein structure with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
        - GitHub: [facebookresearch/esm](https://github.com/facebookresearch/esm)
        - Model Card: [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
        - Weights: [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D)

    Attributes:
        model_path (str): facebook/esm2_t12_35M_UR50D
    """
    model_path = "facebook/esm2_t12_35M_UR50D"


class esm2_150m(ESM):
    """ESM2 150M model

    Note:
        - Mauscript: [Evolutionary-scale prediction of atomic level protein structure with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
        - GitHub: [facebookresearch/esm](https://github.com/facebookresearch/esm)
        - Model Card: [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D)
        - Weights: [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D)

    Attributes:
        model_path (str): facebook/esm2_t30_150M_UR50D
    """
    model_path = "facebook/esm2_t30_150M_UR50D"


class esm2_650m(ESM):
    """ESM2 650M model

    Note:
        - Mauscript: [Evolutionary-scale prediction of atomic level protein structure with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
        - GitHub: [facebookresearch/esm](https://github.com/facebookresearch/esm)
        - Model Card: [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
        - Weights: [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)

    Attributes:
        model_path (str): facebook/esm2_t33_650M_UR50D
    """
    model_path = "facebook/esm2_t33_650M_UR50D"


class esm2_3b(ESM):
    """ESM2 3B model

    Note:
        - Mauscript: [Evolutionary-scale prediction of atomic level protein structure with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
        - GitHub: [facebookresearch/esm](https://github.com/facebookresearch/esm)
        - Model Card: [facebook/esm2_t36_3B_UR50D](https://huggingface.co/facebook/esm2_t36_3B_UR50D)
        - Weights: [facebook/esm2_t36_3B_UR50D](https://huggingface.co/facebook/esm2_t36_3B_UR50D)

    Attributes:
        model_path (str): facebook/esm2_t36_3B_UR50D
    """
    model_path = "facebook/esm2_t36_3B_UR50D"


class esm2_15b(ESM):
    """ESM2 15B model

    Note:
        - Mauscript: [Evolutionary-scale prediction of atomic level protein structure with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
        - GitHub: [facebookresearch/esm](https://github.com/facebookresearch/esm)
        - Model Card: [facebook/esm2_t48_15B_UR50D](https://huggingface.co/facebook/esm2_t48_15B_UR50D)
        - Weights: [facebook/esm2_t48_15B_UR50D](https://huggingface.co/facebook/esm2_t48_15B_UR50D)

    Attributes:
        model_path (str): facebook/esm2_t48_15B_UR50D
    """
    model_path = "facebook/esm2_t48_15B_UR50D"


class enformer(Enformer):
    """Enformer model

    Note:
        - Mauscript: [Effective gene expression prediction from sequence by integrating long-range interactions](https://www.nature.com/articles/s41592-021-01252-x)
        - GitHub: [lucidrains/enformer-pytorch](https://github.com/lucidrains/enformer-pytorch)
        - Model Card: [EleutherAI/enformer-official-rough](https://huggingface.co/EleutherAI/enformer-official-rough)
        - Weights: [EleutherAI/enformer-official-rough](https://huggingface.co/EleutherAI/enformer-official-rough)
        
    Attributes:
        model_path (str): EleutherAI/enformer-official-rough
    """
    model_path = "EleutherAI/enformer-official-rough"


class borzoi(Borzoi):
    """Borzoi model

    Note:
        - Mauscript: [Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation](https://www.nature.com/articles/s41588-024-02053-6)
        - GitHub: [johahi/borzoi](https://github.com/johahi/borzoi-pytorch)
        - Weights: [johahi/borzoi-replicate-0](https://huggingface.co/johahi/borzoi-replicate-0)

    Attributes:
        model_path (str): johahi/borzoi-replicate-0
    """
    model_path = "johahi/borzoi-replicate-0"


class flashzoi(Borzoi):
    """Flashzoi model

    Note:
        - Mauscript: [Flashzoi: A fast and accurate model for predicting RNA-seq coverage from DNA sequence](https://www.biorxiv.org/content/10.1101/2024.12.18.629121v1)
        - GitHub: [johahi/flashzoi](https://github.com/johahi/flashzoi)
        - Weights: [johahi/flashzoi-replicate-0](https://huggingface.co/johahi/flashzoi-replicate-0)

    Attributes:
        model_path (str): johahi/flashzoi-replicate-0
    """
    model_path = "johahi/flashzoi-replicate-0"


class scfoundation(SCFoundation):
    """scFoundation model

    Note:
        - Mauscript: [Large-scale foundation model on single-cell transcriptomics](https://www.nature.com/articles/s41592-024-02305-7)
        - GitHub: [genbio-ai/scFoundation](https://github.com/biomap-research/scFoundation)
        - Model Card: [genbio-ai/scFoundation](https://huggingface.co/genbio-ai/scFoundation)
        - Weights: [genbio-ai/scFoundation](https://huggingface.co/genbio-ai/scFoundation)

    Attributes:
        model_path (str): genbio-ai/scFoundation
    """
    model_path = "genbio-ai/scFoundation"


class aido_tissue_3m(GenBioCellSpatialFoundation):
    """AIDO.Tissue model with 3M parameters adapted from `aido_cell_3m` to incorporate tissue context.

    Note:
        - Mauscript: [Scaling Dense Representations for Single Cell with Transcriptome-Scale Context](https://www.biorxiv.org/content/10.1101/2024.11.28.625303v1)
        - Model Card: [AIDO.Tissue-3M](https://huggingface.co/genbio-ai/AIDO.Tissue-3M)
        - Weights: [genbio-ai/AIDO.Tissue-3M](https://huggingface.co/genbio-ai/AIDO.Tissue-3M)

    Attributes:
        model_path: genbio-ai/AIDO.Tissue-3M
    """
    model_path = "genbio-ai/AIDO.Tissue-3M"


class aido_tissue_60m(GenBioCellSpatialFoundation):
    """AIDO.Tissue model with 60M parameters adapted from AIDO.Cell to incorporate tissue context.
    
    Note:
        - Mauscript: [Scaling Dense Representations for Single Cell with Transcriptome-Scale Context](https://www.biorxiv.org/content/10.1101/2024.11.28.625303v1)
        - Model Card: [AIDO.Tissue-60M](https://huggingface.co/genbio-ai/AIDO.Tissue-60M)
        - Weights: [genbio-ai/AIDO.Tissue-60M](https://huggingface.co/genbio-ai/AIDO.Tissue-60M)

    Attributes:
        model_path (str): genbio-ai/AIDO.Tissue-60M
    """
    model_path = "genbio-ai/AIDO.Tissue-60M"

class geneformer(Geneformer):
    """Geneformer model

    Note:
        - Mauscript: [Transfer learning enables predictions in network biology](https://www.nature.com/articles/s41586-023-06139-9)
        - Model Card: [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer)
        - Weights: [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer)
    
    Attributes:
        model_path (str): ctheodoris/Geneformer
    """
    model_path = "ctheodoris/Geneformer"