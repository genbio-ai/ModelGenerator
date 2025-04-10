import os
from modelgenerator.backbones.backbones import *
from modelgenerator.backbones.base import *


aido_rna_1m_mars = type(
    "aido_rna_1m_mars",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.RNA-1M-MARS",
    },
)
aido_rna_25m_mars = type(
    "aido_rna_25m_mars",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.RNA-25M-MARS",
    },
)
aido_rna_300m_mars = type(
    "aido_rna_300m_mars",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.RNA-300M-MARS",
    },
)
aido_rna_650m = type(
    "aido_rna_650m",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.RNA-650M",
    },
)
aido_rna_650m_cds = type(
    "aido_rna_650m_cds",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.RNA-650M-CDS",
    },
)
aido_rna_1b600m = type(
    "aido_rna_1b600m",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.RNA-1.6B",
    },
)
aido_rna_1b600m_cds = type(
    "aido_rna_1b600m_cds",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.RNA-1.6B-CDS",
    },
)
aido_dna_dummy = type(
    "aido_dna_dummy",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.DNA-dummy",
    },
)
aido_dna_300m = type(
    "aido_dna_300m",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.DNA-300M",
    },
)
aido_dna_7b = type(
    "aido_dna_7b",
    (GenBioBERT,),
    {
        "model_path": "genbio-ai/AIDO.DNA-7B",
    },
)
aido_protein_16b = type(
    "aido_protein_16b",
    (GenBioFM,),
    {
        "model_path": "genbio-ai/AIDO.Protein-16B",
    },
)

aido_protein_16b_v1 = type(
    "aido_protein_16b_v1",
    (GenBioFM,),
    {
        "model_path": "genbio-ai/AIDO.Protein-16B-v1",
    },
)

aido_protein_rag_16b = type(
    "aido_protein_rag_16b",
    (GenBioFM,),
    {
        "model_path": "genbio-ai/AIDO.Protein-RAG-16B",
    },
)

aido_protein_rag_3b = type(
    "aido_protein_rag_3b",
    (GenBioFM,),
    {
        "model_path": "genbio-ai/AIDO.Protein-RAG-3B",
    },
)

aido_protein2structoken_16b = type(
    "aido_protein2structoken_16b",
    (GenBioFM,),
    {
        "model_path": "genbio-ai/AIDO.Protein2StructureToken-16B",
    },
)

class aido_protein_debug(GenBioFM):
    """
    A small protein mixture-of-experts transformer model created from scratch for debugging purposes only
    """

    def __init__(self, *args, **kwargs):
        config_overwrites = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "num_experts": 2,
        }
        super().__init__(
            *args, from_scratch=True, config_overwrites=config_overwrites, **kwargs
        )


class aido_dna_debug(GenBioBERT):
    """
    A small dna dense transformer model created from scratch for debugging purposes only
    """

    def __init__(self, *args, **kwargs):
        config_overwrites = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
        }
        super().__init__(
            *args, from_scratch=True, config_overwrites=config_overwrites, **kwargs
        )


dna_onehot = type(
    "onehot",
    (Onehot,),
    {
        "vocab_file": os.path.join(
            Path(__file__).resolve().parent.parent.parent,
            "modelgenerator/huggingface_models/rnabert/vocab.txt",
        ),
    },
)

protein_onehot = type(
    "onehot",
    (Onehot,),
    {
        "vocab_file": os.path.join(
            Path(__file__).resolve().parent.parent.parent,
            "modelgenerator/huggingface_models/fm4bio/vocab_protein.txt",
        ),
    },
)

aido_cell_3m = type(
    "aido_cell_3m",
    (GenBioCellFoundation,),
    {"model_path": "genbio-ai/AIDO.Cell-3M"},
)

aido_cell_10m = type(
    "aido_cell_10m",
    (GenBioCellFoundation,),
    {"model_path": "genbio-ai/AIDO.Cell-10M"},
)

aido_cell_100m = type(
    "aido_cell_100m",
    (GenBioCellFoundation,),
    {"model_path": "genbio-ai/AIDO.Cell-100M"},
)


esm2_8m = type(
    "esm2_8m",
    (ESM,),
    {
        "model_path": "facebook/esm2_t6_8M_UR50D",
    },
)

esm2_35m = type(
    "esm2_35m",
    (ESM,),
    {
        "model_path": "facebook/esm2_t12_35M_UR50D",
    },
)

esm2_150m = type(
    "esm2_150m",
    (ESM,),
    {
        "model_path": "facebook/esm2_t30_150M_UR50D",
    },
)

esm2_650m = type(
    "esm2_650m",
    (ESM,),
    {
        "model_path": "facebook/esm2_t33_650M_UR50D",
    },
)

esm2_3b = type(
    "esm2_3b",
    (ESM,),
    {
        "model_path": "facebook/esm2_t36_3B_UR50D",
    },
)

esm2_15b = type(
    "esm2_15b",
    (ESM,),
    {
        "model_path": "facebook/esm2_t48_15B_UR50D",
    },
)


enformer = type(
    "enformer",
    (Enformer,),
    {
        "model_path": "EleutherAI/enformer-official-rough",
    },
)

borzoi = type(
    "borzoi",
    (Borzoi,),
    {
        "model_path": "johahi/borzoi-replicate-0",
    },
)

flashzoi = type(
    "flashzoi",
    (Borzoi,),
    {
        "model_path": "johahi/flashzoi-replicate-0",
    }
)

scfoundation = type(
    "scfoundation",
    (SCFoundation,),
    {
        "model_path": "genbio-ai/AIDO.scFoundation",
    },
)

aido_tissue_3m = type(
    "aido_tissue_3m",
    (GenBioCellSpatialFoundation,),
    {
        "model_path": "genbio-ai/AIDO.Tissue-3M",
    },
)

aido_tissue_60m = type(
    "aido_tissue_60m",
    (GenBioCellSpatialFoundation,),
    {
        "model_path": "genbio-ai/AIDO.Tissue-60M",
    },
)
