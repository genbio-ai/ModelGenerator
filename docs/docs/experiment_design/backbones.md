# Adding Backbones

Backbones are pre-trained foundation models. 

Foundation models are essential to modern ML but are often difficult to work with.
Design decisions made during pre-training (tokenization, architecture, io format) cannot be changed.
At best, this results in many reimplementations for benchmarking or finetuning tasks, and a high risk of buggy code.
At worst, these decisions can lock in users and exclude certain tasks and use-cases.

AIDO.ModelGenerator eliminates the need for reimplementation and makes backbones task-agnostic: wrap your backbone in a standard interface, and reuse it across all inference and finetuning tasks.
It also makes compatibility transparent: if a backbone fits the required interface, it can be used for any data-appropriate task.

> Note: Backbones for 1D sequence modeling are univerally supported. Other types of backbones included in AIDO.ModelGenerator (e.g. structure, image) are not yet universally supported, but will be in the future.

Available Backbones: 

- DNA: `aido_dna_7b`, `aido_dna_300m`, `aido_dna_dummy`, `aido_dna_debug`, `dna_onehot`
- RNA: `aido_rna_1b600m`, `aido_rna_1b600m_cds`, `aido_rna_650m`, `aido_rna_650m_cds`, `aido_rna_300m_mars`, `aido_rna_25m_mars`, `aido_rna_1m_mars`, `aido_dna_dummy`, `aido_dna_debug`, `dna_onehot`
- Protein: `aido_protein_16b`, `aido_protein_16b_v1`, `aido_protein2structoken_16b`, `aido_protein_debug`, `protein_onehot`, `aido_protein_rag_16b`, `aido_protein_rag_3b`
- Cell (gene expression): `aido_cell_100m`, `aido_cell_10m`, `aido_cell_3m`
- OneHot: dummy model, only tokenizes, useful for non-FM baselines and quick tests

At their core, backbones are PyTorch `nn.Module` objects with a few extra interfaces. 
To implement a new backbone, subclass a backbone interface and implement the required methods.

::: modelgenerator.backbones.SequenceBackboneInterface
    handler: python
    options:
      filters:
        - "!^__"
      show_root_heading: true
      show_source: true