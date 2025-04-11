# AIDO.Protein-RAG

AIDO.RAGProtein-16B is a multimodal protein language model that integrates Multiple Sequence Alignment (MSA) and structural data, building upon the [AIDO.Protein-16B](https://huggingface.co/genbio-ai/AIDO.Protein-16B) foundation. The training process comprises three main stages:

1. 2D RoPE encoding fine-tuning
2. Initial training on 100 billion tokens from UniRef50/UniClust30 MSA data
3. Subsequent training on 80 billion tokens from AlphaFold Database MSA and structural data

For more details, please refer to the [AIDO.Protein-RAG](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-16B) model card, and our manuscipts

1. [Retrieval Augmented Protein Language Models for Protein Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.12.02.626519v1)

2. [Mixture of Experts Enable Efficient and Effective Protein Understanding and Design](https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1)


## Evaluation

### ProteinGym DMS Benchmark

The Deep Mutational Scanning (DMS) Benchmark in ProteinGym is a comprehensive collection of 283 standardized DMS assays containing over 2.7 million mutated protein sequences spanning more than 200 diverse protein families. These assays capture various functional properties including ligand binding, thermostability, viral replication, drug resistance, and other critical biological characteristics. The dataset encompasses diverse taxonomic groups (humans, other eukaryotes, prokaryotes, and viruses) and includes multiple mutation types such as single amino acid substitutions and indels (insertions/deletions).

The primary objective of the DMS Benchmark is to model protein fitness landscapes - the complex relationships between genetic mutations and their impacts on protein fitness and functionality. We have fine-tuned [AIDO.Protein-RAG-16B](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-16B) models using this benchmark, and uploaded a new version with MSAs and embeddings from AIDO.StructureTokenizer to [ProteinGYM-DMS-RAG](https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS-RAG) on HuggingFace.

The implementation in AIDO.ModelGenerator employs LoRA fine-tuning with a 5-fold cross-validation scheme, following the random split strategy described in the original [ProteinGym paper](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1). For efficient training, the system utilizes Distributed Data Parallel (DDP).

To reproduce, simply use
```bash
mgen fit \
    --config <DMS_RAG/configs/*.yaml> \
    --trainer.precision bf16-mixed \
    --trainer.accumulate_grad_batches 1 \
    --trainer.num_nodes 1 \
    --data.batch_size 1 \
    --model.init_args.backbone.init_args.config_overwrites.str_embedding_in 384 \
    --model.init_args.backbone.class_path "aido_protein_rag_16b" \
    --model.init_args.backbone.init_args.max_length 12800 \
    --data.init_args.max_context_length 12800 \
    --trainer.log_every_n_steps 1 \
    --trainer.profiler null \
    --trainer.devices 1
```

### xTrimoPGLM Benchmarks

We also fine-tune and evaluate our pre-trained protein language models for various downstream tasks from the xTrimoPGLM benchmark, including

1. [Contact Prediction](https://huggingface.co/datasets/genbio-ai/contact_prediction_binary_rag)
2. [Fluorescence Prediction](https://huggingface.co/datasets/genbio-ai/fluorescence_prediction_rag)
3. [SSP-Q3](https://huggingface.co/datasets/genbio-ai/ssp_q3_rag)
4. [Fold Prediction](https://huggingface.co/datasets/genbio-ai/fold_prediction_rag)

To reproduce these results, simply use
```bash
mgen fit \
    --config <xTrimo_RAG/configs/*.yaml> \
    --trainer.precision bf16-mixed \
    --trainer.accumulate_grad_batches 1 \
    --trainer.num_nodes 1 \
    --data.batch_size 1 \
    --model.init_args.backbone.init_args.config_overwrites.str_embedding_in 384 \
    --model.init_args.backbone.class_path "aido_protein_rag_16b" \
    --model.init_args.backbone.init_args.max_length 12800 \
    --data.init_args.max_context_length 12800 \
    --trainer.log_every_n_steps 1 \
    --trainer.profiler null \
    --trainer.devices 1
```