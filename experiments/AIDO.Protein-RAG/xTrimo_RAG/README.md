# Fine-tuning AIDO.Protein-RAG for Protein Downstream Tasks

This document explains how to fine-tune and evaluate our pre-trained protein language models for various downstream tasks from the xTrimoPGLM benchmark. For detailed task descriptions, please refer to our papers:

[Retrieval Augmented Protein Language Models for Protein Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.12.02.626519v1)

[Mixture of Experts Enable Efficient and Effective Protein Understanding and Design](https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1)

**Model Access:**

Download models and task datasets from our Hugging Face repositories:

[AIDO.Protein-RAG-16B](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-16B)

[AIDO.Protein-RAG-3B](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-3B).

Note: All subsequent scripts should be executed from the `ModelGenerator/experiments/AIDO.Protein-RAG/xTrimo_RAG` directory.

## Environment Configuration

There are two approaches to initiate fine-tuning:

1. Use the `run_dms.sh` script to automatically log into designated nodes via SSH and start training
2. Manually launch training using `torchrun`

When using the `run_dms.sh` script, modify the `init_env.sh` file to configure node environment initialization after SSH login.

## Contact Prediction

We fine-tune AIDO.Protein-RAG-16B and AIDO.Protein-RAG-3B for contact prediction using LoRA. Hyperparameter configurations are specified in:

* `contact_prediction_binary_fsdp.yaml`
* `contact_prediction_binary_ddp.yaml`

### Fine-tuning Commands

```bash
# 16B model
./run_xtrimo.sh --nodes node1,node2 configs/contact_prediction_binary_fsdp.yaml contact_AIDO_PROTEIN_RAG_16B.w.str

# 3B model (structure input disabled)
./run_xtrimo.sh --nodes node1,node2 --mask-str \
	--backbone modelgenerator.backbones.aido_protein_rag_3b \
	configs/contact_prediction_binary_fsdp.yaml \
	contact_AIDO_PROTEIN_RAG_3B
```

Alternatively, use `torchrun` directly on each node. Add `--dry-run` to `./run_xtrimo.sh` to preview commands without execution:

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr main_node \
    --master_port 33369 \
    $(which mgen) fit \
        --config configs/contact_prediction_binary_fsdp.yaml \
        --config configs/wandb.yaml \
        --trainer.logger.project xTrimo_Benchmark \
        --trainer.logger.name contact_AIDO_16B.w.str \
        --trainer.logger.id contact_AIDO_16B.w.str_vxdp5ldt \
        --trainer.logger.save_dir logs/xTrimo_Benchmark \
        --trainer.default_root_dir logs/xTrimo_Benchmark \
        --trainer.precision bf16-mixed \
        --trainer.accumulate_grad_batches 1 \
        --trainer.num_nodes 2 \
        --data.batch_size 1
```

### Key Configuration Notes:

1. FSDP with global batch size 16 (8 GPUs per node)
2. Batch size limited to 1 due to 12.8K context length
3. Checkpoints saved to:  `logs/xTrimo_Benchmark/xTrimo_Benchmark/contact_AIDO_16B.w.str`
4. Use `--mask-str` flag to disable structure input

### Evaluation Commands

```bash
# 16B model
./run_xtrimo_eval.sh --node node_name \
    configs/contact_prediction_binary_fsdp.yaml \
    contact_AIDO_16B.w.str

# 3B model (structure input disabled)
./run_xtrimo_eval.sh --node node_name --mask-str \
    --backbone modelgenerator.backbones.aido_protein_rag_3b \
    configs/contact_prediction_binary_fsdp.yaml \
    contact_AIDO_3B
```

## Secondary structure prediction

Fine-tuning configuration details in `ssp_q3.yaml`.

### Fine-tuning Commands

```bash
# 16B model
./run_xtrimo.sh --nodes node1,node2 configs/ssp_q3.yaml ssp3_AIDO_16B.w.str

# 3B model (structure input disabled)
./run_xtrimo.sh --nodes node1,node2 --mask-str \
    --backbone modelgenerator.backbones.aido_protein_rag_3b \
    configs/ssp_q3.yaml ssp3_AIDO_3B
```

### Evaluation Commands

```bash
# 16B model
./run_xtrimo_eval.sh --node node_name \
    configs/ssp_q3.yaml \
    ssp3_AIDO_16B.w.str

# 3B model (structure input disabled)
./run_xtrimo_eval.sh --node node_name --mask-str \
    --backbone modelgenerator.backbones.aido_protein_rag_3b \
    configs/ssp_q3.yaml \
    ssp3_AIDO_3B
```

## Sequence-Level Classification: Fold Prediction

Configuration details in `fold_prediction.yaml`.

### Fine-tuning Commands

```bash
# 16B model
./run_xtrimo.sh --nodes node1,node2 configs/fold_prediction.yaml fold_AIDO_16B.w.str

# 3B model (structure input disabled)
./run_xtrimo.sh --nodes node1,node2 --mask-str \
    --backbone modelgenerator.backbones.aido_protein_rag_3b \
    configs/fold_prediction.yaml fold_AIDO_3B
```

### Evaluation Commands

```bash
# 16B model
./run_xtrimo_eval.sh --node node_name configs/fold_prediction.yaml fold_AIDO_16B.w.str

# 3B model (structure input disabled)
./run_xtrimo_eval.sh --node node_name --mask-str \
    --backbone modelgenerator.backbones.aido_protein_rag_3b \
    configs/fold_prediction.yaml fold_AIDO_3B
```

## Sequence-level Regression: Fluorescence Prediction

Configuration details in `fluorescence_prediction.yaml`.

### Fine-tuning Commands

```bash
# 16B model
./run_xtrimo.sh --nodes node1,node2 configs/fluorescence_prediction.yaml fluo_AIDO_16B.w.str

# 3B model (structure input disabled)
./run_xtrimo.sh --nodes node1,node2 --mask-str \
    --backbone modelgenerator.backbones.aido_protein_rag_3b \
    configs/fluorescence_prediction.yaml fluo_AIDO_3B
```

### Evaluation Commands

```bash
# 16B model
./run_xtrimo_eval.sh --node node_name configs/fluorescence_prediction.yaml fluo_AIDO_16B.w.str

# 3B model (structure input disabled)
./run_xtrimo_eval.sh --node node_name --mask-str \
    --backbone modelgenerator.backbones.aido_protein_rag_3b \
    configs/fluorescence_prediction.yaml fluo_AIDO_3B
```

