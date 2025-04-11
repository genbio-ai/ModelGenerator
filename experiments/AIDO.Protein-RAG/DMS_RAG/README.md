# ProteinGym DMS Benchmark

The Deep Mutational Scanning (DMS) Benchmark in ProteinGym is a comprehensive collection of 283 standardized DMS assays containing over 2.7 million mutated protein sequences spanning more than 200 diverse protein families. These assays capture various functional properties including ligand binding, thermostability, viral replication, drug resistance, and other critical biological characteristics. The dataset encompasses diverse taxonomic groups (humans, other eukaryotes, prokaryotes, and viruses) and includes multiple mutation types such as single amino acid substitutions and indels (insertions/deletions).

The primary objective of the DMS Benchmark is to model protein fitness landscapes - the complex relationships between genetic mutations and their impacts on protein fitness and functionality. We have fine-tuned [AIDO.Protein-RAG-16B](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-16B) models using this benchmark. The implementation in AIDO.ModelGenerator employs LoRA fine-tuning with a 5-fold cross-validation scheme, following the random split strategy described in the original [ProteinGym paper](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1). For efficient training, the system utilizes Distributed Data Parallel (DDP).

Note:​​ All subsequent scripts should be executed within the `ModelGenerator/experiments/AIDO.Protein-RAG/DMS_RAG` directory.

## Environment Configuration

There are two approaches to initiate fine-tuning:

1. Use the `run_dms.sh` script to automatically log into designated nodes via SSH and start training
2. Manually launch training using `torchrun`

When using the `run_dms.sh` script, modify the `init_env.sh` file to configure node environment initialization after SSH login.

## Fine-Tuning Procedure

To perform 5-fold cross-validation fine-tuning:

```bash
# AIDO.Protein-RAG-16B
./run_dms.sh --nodes node1,node2 Q2N0S5_9HIV1_Haddox_2018

# AIDO.Protein-RAG-3B (without structural input support)
./run_dms.sh --nodes node1,node2 --mask-str --backbone modelgenerator.backbones.aido_protein_rag_3b Q2N0S5_9HIV1_Haddox_2018
```

Alternatively, use `torchrun` directly on each node. Add `--dry-run` to `./run_xtrimo.sh` to preview commands without execution:

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr main_node \
    --master_port 34381 \
    $(which mgen) fit \
        --config configs/substitution_LoRA_DDP.yaml \
        --config configs/wandb.yaml \
        --trainer.logger.project DMS_Benchmark \
        --trainer.logger.name Q2N0S5_9HIV1_Haddox_2018_fold0 \
        --trainer.logger.id Q2N0S5_9HIV1_Haddox_2018_fold0_98hto81p \
        --trainer.default_root_dir logs/DMS_Benchmark \
        --trainer.logger.save_dir logs/DMS_Benchmark \
        --data.train_split_files [singles_substitutions/Q2N0S5_9HIV1_Haddox_2018.tsv] \
        --data.cv_test_fold_id 0 \
        --trainer.precision bf16-mixed \
        --trainer.accumulate_grad_batches 1 \
        --trainer.num_nodes 2 \
        --data.batch_size 1 \
        --model.init_args.backbone.init_args.config_overwrites.str_embedding_in 384 \
        --trainer.callbacks.patience 10
```

## Evaluation Protocol

To evaluate using 5-fold cross-validation:

```bash
# AIDO.Protein-RAG-16B
./run_dms_eval.sh --node node_name Q2N0S5_9HIV1_Haddox_2018

# AIDO.Protein-RAG-3B
./run_dms_eval.sh --node node_name --mask-str --backbone modelgenerator.backbones.aido_protein_rag_3b Q2N0S5_9HIV1_Haddox_2018
```

