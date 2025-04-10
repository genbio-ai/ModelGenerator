# ProteinGym DMS Benchmark
The Deep Mutational Scanning (DMS) Benchmark in ProteinGym is a comprehensive collection of 283 standardized DMS assays, comprising more than 2.7 million mutated protein sequences from over 200 diverse protein families. These assays capture a wide range of functional properties, such as ligand binding, thermostability, viral replication, drug resistance, and more. The dataset spans diverse taxa (humans, other eukaryotes, prokaryotes, and viruses) and includes a variety of mutation types, such as single amino acid substitutions and indels (insertions or deletions). The primary goal of the DMS Benchmark is to model protein fitness landscapes, which represent the relationship between genetic mutations and their effects on protein fitness or functionality.
We finetune the [AIDO.Protein-16B](https://huggingface.co/genbio-ai/AIDO.DNA-16B) and [AIDO.Protein-16B-v1](https://huggingface.co/genbio-ai/AIDO.DNA-16B-v1) models on the DMS benmark.
AIDO.ModelGenerator implements both Linear Probing and LoRA finetuning, following a 5-fold cross-validation scheme with the random split strategy proposed in the original [ProteinGym paper](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1). ModelGenerator also implements both DDP and FSDP for efficient finetuning.

For tasks involving indels with limited data, we apply Linear Probing, while LoRA is used for substitution tasks and other indel tasks to balance computational efficiency with fine-tuning effectiveness.

To finetune with 5-fold cross validation, run
```
sbatch train_script.sh
```
where the `train_script.sh` is
```
mgen fit --config config.yaml \
    --data.train_split_files <list of train files> \ # For example, ["B1LPA6_ECOSM_Russ_2020_indels.tsv"]
    --data.cv_test_fold_id ${FOLD} \
```
To test with specific ckpt, run
```
sbatch test_script.sh
```
where the `test_script.sh` is
```
mgen test --config config.yaml \
    --model.strict_load false \
    --data.train_split_files <list of train files> \ # For example, ["B1LPA6_ECOSM_Russ_2020_indels.tsv"]
    --data.cv_test_fold_id ${FOLD} \
    --ckpt_path <you choose>
```
The `config.yaml` with LoRA finetuneing and DDP shoud be
```
trainer:
  strategy:
    class_path: lightning.pytorch.strategies.DDPStrategy
  callbacks:
  - class_path: lightning.pytorch.callbacks.early_stopping.EarlyStopping
    dict_kwargs:
      monitor: val_spearman
      mode: max
      patience: <you choose>
  - class_path: modelgenerator.callbacks.PredictionWriter
    dict_kwargs:
      output_dir: dms_prediction
      filetype: tsv
      write_cols: ['predictions','labels']
data:
  class_path: modelgenerator.data.DMSFitnessPrediction
  init_args:
    path: genbio-ai/ProteinGYM-DMS
    train_split_files:
    - <you choose>
    normalize: true
    train_split_name: train
    test_split_files: null
    valid_split_files: null
    random_seed: 42
    cv_num_folds: 5
    cv_test_fold_id: 0
    cv_enable_val_fold: true
    cv_fold_id_col: fold_id
model:
  class_path: modelgenerator.tasks.SequenceRegression
  init_args:
    backbone:
      class_path: <you choose>
      init_args:
        from_scratch: false
        use_peft: true
        save_peft_only: true
        lora_r: 16
        lora_alpha: 32
        lora_dropout: 0.05
    adapter:
      class_path: modelgenerator.adapters.MLPPoolAdapter
      init_args:
        pooling: mean_pooling
        hidden_sizes:
        - 128
        bias: true
        dropout: 0.1
        dropout_in_middle: false
```
If you want to use FSDP, the `trainer.strategy` should change to
```
trainer:
  strategy:
    class_path: lightning.pytorch.strategies.FSDPStrategy
    init_args:
      auto_wrap_policy: modelgenerator.distributed.fsdp.wrap.AutoWrapPolicy
      sharding_strategy: HYBRID_SHARD
```

Here are three examples demonstrating how to load finetuned HF models for inference, utilizing either FSDP or DDP, with support for LoRA or Linear Probing.
1. [FSDP + LoRA]("https://huggingface.co/genbio-ai/AIDO.Protein-16B-dms-substitutions-CP2C9_HUMAN_Amorosi_2021_abundance")
```
mgen test --config configs/substitution_LoRA_FSDP.yaml --ckpt_path "genbio-ai/AIDO.Protein-16B-dms-substitutions-CP2C9_HUMAN_Amorosi_2021_abundance/fold0/model.ckpt"
```
2. [DDP + LoRA]("https://huggingface.co/genbio-ai/AIDO.Protein-16B-dms-substitutions-LGK_LIPST_Klesmith_2015")
```
mgen test --config configs/substitution_LoRA_DDP.yaml --ckpt_path "genbio-ai/AIDO.Protein-16B-dms-substitutions-LGK_LIPST_Klesmith_2015/fold0/model.ckpt"
```
3. [DDP + LinearProbing]("https://huggingface.co/genbio-ai/AIDO.Protein-16B-dms-indels-FECA_ECOLI_Tsuboyama_2023_2D1U_indels")
```
mgen test --config configs/indels_LP_DDP.yaml --ckpt_path "genbio-ai/AIDO.Protein-16B-dms-indels-FECA_ECOLI_Tsuboyama_2023_2D1U_indels/fold0/model.ckpt"
```