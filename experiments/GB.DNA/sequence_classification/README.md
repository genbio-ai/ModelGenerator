# Sequence Classification Task
We apply the [GB.DNA-7B](https://huggingface.co/genbio-ai/GB.DNA-7B) models to sequence classification and property prediction tasks, using standard classification benchmarks from prominent works on DNA encoders covering a breadth of genomic functions related to transcriptional regulation and transcript processing.
GB.ModelGenerator implements two related benchmarks: Genome Understanding Evaluation proposed by [DNABERT-2](https://arxiv.org/abs/2306.15006) and Nucleotide Transformer Benchmark proposed by [Nucleotide Transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1)

We use LoRA finetuning with a CLS or MeanPool Adapter to finetune our model on these two benchmarks.

To finetune, run
```
mgen fit --config config.yaml
```
To test with specific ckpt, run
```
mgen test --config config.yaml --ckpt_path=<selected_ckpt_path> --model.strict_loading=false
```
where the config should be
```
model:
  class_path: modelgenerator.tasks.SequenceClassification
  init_args:
    backbone:
      class_path: <you-choose>
      init_args:
        from_scratch: false
        use_peft: true
        save_peft_only: true
        lora_r: 16
        lora_alpha: 32
        lora_dropout: 0.1
    adapter: <you choose> # For example, modelgenerator.adapters.LinearCLSAdapter
data: # depend on benchmark
trainer:
  callbacks:
  - class_path: lightning.pytorch.callbacks.early_stopping.EarlyStopping
    dict_kwargs:
      monitor: val_mcc
      mode: max
      patience: <you choose>
```
For GUE benchmark, the config.data should be
```
data:
  class_path: modelgenerator.data.GUEClassification
  init_args:
    config_name: <you choose>
    x_col: sequence
    y_col: label
    train_split_name: train
    test_split_name: test
    valid_split_name: null
    valid_split_size: 0.1 # randomly split valid set from train set
```
For NT benchmark, the config.data should be
```
data:
  class_path: modelgenerator.data.NTClassification
  init_args:
    config_name: enhancers
    x_col: sequence
    y_col: label
    train_split_name: train
    test_split_name: test
    valid_split_name: null
    valid_split_size: 0.1
```
Here are some examples of how to load HF model for inference
1. NT Benchmark: [enhancers](https://huggingface.co/genbio-ai/GB.DNA-7B-nt-enhancers)
```
mgen test --config nt_enhancers.yaml --ckpt_path "genbio-ai/GB.DNA-7B-nt-enhancers/model.ckpt"
```
2. NT Benchmark: [promoter_all](https://huggingface.co/genbio-ai/GB.DNA-7B-nt-promoter-all)
```
mgen test --config nt_promoter_all.yaml --ckpt_path "genbio-ai/GB.DNA-7B-nt-promoter-all/model.ckpt"
```
3. GUE Benchmark: [core_promoter_all](https://huggingface.co/genbio-ai/GB.DNA-7B-gue-core-promoter-all)
```
mgen test --config gue_core_promoter_all.yaml --ckpt_path "genbio-ai/GB.DNA-7B-gue-core-promoter-all/model.ckpt"
```
4. GUE Benchmark: [splice_reconsturction](https://huggingface.co/genbio-ai/GB.DNA-7B-gue-splice-reconstruction)
```
mgen test --config gue_splice_reconstruction.yaml --ckpt_path "genbio-ai/GB.DNA-7B-gue-splice-reconstruction/model.ckpt"
```
